from __future__ import annotations

from typing import Any, NamedTuple, Callable, Optional

import jax
import jax.numpy as jnp
import optax
from jax import tree_util as jtu

from .ggn_utils import GGNMatvecFn

Array = jax.Array
PyTree = Any


# -------------------------
# Small PyTree linear algebra
# -------------------------

def _tree_add(x: PyTree, y: PyTree) -> PyTree:
    return jtu.tree_map(lambda a, b: a + b, x, y)


def _tree_add_scaled(x: PyTree, y: PyTree, alpha: Array) -> PyTree:
    return jtu.tree_map(lambda a, b: a + alpha * b, x, y)


def _tree_scale(x: PyTree, alpha: Array) -> PyTree:
    return jtu.tree_map(lambda a: alpha * a, x)


def _tree_zeros_like(x: PyTree) -> PyTree:
    return jtu.tree_map(jnp.zeros_like, x)


def _tree_dot(x: PyTree, y: PyTree) -> Array:
    """Inner product ⟨x, y⟩ over a PyTree."""
    xs, ys = jtu.tree_leaves(x), jtu.tree_leaves(y)
    return sum(jnp.vdot(a, b) for a, b in zip(xs, ys))


def _tree_where(mask: Array, x_old: PyTree, x_new: PyTree) -> PyTree:
    """Choose between two PyTrees based on a scalar boolean mask."""
    return jtu.tree_map(lambda o, n: jnp.where(mask, o, n), x_old, x_new)


# -------------------------
# Conjugate gradient solver
# -------------------------

def _cg_solve(
    matvec: Callable[[PyTree], PyTree],
    b: PyTree,
    max_iters: int,
    tol_sq: Array,
) -> PyTree:
    """Solve (B x = b) with (precondition-free) conjugate gradient in PyTree space.

    Args:
      matvec: function v -> B v.
      b: right hand side (PyTree), same structure as params/grads.
      max_iters: maximum CG iterations.
      tol_sq: absolute tolerance on ||r||^2 (scalar).

    Returns:
      x: approximate solution of B x = b, as PyTree.
    """
    x0 = _tree_zeros_like(b)
    r0 = b  # since x0 = 0, r = b - Bx = b
    p0 = r0
    rs0 = _tree_dot(r0, r0)

    def body_fun(i, carry):
        x, r, p, rs, done = carry

        Bp = matvec(p)
        pBp = _tree_dot(p, Bp)
        alpha = rs / (pBp + 1e-12)

        x_new = _tree_add_scaled(x, p, alpha)
        r_new = _tree_add_scaled(r, Bp, -alpha)
        rs_new = _tree_dot(r_new, r_new)

        beta = rs_new / (rs + 1e-12)
        p_new = _tree_add_scaled(r_new, p, beta)

        # Convergence check on residual
        done_now = rs_new < tol_sq
        done_new = jnp.logical_or(done, done_now)

        # Once done, keep state frozen
        x = _tree_where(done, x, x_new)
        r = _tree_where(done, r, r_new)
        p = _tree_where(done, p, p_new)
        rs = jnp.where(done, rs, rs_new)

        return (x, r, p, rs, done_new)

    init_state = (x0, r0, p0, rs0, jnp.array(False))
    x_final, _, _, _, _ = jax.lax.fori_loop(0, max_iters, body_fun, init_state)
    return x_final


# -------------------------
# Lanczos for top eigenvalues (no eigenvectors)
# -------------------------

def _lanczos_eigenvalues(
    matvec: Callable[[Array], Array],
    dim: int,
    num_iter: int,
    key: Array,
    eps: float = 1e-6,
) -> Array:
    """Approximate top eigenvalues of a symmetric PSD operator via Lanczos.

    We never lift eigenvectors back to parameter space; we only use the
    eigenvalues of the tridiagonal T (Ritz values).

    Args:
      matvec: function v -> A v, with A symmetric PSD (e.g., GGN).
      dim: dimension of the flattened parameter vector.
      num_iter: Lanczos iterations / Krylov subspace size.
      key: PRNG key.
      eps: small numerical guard.

    Returns:
      eigenvalues: (num_iter,) approximate eigenvalues (sorted descending).
    """
    v0 = jax.random.normal(key, (dim,))
    v0 = v0 / (jnp.linalg.norm(v0) + eps)

    def body_fun(carry, i):
        V, alphas, betas = carry
        v = V[i]  # (dim,)

        w = matvec(v)  # (dim,)
        alpha = jnp.vdot(v, w)
        w = w - alpha * v

        def ortho_against_prev(_w, _i):
            prev_v = V[_i]
            proj = jnp.vdot(prev_v, _w)
            return _w - proj * prev_v

        # Full reorthogonalization (OK for small num_iter)
        w = jax.lax.fori_loop(0, i, lambda j, ww: ortho_against_prev(ww, j), w)

        beta = jnp.linalg.norm(w)
        beta = jnp.where(beta < eps, 0.0, beta)
        next_v = jnp.where(beta > 0, w / (beta + eps), jnp.zeros_like(w))

        V = V.at[i + 1].set(next_v)
        alphas = alphas.at[i].set(alpha)
        betas = betas.at[i].set(beta)

        return (V, alphas, betas), None

    V = jnp.zeros((num_iter + 1, dim))
    V = V.at[0].set(v0)
    alphas = jnp.zeros((num_iter,))
    betas = jnp.zeros((num_iter,))

    (V, alphas, betas), _ = jax.lax.scan(
        body_fun,
        (V, alphas, betas),
        jnp.arange(num_iter),
    )

    k = num_iter
    T = jnp.diag(alphas)
    if k > 1:
        T = T.at[jnp.arange(k - 1), jnp.arange(1, k)].set(betas[: k - 1])
        T = T.at[jnp.arange(1, k), jnp.arange(k - 1)].set(betas[: k - 1])

    evals, _ = jnp.linalg.eigh(T)  # ascending
    idx = jnp.argsort(evals)[::-1]
    evals = evals[idx]
    return evals


# -------------------------
# Optax transformation
# -------------------------

class HFState(NamedTuple):
    count: Array                 # step counter
    damping: Array               # scalar λ
    eigenvalues: Array           # (max_eigenvalues,)
    rng_key: Array               # PRNG key for Lanczos


def hessian_free(
    ggn_matvec_fn: GGNMatvecFn,
    learning_rate: float = 1.0,
    weight_decay: float = 0.0,
    damping: float = 1e-3,
    cg_max_iters: int = 50,
    cg_tol: float = 1e-4,
    # NEW: curvature / spectrum tracking
    curvature_update_every: int = 50,
    max_eigenvalues: int = 16,
    lanczos_iters: Optional[int] = None,
) -> optax.GradientTransformation:
    """Hessian-free / Newton-CG optimizer using Gauss-Newton curvature.

    We approximately solve, at each step:

        (G + λ I) p ≈ -g

    where:
      - g is the (possibly L2-regularized) gradient,
      - G is the GGN matrix, provided by `ggn_matvec_fn`,
      - λ is a (fixed) damping parameter (Levenberg–Marquardt style).

    Then we update:

        params <- params + η p

    Additionally, we *optionally* track the top-k eigenvalues of G (Ritz values)
    using Lanczos on a *flattened* matvec, without storing any eigenvectors.

    Args:
      ggn_matvec_fn: callable (params, vec_pytree, rng_key) -> Hv_pytree.
      learning_rate: scalar step size η applied to the Newton direction.
      weight_decay: L2 regularization coefficient (added to the gradient).
      damping: curvature damping λ (adds λ I to the GGN).
      cg_max_iters: maximum CG iterations per outer step.
      cg_tol: relative tolerance on the CG residual (for ||r||^2).
      curvature_update_every: how often to re-estimate eigenvalues.
      max_eigenvalues: number of eigenvalues to store in the state.
      lanczos_iters: Lanczos iterations for spectrum estimation;
        defaults to `max_eigenvalues`.

    Returns:
      An `optax.GradientTransformation` implementing HF with optional
      eigenvalue tracking.
    """

    lr = float(learning_rate)
    wd = float(weight_decay)
    lam0 = float(damping)
    cg_tol = float(cg_tol)

    if lanczos_iters is None:
        lanczos_iters = max_eigenvalues

    def init_fn(params: PyTree) -> HFState:
        eigenvalues = jnp.zeros((max_eigenvalues,), dtype=jnp.float32)
        rng_key = jax.random.PRNGKey(0)

        return HFState(
            count=jnp.zeros([], dtype=jnp.int32),
            damping=jnp.array(lam0, dtype=jnp.float32),
            eigenvalues=eigenvalues,
            rng_key=rng_key,
        )

    def update_fn(
        grads: PyTree,
        state: HFState,
        params: PyTree | None = None,
    ) -> tuple[PyTree, HFState]:
        assert params is not None, "Hessian-free optimizer requires `params`."

        # Step counter
        count = state.count + 1
        lam = state.damping
        eigenvalues = state.eigenvalues
        rng_key = state.rng_key

        # 1) Optionally add L2 regularization: g <- g + wd * θ
        if wd != 0.0:
            grads_reg = jtu.tree_map(
                lambda g, p: g + wd * p, grads, params
            )
        else:
            grads_reg = grads

        # 2) Build curvature operator B(v) = G v + λ v using GGN matvec.
        dummy_rng = jnp.array(0.0, dtype=jnp.float32)

        def B(v: PyTree) -> PyTree:
            Hv = ggn_matvec_fn(params, v, dummy_rng)
            return jtu.tree_map(lambda hv, vv: hv + lam * vv, Hv, v)

        # 3) Right-hand side b = -g (we want B p ≈ -g).
        b = jtu.tree_map(lambda g: -g, grads_reg)

        # 4) If gradient is (almost) zero, skip CG and return zero updates.
        norm_b2 = _tree_dot(b, b)

        def solve_nonzero(_):
            # Relative tolerance on ||r||^2
            tol_sq = (cg_tol ** 2) * norm_b2
            p = _cg_solve(B, b, cg_max_iters, tol_sq)
            return p

        def solve_zero(_):
            return _tree_zeros_like(b)

        p = jax.lax.cond(
            norm_b2 > 0.0,
            solve_nonzero,
            solve_zero,
            operand=None,
        )

        # 5) Newton step (no extra minus: p already solves B p ≈ -g).
        updates = _tree_scale(p, lr)

        # 6) Optionally re-estimate top eigenvalues of G via Lanczos
        #    every `curvature_update_every` steps.
        def do_curvature_update(carry):
            eigenvalues, rng_key = carry

            flat_params, unravel = jax.flatten_util.ravel_pytree(params)
            dim = flat_params.shape[0]

            def matvec_flat(v_flat: Array) -> Array:
                v_pytree = unravel(v_flat)
                Hv_pytree = ggn_matvec_fn(params, v_pytree, rng_key)
                Hv_flat, _ = jax.flatten_util.ravel_pytree(Hv_pytree)
                return Hv_flat

            rng_key, subkey = jax.random.split(rng_key)
            evals = _lanczos_eigenvalues(
                matvec=matvec_flat,
                dim=dim,
                num_iter=lanczos_iters,
                key=subkey,
            )

            k = min(max_eigenvalues, evals.shape[0])
            new_eigs = jnp.zeros_like(eigenvalues)
            new_eigs = new_eigs.at[:k].set(evals[:k])
            return (new_eigs, rng_key)

        def skip_curvature_update(carry):
            return carry

        do_update = (curvature_update_every > 0) & (
            (count % curvature_update_every) == 0
        )

        eigenvalues, rng_key = jax.lax.cond(
            do_update,
            do_curvature_update,
            skip_curvature_update,
            operand=(eigenvalues, rng_key),
        )

        new_state = HFState(
            count=count,
            damping=lam,
            eigenvalues=eigenvalues,
            rng_key=rng_key,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
