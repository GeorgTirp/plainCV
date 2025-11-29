# optim/hessian_free.py
from __future__ import annotations

from typing import Any, NamedTuple, Callable

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

        # Standard CG update (Martens 2010 HF / Newton-CG). 
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
# Optax transformation
# -------------------------

class HFState(NamedTuple):
    count: Array
    damping: Array  # scalar λ


def hessian_free(
    ggn_matvec_fn: GGNMatvecFn,
    learning_rate: float = 1.0,
    weight_decay: float = 0.0,
    damping: float = 1e-3,
    cg_max_iters: int = 50,
    cg_tol: float = 1e-4,
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

    Args:
      ggn_matvec_fn: callable (params, vec_pytree, rng_key) -> Hv_pytree,
        typically built via `make_ggn_matvec_fn`.
      learning_rate: scalar step size η applied to the Newton direction.
      weight_decay: L2 regularization coefficient (added to the gradient).
      damping: curvature damping λ (adds λ I to the GGN).
      cg_max_iters: maximum CG iterations per outer step.
      cg_tol: relative tolerance on the CG residual (for ||r||^2).

    Returns:
      An `optax.GradientTransformation` implementing HF.
    """

    lr = float(learning_rate)
    wd = float(weight_decay)
    lam = float(damping)

    def init_fn(params: PyTree) -> HFState:
        return HFState(
            count=jnp.zeros([], dtype=jnp.int32),
            damping=jnp.array(lam, dtype=jnp.float32),
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

        # 1) Optionally add L2 regularization: g <- g + wd * θ
        if wd != 0.0:
            grads_reg = jtu.tree_map(
                lambda g, p: g + wd * p, grads, params
            )
        else:
            grads_reg = grads

        # 2) Build curvature operator B(v) = G v + λ v using GGN matvec.
        dummy_rng = jnp.array(0.0, dtype=jnp.float32)  # not used inside GGN

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

        new_state = HFState(count=count, damping=lam)
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
