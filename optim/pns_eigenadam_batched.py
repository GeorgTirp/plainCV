# optim/pns_eigenadam_batched.py
"""
PN-S EigenAdam with a *batched* curvature eigensolver.

Key change vs classic Lanczos:
- Uses a block/subspace method (orthogonal iteration + Rayleigh–Ritz) so each
  curvature refresh produces k eigen-directions while evaluating HVPs in a
  single vmapped/batched call: H @ V for V in R^{dim x k}.

Notes:
- This does NOT make the total compute "free": it still costs ~k HVP work, but
  it batches that work (better accelerator utilization + lower Python overhead).
- The block eigensolver is meant for top-k PSD curvature (GGN/Fisher) or for
  Hessian/Fisher with |λ| sorting (saddle-free style).
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import tree_util as jtu
import optax

Array = jax.Array
Params = Any
PyTree = Any

# Type: given params, direction (same PyTree structure as params), rng -> matvec(direction)
GGNMatvecFn = Callable[[Params, PyTree, Array], PyTree]


# ---------------------------------------------------------------------------
# Utilities: flatten batched pytrees and block eigensolver
# ---------------------------------------------------------------------------

def ravel_pytree_batched(tree_b: PyTree) -> Array:
    """Flatten a batched pytree (leading axis = batch) into (b, dim).

    Assumes every leaf has the same leading batch dimension b.
    """
    leaves, _ = jtu.tree_flatten(tree_b)
    flats = [jnp.reshape(x, (x.shape[0], -1)) for x in leaves]  # (b, leaf_dim)
    return jnp.concatenate(flats, axis=-1)  # (b, dim)


def block_orthogonal_iteration(
    matvec_batch: Callable[[Array], Array],  # (k, dim) -> (k, dim)
    dim: int,
    k: int,
    num_iter: int,
    key: Array,
    eps: float = 1e-6,
    sort_by_abs: bool = False,
) -> Tuple[Array, Array]:
    """Approximate top-k eigenpairs of a symmetric operator using subspace iteration.

    Each iteration computes H @ Q where Q has k columns, by calling matvec_batch on Q^T.

    Args:
      matvec_batch: batched matvec that maps (k, dim) vectors to (k, dim).
      dim: flattened parameter dimension.
      k: number of modes to return.
      num_iter: number of block power iterations (typically small: 2–8).
      key: RNG key for initial subspace.
      eps: numerical guard.
      sort_by_abs: if True, sort Ritz values by |λ| descending.

    Returns:
      evals: (k,) descending.
      evecs: (k, dim) eigenvectors as rows (to match lanczos() shape in your code).
    """
    # Start with random subspace, orthonormal columns
    Q0 = jax.random.normal(key, (dim, k))
    Q0, _ = jnp.linalg.qr(Q0)

    def one_iter(_, Q):
        # Apply H to each column of Q via batching over rows of Q.T
        HQ = matvec_batch(Q.T)        # (k, dim)
        HQ = HQ.T                     # (dim, k)
        Qn, _ = jnp.linalg.qr(HQ)     # re-orthonormalize
        return Qn[:, :k]

    Q = jax.lax.fori_loop(0, num_iter, one_iter, Q0)

    # Rayleigh–Ritz: solve small kxk eigenproblem in the subspace
    HQ = matvec_batch(Q.T).T         # (dim, k)
    B = Q.T @ HQ                     # (k, k)
    B = 0.5 * (B + B.T)              # symmetrize

    evals, Z = jnp.linalg.eigh(B)    # ascending
    if sort_by_abs:
        idx = jnp.argsort(jnp.abs(evals))[::-1]
    else:
        idx = jnp.argsort(evals)[::-1]

    evals = evals[idx]
    Z = Z[:, idx]

    U = Q @ Z                        # (dim, k)
    evecs = U.T                      # (k, dim) rows
    return evals, evecs


# ---------------------------------------------------------------------------
# (Optional) Classic single-vector Lanczos retained as a fallback
# ---------------------------------------------------------------------------

def lanczos(
    matvec: Callable[[Array], Array],
    dim: int,
    num_iter: int,
    key: Array,
    eps: float = 1e-6,
    sort_by_abs: bool = False,
) -> Tuple[Array, Array]:
    """Run (single-vector) Lanczos to approximate top eigenvalues/eigenvectors.

    Returns:
      eigenvalues: (num_iter,) sorted.
      eigenvectors: (num_iter, dim) eigenvectors as rows.
    """
    v0 = jax.random.normal(key, (dim,))
    v0 = v0 / (jnp.linalg.norm(v0) + eps)

    def body_fun(carry, i):
        V, alphas, betas = carry
        v = V[i]  # (dim,)

        w = matvec(v)
        alpha = jnp.vdot(v, w)
        w = w - alpha * v

        def ortho_against_prev(_w, _i):
            prev_v = V[_i]
            proj = jnp.vdot(prev_v, _w)
            return _w - proj * prev_v

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

    (V, alphas, betas), _ = jax.lax.scan(body_fun, (V, alphas, betas), jnp.arange(num_iter))

    k = num_iter
    T = jnp.diag(alphas)
    if k > 1:
        T = T.at[jnp.arange(k - 1), jnp.arange(1, k)].set(betas[: k - 1])
        T = T.at[jnp.arange(1, k), jnp.arange(k - 1)].set(betas[: k - 1])

    evals, evecs_T = jnp.linalg.eigh(T)  # ascending
    if sort_by_abs:
        idx = jnp.argsort(jnp.abs(evals))[::-1]
    else:
        idx = jnp.argsort(evals)[::-1]
    evals = evals[idx]
    evecs_T = evecs_T[:, idx]

    V_k = V[:-1]  # (k, dim)
    eigenvectors_flat = (evecs_T.T @ V_k).reshape(k, dim)
    return evals, eigenvectors_flat


# ---------------------------------------------------------------------------
# Preconditioner in eigenbasis
# ---------------------------------------------------------------------------

def apply_eigen_preconditioner(
    grad_flat: Array,
    eigenvalues: Array,
    eigenvectors: Array,
    damping: float = 1e-4,
    saddle_free_neg: bool = False,
) -> Array:
    """Apply partial Newton-like preconditioner in eigenbasis.

    M ≈ V diag(m_i) V^T + (I - V V^T), where:
      - if saddle_free_neg == False: m_i = 1 / (λ_i + δ)
      - if saddle_free_neg == True:  m_i = 1 / (|λ_i| + δ)

    Additionally uses sqrt scaling (like your original code):
      scale = sqrt(m_i)

    Args:
      grad_flat: (dim,)
      eigenvalues: (k,)
      eigenvectors: (k, dim) rows
    """
    if eigenvalues.size == 0:
        return grad_flat

    V = eigenvectors  # (k, dim)
    lambdas = eigenvalues

    proj = V @ grad_flat            # (k,)
    proj_vec = V.T @ proj           # (dim,)

    if saddle_free_neg:
        lam_eff = jnp.abs(lambdas)
        scale = 1.0 / (lam_eff + damping)
    else:
        scale = 1.0 / (lambdas + damping)

    scale = jnp.sqrt(scale)
    new_subspace = V.T @ (proj * scale)
    g_perp = grad_flat - proj_vec
    return new_subspace + g_perp


def _apply_eigenadam_whole(
    grads: PyTree,
    params: Params,
    eigenvalues: Array,
    eigenvectors: Array,
    precond_damping: float,
    use_saddle_free: bool,
    base_adam: optax.GradientTransformation,
    adam_state: optax.OptState,
) -> Tuple[PyTree, optax.OptState]:
    """Precondition full gradient in eigenbasis, then AdamW."""
    flat_grads, unravel_grads = ravel_pytree(grads)
    precond_flat_grads = apply_eigen_preconditioner(
        grad_flat=flat_grads,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        damping=precond_damping,
        saddle_free_neg=use_saddle_free,
    )
    precond_grads = unravel_grads(precond_flat_grads)
    updates, new_adam_state = base_adam.update(precond_grads, adam_state, params=params)
    return updates, new_adam_state


def _apply_eigenadam_split_spaces(
    grads: PyTree,
    params: Params,
    eigenvalues: Array,
    eigenvectors: Array,
    m_top: Array,
    v_top: Array,
    m_perp: Array,
    v_perp: Array,
    step: Array,
    lr_top: float,
    lr_perp: float,
    beta1: float,
    beta2: float,
    eps: float,
    precond_damping: float,
    use_saddle_free: bool,
    weight_decay: float,
) -> Tuple[PyTree, Array, Array, Array, Array]:
    """Split-space behavior: Newton on top-k, Adam on complement."""
    flat_grads, unravel_grads = ravel_pytree(grads)
    flat_params, _ = ravel_pytree(params)

    V = eigenvectors
    lambdas = eigenvalues

    proj = V @ flat_grads
    g_par = V.T @ proj
    g_perp = flat_grads - g_par

    t = step.astype(jnp.float32) + 1.0
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t

    m_perp = beta1 * m_perp + (1.0 - beta1) * g_perp
    v_perp = beta2 * v_perp + (1.0 - beta2) * (g_perp * g_perp)

    m_perp_hat = m_perp / bc1
    v_perp_hat = v_perp / bc2
    step_perp_flat = -lr_perp * m_perp_hat / (jnp.sqrt(v_perp_hat) + eps)

    if use_saddle_free:
        lam_eff = jnp.abs(lambdas)
    else:
        lam_eff = jnp.maximum(lambdas, 0.0)

    lam_eff = lam_eff + precond_damping
    newton_coeffs = proj / (lam_eff + 1e-12)
    step_top_flat = -lr_top * (V.T @ newton_coeffs)

    step_flat = step_top_flat + step_perp_flat

    if weight_decay != 0.0:
        step_flat = step_flat - lr_perp * weight_decay * flat_params

    updates = unravel_grads(step_flat)
    return updates, m_top, v_top, m_perp, v_perp


# ---------------------------------------------------------------------------
# Optax state
# ---------------------------------------------------------------------------

class PnsEigenAdamState(NamedTuple):
    adam_state: optax.OptState
    step: Array
    eigenvalues: Array
    eigenvectors: Array
    rng_key: Array
    rotation_diff: Array
    m_top: Array
    v_top: Array
    m_perp: Array
    v_perp: Array


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------

def pns_eigenadam_batched(
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    curvature_update_every: int = 100,
    max_eigenvectors: int = 16,
    lanczos_iters: Optional[int] = None,
    ggn_matvec_fn: Optional[GGNMatvecFn] = None,
    precond_damping: float = 1e-4,
    *,
    backend: str = "ggn",
    split_spaces: bool = False,
    lr_top: Optional[float] = None,
    lr_perp: Optional[float] = None,
    # NEW: curvature eigensolver controls
    eigensolver: str = "block_oi",   # "block_oi" or "lanczos"
    block_iters: int = 4,            # iterations of subspace iteration
    independent_rng_per_vec: bool = False,  # for stochastic matvecs
) -> optax.GradientTransformation:
    """PN-S EigenAdam with batched (block) curvature eigensolver.

    Args:
      eigensolver: "block_oi" (batched) or "lanczos" (sequential baseline).
      block_iters: number of block power iterations (2–8 typical).
      independent_rng_per_vec: when using batched matvec, whether to split rng
        per vector (recommended if ggn_matvec_fn uses randomness).
    """
    if lanczos_iters is None:
        lanczos_iters = max_eigenvectors
    k_top = min(max_eigenvectors, lanczos_iters)

    base_adam = optax.adamw(
        learning_rate=learning_rate,
        b1=beta1,
        b2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )

    def init_fn(params: Params) -> PnsEigenAdamState:
        flat_params, _ = ravel_pytree(params)
        dim = flat_params.shape[0]
        dtype = flat_params.dtype

        eigenvalues = jnp.zeros((max_eigenvectors,), dtype=dtype)
        eigenvectors = jnp.zeros((max_eigenvectors, dim), dtype=dtype)

        adam_state = base_adam.init(params)
        rng_key = jax.random.PRNGKey(0)
        step = jnp.array(0, dtype=jnp.int32)
        rotation_diff = jnp.array(0.0, dtype=dtype)

        m_top = jnp.zeros((max_eigenvectors,), dtype=dtype)
        v_top = jnp.zeros((max_eigenvectors,), dtype=dtype)
        m_perp = jnp.zeros((dim,), dtype=dtype)
        v_perp = jnp.zeros((dim,), dtype=dtype)

        return PnsEigenAdamState(
            adam_state=adam_state,
            step=step,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rng_key=rng_key,
            rotation_diff=rotation_diff,
            m_top=m_top,
            v_top=v_top,
            m_perp=m_perp,
            v_perp=v_perp,
        )

    def update_fn(grads: PyTree, state: PnsEigenAdamState, params: Optional[Params] = None):
        assert params is not None, "PN-S EigenAdam requires `params` in update_fn."

        step = state.step + 1
        rng_key = state.rng_key
        eigenvalues = state.eigenvalues
        eigenvectors = state.eigenvectors
        rotation_diff = state.rotation_diff
        m_top = state.m_top
        v_top = state.v_top
        m_perp = state.m_perp
        v_perp = state.v_perp

        lr_top_eff = learning_rate if lr_top is None else lr_top
        lr_perp_eff = learning_rate if lr_perp is None else lr_perp

        use_saddle_free = backend in ("hessian", "fisher")
        sort_by_abs = backend in ("hessian", "fisher")

        # 1) Curvature update
        if ggn_matvec_fn is not None and curvature_update_every > 0:
            should_update = (step % curvature_update_every) == 0

            def do_update(carry):
                (
                    eigenvalues,
                    eigenvectors,
                    rng_key,
                    params,
                    rotation_diff,
                    m_top,
                    v_top,
                ) = carry

                flat_params, unravel_params = ravel_pytree(params)
                dim = flat_params.shape[0]
                dtype = flat_params.dtype

                def matvec_flat(v_flat: Array) -> Array:
                    v_pytree = unravel_params(v_flat)
                    Hv_pytree = ggn_matvec_fn(params, v_pytree, rng_key)
                    Hv_flat, _ = ravel_pytree(Hv_pytree)
                    return Hv_flat

                def matvec_flat_batch(V_flat: Array) -> Array:
                    # V_flat: (b, dim)
                    V_pytree = jax.vmap(unravel_params)(V_flat)

                    if independent_rng_per_vec:
                        keys = jax.random.split(rng_key, V_flat.shape[0])
                        Hv_pytree = jax.vmap(lambda v, k: ggn_matvec_fn(params, v, k))(V_pytree, keys)
                    else:
                        Hv_pytree = jax.vmap(lambda v: ggn_matvec_fn(params, v, rng_key))(V_pytree)

                    return ravel_pytree_batched(Hv_pytree)  # (b, dim)

                rng_key, eig_key = jax.random.split(rng_key)

                if eigensolver == "lanczos":
                    evals, evecs = lanczos(
                        matvec=matvec_flat,
                        dim=dim,
                        num_iter=lanczos_iters,  # static
                        key=eig_key,
                        sort_by_abs=sort_by_abs,
                    )
                else:
                    # batched top-k
                    evals_k, evecs_k = block_orthogonal_iteration(
                        matvec_batch=matvec_flat_batch,
                        dim=dim,
                        k=k_top,
                        num_iter=block_iters,
                        key=eig_key,
                        sort_by_abs=sort_by_abs,
                    )
                    # pad to lanczos_iters-like shape for downstream packing
                    evals = jnp.pad(evals_k, (0, max(0, lanczos_iters - k_top)))
                    evecs = jnp.pad(evecs_k, ((0, max(0, lanczos_iters - k_top)), (0, 0)))

                # For Hessian/Fisher, ensure |λ|-ordering (already handled via sort_by_abs),
                # but keep this extra safety if user swaps solvers.
                if sort_by_abs:
                    order = jnp.argsort(jnp.abs(evals))[::-1]
                    evals = evals[order]
                    evecs = evecs[order, :]

                prev_vecs_k = eigenvectors[:k_top, :]
                new_vecs_k = evecs[:k_top, :]

                diff = new_vecs_k - prev_vecs_k
                frob_num = jnp.linalg.norm(diff)
                frob_den = jnp.linalg.norm(prev_vecs_k)
                frob_den_safe = jnp.where(frob_den > 1e-8, frob_den, 1.0)
                rotation_diff_new = jnp.where(
                    frob_den > 1e-8,
                    frob_num / frob_den_safe,
                    jnp.array(0.0, dtype=dtype),
                )

                eigenvalues_new = jnp.zeros_like(eigenvalues)
                eigenvalues_new = eigenvalues_new.at[:k_top].set(evals[:k_top])

                eigenvectors_new = jnp.zeros_like(eigenvectors)
                eigenvectors_new = eigenvectors_new.at[:k_top, :].set(new_vecs_k)

                if split_spaces and k_top > 0:
                    # Moment transport in top subspace
                    R = new_vecs_k @ prev_vecs_k.T  # (k_top, k_top)
                    m_top_small = m_top[:k_top]
                    v_top_small = v_top[:k_top]
                    m_top_new_small = R @ m_top_small
                    v_top_new_small = R @ v_top_small  # heuristic

                    m_top_new = jnp.zeros_like(m_top)
                    v_top_new = jnp.zeros_like(v_top)
                    m_top_new = m_top_new.at[:k_top].set(m_top_new_small)
                    v_top_new = v_top_new.at[:k_top].set(v_top_new_small)
                else:
                    m_top_new = m_top
                    v_top_new = v_top

                return (
                    eigenvalues_new,
                    eigenvectors_new,
                    rng_key,
                    params,
                    rotation_diff_new,
                    m_top_new,
                    v_top_new,
                )

            def dont_update(carry):
                return carry

            (
                eigenvalues,
                eigenvectors,
                rng_key,
                _,
                rotation_diff,
                m_top,
                v_top,
            ) = jax.lax.cond(
                should_update,
                do_update,
                dont_update,
                operand=(
                    eigenvalues,
                    eigenvectors,
                    rng_key,
                    params,
                    rotation_diff,
                    m_top,
                    v_top,
                ),
            )

        # 2) Apply update
        if not split_spaces:
            updates, new_adam_state = _apply_eigenadam_whole(
                grads=grads,
                params=params,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                precond_damping=precond_damping,
                use_saddle_free=use_saddle_free,
                base_adam=base_adam,
                adam_state=state.adam_state,
            )
            new_m_top = m_top
            new_v_top = v_top
            new_m_perp = m_perp
            new_v_perp = v_perp
        else:
            updates, new_m_top, new_v_top, new_m_perp, new_v_perp = _apply_eigenadam_split_spaces(
                grads=grads,
                params=params,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                m_top=m_top,
                v_top=v_top,
                m_perp=m_perp,
                v_perp=v_perp,
                step=step,
                lr_top=lr_top_eff,
                lr_perp=lr_perp_eff,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                precond_damping=precond_damping,
                use_saddle_free=use_saddle_free,
                weight_decay=weight_decay,
            )
            new_adam_state = state.adam_state

        new_state = PnsEigenAdamState(
            adam_state=new_adam_state,
            step=step,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rng_key=rng_key,
            rotation_diff=rotation_diff,
            m_top=new_m_top,
            v_top=new_v_top,
            m_perp=new_m_perp,
            v_perp=new_v_perp,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
