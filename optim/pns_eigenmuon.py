"""
Matrix-view EigenAdam ("OnsEigenAdam").

This optimizer uses *only* first-order information (per-layer gradient matrices),
no Hessian/GGN/Fisher HVPs.

For each 2D weight matrix W with gradient G:

  1. Define a Gram operator in column space:
         A = G^T G  in R^{n x n}.

  2. Run Lanczos on A via matvecs x -> G^T (G x) to approximate the top-k
     eigenpairs (lambda_i, e_i).

  3. Build a PN-S-style preconditioner in column space:
         M = E diag(1 / (lambda_i + damping)) E^T + (I - E E^T),
     where E = [e_1, ..., e_k] in R^{n x k}.

  4. Precondition the gradient matrix:
         G_pre = G @ M.

Non-2D leaves are left unchanged.

Finally, AdamW is applied to the preconditioned gradients.

This gives a "matrix-orthogonalization-view" EigenAdam that parallels PN-S
EigenAdam conceptually, but works per-layer on gradient matrices and avoids HVPs.
"""

from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax

Array = jax.Array
Params = Any
PyTree = Any


# ---------------------------------------------------------------------------
# Lanczos iterative eigensolver for a symmetric matvec v -> A v
# ---------------------------------------------------------------------------

def lanczos(
    matvec: Callable[[Array], Array],
    dim: int,
    num_iter: int,
    key: Array,
    eps: float = 1e-6,
    sort_by_abs: bool = False,
) -> tuple[Array, Array]:
    """Run Lanczos to approximate top eigenvalues/eigenvectors.

    Args:
      matvec: function v -> A @ v (A is implicit symmetric matrix).
      dim: dimension of the vector space.
      num_iter: number of Lanczos iterations (also size of Krylov subspace).
      key: RNG key for the initial vector.
      eps: small value to guard against breakdown.
      sort_by_abs: if True, sort eigenvalues by |λ| descending; else by λ.

    Returns:
      eigenvalues: (num_iter,) approximated eigenvalues (sorted).
      eigenvectors: (num_iter, dim) corresponding eigenvectors (rows).
    """
    # Random normalized starting vector v0
    v0 = jax.random.normal(key, (dim,))
    v0 = v0 / (jnp.linalg.norm(v0) + eps)

    def body_fun(carry, i):
        V, alphas, betas = carry
        v = V[i]  # (dim,)

        w = matvec(v)              # (dim,)
        alpha = jnp.vdot(v, w)
        w = w - alpha * v

        def ortho_against_prev(_w, _i):
            prev_v = V[_i]
            proj = jnp.vdot(prev_v, _w)
            return _w - proj * prev_v

        # Re-orthogonalize w against previous basis vectors
        w = jax.lax.fori_loop(0, i, lambda j, ww: ortho_against_prev(ww, j), w)

        beta = jnp.linalg.norm(w)
        beta = jnp.where(beta < eps, 0.0, beta)
        next_v = jnp.where(beta > 0, w / (beta + eps), jnp.zeros_like(w))

        V = V.at[i + 1].set(next_v)
        alphas = alphas.at[i].set(alpha)
        betas = betas.at[i].set(beta)
        return (V, alphas, betas), None

    V = jnp.zeros((num_iter + 1, dim), dtype=v0.dtype)
    V = V.at[0].set(v0)
    alphas = jnp.zeros((num_iter,), dtype=v0.dtype)
    betas = jnp.zeros((num_iter,), dtype=v0.dtype)

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
# Per-matrix Gram-based PN-S preconditioning
# ---------------------------------------------------------------------------

def _precondition_matrix_grad(
    grad_mat: Array,
    *,
    max_eigenvectors: int,
    lanczos_iters: int,
    damping: float,
    key: Array,
    eps: float = 1e-6,
    sqrt_scaling: bool = False,
) -> tuple[Array, Array | None]:
    """
    Precondition a single 2D gradient matrix G using a PN-S-style operator
    built from the Gram matrix.

    Let G ∈ R^{m×n}. We choose the smaller side and define:

      - If n <= m (column Gram):
            A = G^T G ∈ R^{n×n}
        with eigenpairs A e_i = λ_i e_i, stack E = [e_1,...,e_k] ∈ R^{n×k}.

        Project & split:
            g_t^∥ = E E^T g_t,   g_t^⊥ = (I - E E^T) g_t

        Scaling coefficients:
            s_i = 1 / sqrt(λ_i + δ)   if sqrt_scaling
                = 1 / (λ_i + δ)       otherwise

        S = diag(s_i). New gradient:
            g̃_t = E S E^T g_t + (I - E E^T) g_t

        In matrix form for the layer:
            M_col = E S E^T + (I - E E^T) ∈ R^{n×n}
            G_pre = G @ M_col.

      - If m < n (row Gram):
            A = G G^T ∈ R^{m×m}
        with eigenpairs A f_i = λ_i f_i, stack F = [f_1,...,f_k] ∈ R^{m×k}.

        Analogously:
            M_row = F S F^T + (I - F F^T) ∈ R^{m×m}
            G_pre = M_row @ G.

    Args:
      grad_mat: (m, n) gradient matrix G.
      max_eigenvectors: maximum k (top modes) to track.
      lanczos_iters: number of Lanczos iterations.
      damping: δ in 1 / (λ_i + δ).
      key: PRNG key for Lanczos initial vector.
      eps: numerical guard.
      sqrt_scaling: if True, use 1 / sqrt(λ + δ); if False, 1 / (λ + δ).

    Returns:
      precond_grad_mat: (m, n) preconditioned gradient matrix.
      eigenvalues: (k,) eigenvalues used for the preconditioner, or None.
    """
    if grad_mat.ndim != 2:
        # Only operate on true matrices.
        return grad_mat, None

    m, n = grad_mat.shape
    if m == 0 or n == 0:
        return grad_mat, jnp.zeros((max_eigenvectors,), dtype=grad_mat.dtype)

    # Work in the smaller dimension.
    d = min(m, n)
    k = int(min(max_eigenvectors, lanczos_iters, d))
    if k <= 0:
        return grad_mat, jnp.zeros((max_eigenvectors,), dtype=grad_mat.dtype)

    # -------------------------------
    # Column Gram: A = G^T G (n <= m)
    # -------------------------------
    if n <= m:
        def gram_matvec_col(x: Array) -> Array:
            # x: (n,)
            y = grad_mat @ x        # (m,)
            return grad_mat.T @ y   # (n,)

        evals, evecs = lanczos(
            matvec=gram_matvec_col,
            dim=n,
            num_iter=k,
            key=key,
            eps=eps,
            sort_by_abs=False,   # PSD; largest λ first
        )
        lambdas = evals                     # (k,)
        V = evecs                           # (k, n)
        E = V.T                             # (n, k)

        # g_t^∥ = E E^T g_t  →  G_parallel = G E E^T
        G_top = grad_mat @ E               # (m, k)
        G_parallel = G_top @ E.T           # (m, n)
        G_perp = grad_mat - G_parallel     # (m, n)

        # s_i from eigenvalues
        if sqrt_scaling:
            scale = 1.0 / jnp.sqrt(lambdas + damping + 1e-12)
        else:
            scale = 1.0 / (lambdas + damping + 1e-12)

        G_top_scaled = G_top * scale[None, :]   # (m, k)
        G_top_pre = G_top_scaled @ E.T          # (m, n)

        precond_grad = G_top_pre + G_perp       # (m, n)

    # ---------------------------
    # Row Gram: A = G G^T (m < n)
    # ---------------------------
    else:
        def gram_matvec_row(x: Array) -> Array:
            # x: (m,)
            y = grad_mat.T @ x      # (n,)
            return grad_mat @ y     # (m,)

        evals, evecs = lanczos(
            matvec=gram_matvec_row,
            dim=m,
            num_iter=k,
            key=key,
            eps=eps,
            sort_by_abs=False,
        )
        lambdas = evals                     # (k,)
        V = evecs                           # (k, m)
        F = V.T                             # (m, k)

        # g_t^∥ = F F^T g_t  →  G_parallel = F F^T G
        G_top_row = F.T @ grad_mat          # (k, n)
        G_parallel = F @ G_top_row          # (m, n)
        G_perp = grad_mat - G_parallel      # (m, n)

        if sqrt_scaling:
            scale = 1.0 / jnp.sqrt(lambdas + damping + 1e-12)
        else:
            scale = 1.0 / (lambdas + damping + 1e-12)

        G_top_row_scaled = scale[:, None] * G_top_row  # (k, n)
        G_top_pre = F @ G_top_row_scaled               # (m, n)

        precond_grad = G_top_pre + G_perp              # (m, n)

    if k < max_eigenvectors:
        eigvals = jnp.zeros((max_eigenvectors,), dtype=lambdas.dtype)
        eigvals = eigvals.at[:k].set(lambdas)
    else:
        eigvals = lambdas
    return precond_grad, eigvals


# ---------------------------------------------------------------------------
# Optimizer state and wrapper
# ---------------------------------------------------------------------------

class OnsEigenAdamState(NamedTuple):
    """State for matrix-view EigenAdam (OnsEigenAdam)."""
    adam_state: optax.OptState
    step: Array
    eigenvalues: PyTree


def pns_eigenmuon(
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    max_eigenvectors: int = 8,
    lanczos_iters: Optional[int] = None,
    precond_damping: float = 1e-4,
    sqrt_scaling: bool = False,
) -> optax.GradientTransformation:
    """
    Matrix-view EigenAdam ("OnsEigenAdam").

    This optimizer:

      - Walks the gradient PyTree.
      - For every 2D leaf (gradient matrix G ∈ R^{m×n}):
          * builds the Gram operator A = G^T G in column space implicitly,
          * runs k-step Lanczos on A with matvec x -> G^T (G x),
          * applies a PN-S-style preconditioner in the top-k eigenbasis:
                M = E diag(s_i) E^T + (I - E E^T),
                s_i = 1 / (λ_i + δ) or 1 / sqrt(λ_i + δ),
                G_pre = G @ M,
          * returns the preconditioned gradient matrix G_pre.
      - Leaves non-2D leaves untouched.
      - Feeds the resulting preconditioned gradients into AdamW.

    Args:
      learning_rate: base learning rate (used by AdamW).
      beta1: Adam β1.
      beta2: Adam β2.
      eps: Adam epsilon.
      weight_decay: AdamW decoupled weight decay.
      max_eigenvectors: maximum number of top modes (k) per matrix.
      lanczos_iters: number of Lanczos iterations; if None, defaults to max_eigenvectors.
      precond_damping: δ in 1 / (λ + δ).
      sqrt_scaling: if True, use 1 / sqrt(λ + δ) instead of 1 / (λ + δ).

    Returns:
      An optax.GradientTransformation (init_fn, update_fn).
    """
    if lanczos_iters is None:
        lanczos_iters = max_eigenvectors

    base_adam = optax.adamw(
        learning_rate=learning_rate,
        b1=beta1,
        b2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )

    def init_fn(params: Params) -> OnsEigenAdamState:
        adam_state = base_adam.init(params)
        step = jnp.array(0, dtype=jnp.int32)

        def init_leaf(p: Array) -> Array | None:
            if isinstance(p, jax.Array) and p.ndim == 2:
                return jnp.zeros((max_eigenvectors,), dtype=p.dtype)
            return None

        eigenvalues = jtu.tree_map(init_leaf, params)
        return OnsEigenAdamState(
            adam_state=adam_state,
            step=step,
            eigenvalues=eigenvalues,
        )

    def update_fn(
        grads: PyTree,
        state: OnsEigenAdamState,
        params: Optional[Params] = None,
    ):
        if params is None:
            raise ValueError("ons_eigenadam requires `params` in update_fn.")

        step = state.step + 1

        # PRNG key based on step; reused for all leaves for simplicity.
        # This is fine: each matrix has different G, so Lanczos outputs differ.
        key = jax.random.PRNGKey(step)

        def map_leaf(g: Array) -> tuple[Array, Array | None]:
            # Only touch true matrices; everything else is passed through.
            if not isinstance(g, jax.Array) or g.ndim != 2:
                return g, None

            return _precondition_matrix_grad(
                grad_mat=g,
                max_eigenvectors=max_eigenvectors,
                lanczos_iters=lanczos_iters,
                damping=precond_damping,
                key=key,
                sqrt_scaling=sqrt_scaling,
            )

        precond_with_eigs = jtu.tree_map(map_leaf, grads)
        is_pair = lambda x: isinstance(x, tuple)
        precond_grads = jtu.tree_map(
            lambda pair: pair[0], precond_with_eigs, is_leaf=is_pair
        )
        eigenvalues = jtu.tree_map(
            lambda pair: pair[1], precond_with_eigs, is_leaf=is_pair
        )

        updates, new_adam_state = base_adam.update(
            precond_grads,
            state.adam_state,
            params=params,
        )

        new_state = OnsEigenAdamState(
            adam_state=new_adam_state,
            step=step,
            eigenvalues=eigenvalues,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
