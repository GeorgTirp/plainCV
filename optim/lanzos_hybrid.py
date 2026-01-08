from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax.flatten_util import ravel_pytree
import optax

Array = jax.Array
Params = Any
PyTree = Any

# Given params, direction (same PyTree structure as params), rng -> matvec(direction)
GGNMatvecFn = Callable[[Params, PyTree, Array], PyTree]


# ---------------------------------------------------------------------------
# Shared Lanczos iterative eigensolver (for both Gram + global curvature)
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
# Per-matrix (muon) Gram-based PN-S preconditioning
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
) -> Array:
    """
    Muon-style preconditioner: use Gram matrix of a single 2D gradient G.

    Let G ∈ R^{m×n}. We choose the smaller side and define:

      - If n <= m (column Gram):
            A = G^T G ∈ R^{n×n}
        with eigenpairs A e_i = λ_i e_i, stack E = [e_1,...,e_k] ∈ R^{n×k}.

        M_col = E diag(s_i) E^T + (I - E E^T) ∈ R^{n×n}
        G_pre = G @ M_col.

      - If m < n (row Gram):
            A = G G^T ∈ R^{m×m}
        with eigenpairs A f_i = λ_i f_i, stack F = [f_1,...,f_k] ∈ R^{m×k}.

        M_row = F diag(s_i) F^T + (I - F F^T) ∈ R^{m×m}
        G_pre = M_row @ G.

      s_i = 1/(λ_i+δ) or 1/sqrt(λ_i+δ).

    Non-2D leaves are left unchanged.
    """
    if grad_mat.ndim != 2:
        return grad_mat

    m, n = grad_mat.shape
    if m == 0 or n == 0:
        return grad_mat

    # Work in the smaller dimension (static, since weight shapes are static).
    d = min(m, n)
    k = int(min(max_eigenvectors, lanczos_iters, d))
    if k <= 0:
        return grad_mat

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

        G_top = grad_mat @ E               # (m, k)
        G_parallel = G_top @ E.T           # (m, n)
        G_perp = grad_mat - G_parallel     # (m, n)

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

    return precond_grad


# ---------------------------------------------------------------------------
# Global PN-S eigenbasis preconditioner (Hessian/GGN/Fisher)
# ---------------------------------------------------------------------------

def apply_eigen_preconditioner(
    grad_flat: Array,
    eigenvalues: Array,
    eigenvectors: Array,
    damping: float = 1e-4,
    saddle_free_neg: bool = False,
) -> Array:
    """Apply PN-S preconditioner in a global eigenbasis.

    M ≈ V diag(m_i) V^T + (I - V V^T), where:

      - If saddle_free_neg == False:
           m_i = 1 / (λ_i + δ)
      - If saddle_free_neg == True:
           m_i = 1 / (|λ_i| + δ)

    We actually use a sqrt scaling (EigenAdam style) by default.
    """
    if eigenvalues.size == 0:
        return grad_flat

    V = eigenvectors  # (k, dim)
    lambdas = eigenvalues  # (k,)

    # Project gradient into eigenbasis: g_i = v_i^T g
    proj = V @ grad_flat  # (k,)

    # Component of g in span(V)
    proj_vec = V.T @ proj  # (dim,)

    if saddle_free_neg:
        lam_abs = jnp.abs(lambdas)
        scale = 1.0 / (lam_abs + damping)
    else:
        scale = 1.0 / (lambdas + damping)

    scale = jnp.sqrt(scale)        # EigenAdam-style
    scaled = proj * scale          # (k,)
    new_subspace = V.T @ scaled    # (dim,)

    # Orthogonal component left untouched
    g_perp = grad_flat - proj_vec

    return new_subspace + g_perp


# ---------------------------------------------------------------------------
# Hybrid state: global curvature + shared AdamW
# ---------------------------------------------------------------------------

class HybridEigenState(NamedTuple):
    adam_state: optax.OptState
    step: Array
    rng_key: Array

    # Global curvature basis (for Hessian/GGN/Fisher)
    eigenvalues: Array      # (global_max_eigenvectors,)
    eigenvectors: Array     # (global_max_eigenvectors, dim)
    rotation_diff: Array    # scalar, Frobenius distance between old/new bases


# ---------------------------------------------------------------------------
# Hybrid optimizer: muon (2D) + global PN-S, single AdamW call
# ---------------------------------------------------------------------------

def pns_eigen_hybrid(
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    # ---- global PN-S (Hessian/GGN/Fisher) part ----
    ggn_matvec_fn: Optional[GGNMatvecFn] = None,
    global_max_eigenvectors: int = 16,
    global_lanczos_iters: Optional[int] = None,
    global_precond_damping: float = 1e-4,
    curvature_update_every: int = 100,
    backend: str = "ggn",   # "ggn", "hessian", or "fisher"
    # ---- per-matrix (muon) part ----
    muon_max_eigenvectors: int = 8,
    muon_lanczos_iters: Optional[int] = None,
    muon_precond_damping: float = 1e-4,
    muon_sqrt_scaling: bool = False,
) -> optax.GradientTransformation:
    """
    Hybrid PN-S optimizer:

      - (Optional) Muon-style per-matrix Gram preconditioning on all 2D leaves.
      - (Optional) Global PN-S preconditioning via Hessian/GGN/Fisher HVPs.
      - Single AdamW call at the end.

    Skipping logic:
      - If muon_lanczos_iters <= 0 or muon_max_eigenvectors <= 0:
            → muon preconditioning is skipped.
      - If global_lanczos_iters <= 0, or curvature_update_every <= 0,
        or ggn_matvec_fn is None:
            → global PN-S is skipped.

    You get:
      g_raw  --(muon, if enabled)--> g_muon
             --(global PN-S, if enabled)--> g_pre
             --(AdamW)--> parameter update.
    """
    # ----- set defaults / static flags -----
    if muon_lanczos_iters is None:
        muon_lanczos_iters = muon_max_eigenvectors

    if global_lanczos_iters is None:
        global_lanczos_iters = global_max_eigenvectors

    enable_muon = (muon_lanczos_iters is not None and
                   muon_lanczos_iters > 0 and
                   muon_max_eigenvectors > 0)

    enable_global = (ggn_matvec_fn is not None and
                     global_lanczos_iters is not None and
                     global_lanczos_iters > 0 and
                     curvature_update_every > 0 and
                     global_max_eigenvectors > 0)

    # Static number of tracked modes for global PN-S
    k_top = int(min(global_max_eigenvectors, global_lanczos_iters))

    base_adam = optax.adamw(
        learning_rate=learning_rate,
        b1=beta1,
        b2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )

    def init_fn(params: Params) -> HybridEigenState:
        flat_params, _ = ravel_pytree(params)
        dim = flat_params.shape[0]
        dtype = flat_params.dtype

        if enable_global:
            eigenvalues = jnp.zeros((global_max_eigenvectors,), dtype=dtype)
            eigenvectors = jnp.zeros((global_max_eigenvectors, dim), dtype=dtype)
        else:
            # We still create zero-sized arrays for simplicity
            eigenvalues = jnp.zeros((0,), dtype=dtype)
            eigenvectors = jnp.zeros((0, flat_params.shape[0]), dtype=dtype)

        adam_state = base_adam.init(params)
        rng_key = jax.random.PRNGKey(0)
        step = jnp.array(0, dtype=jnp.int32)
        rotation_diff = jnp.array(0.0, dtype=dtype)

        return HybridEigenState(
            adam_state=adam_state,
            step=step,
            rng_key=rng_key,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rotation_diff=rotation_diff,
        )

    def update_fn(
        grads: PyTree,
        state: HybridEigenState,
        params: Optional[Params] = None,
    ):
        if params is None:
            raise ValueError("pns_eigen_hybrid requires `params` in update_fn.")

        step = state.step + 1
        rng_key = state.rng_key
        eigenvalues = state.eigenvalues
        eigenvectors = state.eigenvectors
        rotation_diff = state.rotation_diff

        # ---------------------------------------------------------------
        # 1. Muon-style per-matrix PN-S (pure preconditioner on 2D leaves)
        # ---------------------------------------------------------------
        if enable_muon:
            rng_key, muon_key = jax.random.split(rng_key)

            def map_leaf(g: Array) -> Array:
                if not isinstance(g, jax.Array) or g.ndim != 2:
                    return g
                return _precondition_matrix_grad(
                    grad_mat=g,
                    max_eigenvectors=muon_max_eigenvectors,
                    lanczos_iters=muon_lanczos_iters,
                    damping=muon_precond_damping,
                    key=muon_key,
                    sqrt_scaling=muon_sqrt_scaling,
                )

            grads_muon = jtu.tree_map(map_leaf, grads)
        else:
            grads_muon = grads

        # ---------------------------------------------------------------
        # 2. Global PN-S curvature update (if enabled) every N steps
        # ---------------------------------------------------------------
        use_saddle_free = (backend in ("hessian", "fisher"))
        sort_by_abs = (backend in ("hessian", "fisher"))

        if enable_global:
            should_update = (step % curvature_update_every) == 0

            def do_update(carry):
                eigenvalues, eigenvectors, rng_key, params, rotation_diff = carry

                flat_params, unravel_params = ravel_pytree(params)
                dim = flat_params.shape[0]
                dtype = flat_params.dtype

                def matvec_flat(v_flat: Array) -> Array:
                    v_pytree = unravel_params(v_flat)
                    Hv_pytree = ggn_matvec_fn(params, v_pytree, rng_key)
                    Hv_flat, _ = ravel_pytree(Hv_pytree)
                    return Hv_flat

                rng_key, lanczos_key = jax.random.split(rng_key)
                evals, evecs = lanczos(
                    matvec=matvec_flat,
                    dim=dim,
                    num_iter=global_lanczos_iters,  # STATIC int
                    key=lanczos_key,
                    sort_by_abs=sort_by_abs,
                )  # evecs: (global_lanczos_iters, dim)

                if backend in ("hessian", "fisher"):
                    order = jnp.argsort(jnp.abs(evals))[::-1]
                    evals = evals[order]
                    evecs = evecs[order, :]

                # Use a STATIC number of modes: k_top.
                prev_vecs_k = eigenvectors[:k_top, :]  # (k_top, dim)
                new_vecs_k = evecs[:k_top, :]          # (k_top, dim)

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

                return (
                    eigenvalues_new,
                    eigenvectors_new,
                    rng_key,
                    params,
                    rotation_diff_new,
                )

            def dont_update(carry):
                return carry

            (
                eigenvalues,
                eigenvectors,
                rng_key,
                _,
                rotation_diff,
            ) = jax.lax.cond(
                should_update,
                do_update,
                dont_update,
                operand=(eigenvalues, eigenvectors, rng_key, params, rotation_diff),
            )
        # else: global PN-S disabled → eigenvalues/eigenvectors unchanged / unused

        # ---------------------------------------------------------------
        # 3. Global PN-S preconditioning on (possibly muon-preconditioned) grads
        # ---------------------------------------------------------------
        if enable_global:
            flat_grads_muon, unravel_grads = ravel_pytree(grads_muon)
            precond_flat = apply_eigen_preconditioner(
                grad_flat=flat_grads_muon,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                damping=global_precond_damping,
                saddle_free_neg=use_saddle_free,
            )
            precond_grads = unravel_grads(precond_flat)
        else:
            precond_grads = grads_muon

        # ---------------------------------------------------------------
        # 4. Single AdamW step on the fully preconditioned gradients
        # ---------------------------------------------------------------
        updates, new_adam_state = base_adam.update(
            precond_grads,
            state.adam_state,
            params=params,
        )

        new_state = HybridEigenState(
            adam_state=new_adam_state,
            step=step,
            rng_key=rng_key,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rotation_diff=rotation_diff,
        )

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
