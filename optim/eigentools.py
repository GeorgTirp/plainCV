from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax

Array = jax.Array
Params = Any
PyTree = Any

# Type: given params, direction (same PyTree structure as params), rng -> matvec(direction)
GGNMatvecFn = Callable[[Params, PyTree, Array], PyTree]


class PnsEigenAdamState(NamedTuple):
    opt_state: Optional[optax.OptState] = None
    adam_state: Optional[optax.OptState] = None
    step: Optional[Array] = None
    eigenvalues: Optional[Array] = None
    eigenvectors: Optional[Array] = None
    rng_key: Optional[Array] = None
    rotation_diff: Optional[Array] = None
    m_top: Optional[Array] = None
    v_top: Optional[Array] = None
    m_perp: Optional[Array] = None
    v_perp: Optional[Array] = None
    lr_perp_eff: Optional[Array] = None
    last_refresh_step: Optional[Array] = None
    active_k: Optional[Array] = None
    innovation_residual: Optional[Array] = None


class EigenTrackingState(NamedTuple):
    step: Array
    eigenvalues: Array
    eigenvectors: Array
    extra_eigenvalues: Array
    extra_eigenvectors: Array
    alpha: Array
    alpha_lambda: Array
    alpha_valid: Array
    eff_cond: Array
    rng_key: Array
    rotation_diff: Array


def _project_rows(matrix: Array, vector: Array) -> Array:
    """Project each row of a matrix onto a vector without a large GEMM."""
    return jax.lax.map(lambda row: jnp.vdot(row, vector), matrix)


def _expand_from_basis(coeffs_matrix: Array, basis_rows: Array) -> Array:
    """Form row-wise linear combinations of a basis without a fused k x dim GEMM."""
    return jax.lax.map(
        lambda coeffs: jnp.tensordot(coeffs, basis_rows, axes=1),
        coeffs_matrix,
    )


def init_eigentracking(
    params: Params,
    k: int,
    *,
    extra_modes: int = 0,
    seed: int = 0,
) -> EigenTrackingState:
    flat_params, _ = ravel_pytree(params)
    dim = flat_params.shape[0]
    dtype = flat_params.dtype
    return EigenTrackingState(
        step=jnp.array(0, dtype=jnp.int32),
        eigenvalues=jnp.zeros((k,), dtype=dtype),
        eigenvectors=jnp.zeros((k, dim), dtype=dtype),
        extra_eigenvalues=jnp.zeros((extra_modes,), dtype=dtype),
        extra_eigenvectors=jnp.zeros((extra_modes, dim), dtype=dtype),
        alpha=jnp.full((k,), jnp.nan, dtype=dtype),
        alpha_lambda=jnp.full((k,), jnp.nan, dtype=dtype),
        alpha_valid=jnp.zeros((k,), dtype=bool),
        eff_cond=jnp.array(0.0, dtype=dtype),
        rng_key=jax.random.PRNGKey(seed),
        rotation_diff=jnp.array(0.0, dtype=dtype),
    )


def _subspace_rotation_diff(
    prev_vecs: Array,
    new_vecs: Array,
    eps: float,
) -> Array:
    prev_norm = jnp.linalg.norm(prev_vecs)

    def compute_diff(_: None) -> Array:
        overlap = prev_vecs @ new_vecs.T
        overlap_sq = jnp.sum(overlap * overlap)
        k = jnp.asarray(prev_vecs.shape[0], dtype=new_vecs.dtype)
        diff_sq = jnp.maximum(0.0, 2.0 * k - 2.0 * overlap_sq)
        return jnp.sqrt(diff_sq)

    return jax.lax.cond(
        prev_norm > eps,
        compute_diff,
        lambda _: jnp.array(0.0, dtype=new_vecs.dtype),
        operand=None,
    )


def _make_lanczos_warm_start(
    prev_eigenvectors: Array,
    prev_eigenvalues: Array,
    eps: float,
) -> Array:
    """
    Build a single warm-start vector from the previously tracked eigenspace.

    We use an abs-eigenvalue-weighted combination of the previous basis rows.
    If the previous basis is all zeros, lanczos(...) will automatically fall
    back to its random initialization.
    """
    weights = jnp.abs(prev_eigenvalues)
    weights = weights / (jnp.sum(weights) + eps)
    warm_start = jnp.tensordot(weights, prev_eigenvectors, axes=1)
    return warm_start


def _align_eigenvector_rows(
    prev_vecs: Array,
    new_vecs: Array,
) -> Array:
    """Align row signs to the previous iterate for smoother tracking."""
    dot = jnp.sum(prev_vecs * new_vecs, axis=1, keepdims=True)
    sign = jnp.sign(dot)
    sign = jnp.where(sign == 0.0, 1.0, sign)
    return new_vecs * sign


def track_eigenstate(
    params: Params,
    grads: PyTree,
    updates: PyTree,
    step: Array,
    eigen_state: EigenTrackingState,
    *,
    matvec_fn: GGNMatvecFn,
    num_iter: Optional[int] = None,
    sort_by_abs: bool = False,
    use_light_ortho: bool = False,
    light_ortho_every: int = 4,
    eps: float = 1e-12,
    alpha_grad_tol_abs: float = 1e-10,
    alpha_grad_tol_rel: float = 1e-3,
) -> EigenTrackingState:
    flat_params, unravel_params = ravel_pytree(params)
    dim = flat_params.shape[0]
    grad_flat, _ = ravel_pytree(grads)
    upd_flat, _ = ravel_pytree(updates)

    rng_key, lanczos_key = jax.random.split(eigen_state.rng_key)
    k = eigen_state.eigenvalues.shape[0]
    extra_k = eigen_state.extra_eigenvalues.shape[0]
    total_keep = k + extra_k
    if total_keep == 0:
        return eigen_state._replace(step=step, rng_key=rng_key)

    lanczos_steps = max(total_keep, total_keep if num_iter is None else int(num_iter))

    def matvec_flat(v_flat: Array) -> Array:
        v_pytree = unravel_params(v_flat)
        hv_pytree = matvec_fn(params, v_pytree, rng_key)
        hv_flat, _ = ravel_pytree(hv_pytree)
        return hv_flat

    prev_all_eigenvectors = jnp.concatenate(
        [eigen_state.eigenvectors, eigen_state.extra_eigenvectors],
        axis=0,
    )
    prev_all_eigenvalues = jnp.concatenate(
        [eigen_state.eigenvalues, eigen_state.extra_eigenvalues],
        axis=0,
    )

    # ---- Point 4 fix: warm-start Lanczos from previous eigenspace ----
    warm_start_v = _make_lanczos_warm_start(
        prev_all_eigenvectors,
        prev_all_eigenvalues,
        eps,
    )

    evals, evecs = lanczos(
        matvec=matvec_flat,
        dim=dim,
        num_iter=lanczos_steps,
        key=lanczos_key,
        eps=eps,
        sort_by_abs=sort_by_abs,
        init_v=warm_start_v,
        use_light_ortho=use_light_ortho,
        light_ortho_every=light_ortho_every,
    )

    eigenvalues = evals[:k]
    eigenvectors = evecs[:k, :]
    extra_eigenvalues = evals[k : k + extra_k]
    extra_eigenvectors = evecs[k : k + extra_k, :]

    prev_vecs = eigen_state.eigenvectors
    eigenvectors = _align_eigenvector_rows(prev_vecs, eigenvectors)

    prev_extra_vecs = eigen_state.extra_eigenvectors
    extra_eigenvectors = _align_eigenvector_rows(prev_extra_vecs, extra_eigenvectors)

    if k > 0:
        rotation_diff = _subspace_rotation_diff(prev_vecs, eigenvectors, eps)

        g_proj = _project_rows(eigenvectors, grad_flat)
        d_proj = _project_rows(eigenvectors, upd_flat)

        # Relative threshold is taken against the largest projected gradient
        # magnitude in the tracked subspace, with an absolute floor.
        g_ref = jnp.maximum(jnp.max(jnp.abs(g_proj)), eps)
        g_tol = jnp.maximum(
            jnp.asarray(alpha_grad_tol_abs, dtype=g_proj.dtype),
            jnp.asarray(alpha_grad_tol_rel, dtype=g_proj.dtype) * g_ref,
        )
        alpha_valid = jnp.abs(g_proj) > g_tol

        safe_g_proj = jnp.where(alpha_valid, g_proj, 1.0)
        alpha_raw = -d_proj / safe_g_proj
        alpha = jnp.where(alpha_valid, alpha_raw, jnp.nan)

        alpha_lambda_raw = alpha_raw * eigenvalues
        alpha_lambda = jnp.where(alpha_valid, alpha_lambda_raw, jnp.nan)

        alpha_lambda_abs = jnp.abs(jnp.where(alpha_valid, alpha_lambda_raw, 0.0))
        valid_for_cond = jnp.logical_and(alpha_valid, alpha_lambda_abs > eps)

        max_abs = jnp.max(jnp.where(valid_for_cond, alpha_lambda_abs, 0.0))
        min_abs = jnp.min(
            jnp.where(
                valid_for_cond,
                alpha_lambda_abs,
                jnp.full_like(alpha_lambda_abs, jnp.inf),
            )
        )
        eff_cond = jnp.where(
            jnp.any(valid_for_cond),
            max_abs / jnp.maximum(min_abs, eps),
            jnp.array(0.0, dtype=alpha_lambda.dtype),
        )
    else:
        rotation_diff = jnp.array(0.0, dtype=eigenvalues.dtype)
        alpha = eigen_state.alpha
        alpha_lambda = eigen_state.alpha_lambda
        alpha_valid = eigen_state.alpha_valid
        eff_cond = jnp.array(0.0, dtype=eigenvalues.dtype)

    return eigen_state._replace(
        step=step,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        extra_eigenvalues=extra_eigenvalues,
        extra_eigenvectors=extra_eigenvectors,
        alpha=alpha,
        alpha_lambda=alpha_lambda,
        alpha_valid=alpha_valid,
        eff_cond=eff_cond,
        rng_key=rng_key,
        rotation_diff=rotation_diff,
    )


def lanczos(
    matvec: Callable[[Array], Array],
    dim: int,
    num_iter: int,
    key: Array,
    eps: float = 1e-6,
    sort_by_abs: bool = False,
    init_v: Optional[Array] = None,
    use_light_ortho: bool = False,
    light_ortho_every: int = 4,
) -> Tuple[Array, Array]:
    v0_rand = jax.random.normal(key, (dim,))
    v0_rand = v0_rand / (jnp.linalg.norm(v0_rand) + eps)

    if init_v is None:
        v0 = v0_rand
    else:
        init_norm = jnp.linalg.norm(init_v)
        init_is_valid = jnp.logical_and(jnp.isfinite(init_norm), init_norm > eps)
        init_dir = init_v / (init_norm + eps)
        v0 = jnp.where(init_is_valid, init_dir, v0_rand)

    def body_fun(carry, i):
        v_basis, alphas, betas = carry
        v = v_basis[i]

        w = matvec(v)
        alpha = jnp.vdot(v, w)
        w = w - alpha * v

        def ortho_against_prev(current_w, basis_idx):
            prev_v = v_basis[basis_idx]
            proj = jnp.vdot(prev_v, current_w)
            return current_w - proj * prev_v

        full_reorth = lambda ww: jax.lax.fori_loop(
            0,
            i,
            lambda basis_idx, current_w: ortho_against_prev(current_w, basis_idx),
            ww,
        )

        def prev_only_reorth(ww):
            return jax.lax.cond(
                i > 0,
                lambda x: ortho_against_prev(x, i - 1),
                lambda x: x,
                ww,
            )

        if use_light_ortho:
            do_full = (i % light_ortho_every) == 0
            w = jax.lax.cond(do_full, full_reorth, prev_only_reorth, w)
        else:
            w = full_reorth(w)

        beta = jnp.linalg.norm(w)
        beta = jnp.where(beta < eps, 0.0, beta)
        next_v = jnp.where(beta > 0, w / (beta + eps), jnp.zeros_like(w))

        v_basis = v_basis.at[i + 1].set(next_v)
        alphas = alphas.at[i].set(alpha)
        betas = betas.at[i].set(beta)
        return (v_basis, alphas, betas), None

    v_basis = jnp.zeros((num_iter + 1, dim))
    v_basis = v_basis.at[0].set(v0)
    alphas = jnp.zeros((num_iter,))
    betas = jnp.zeros((num_iter,))

    (v_basis, alphas, betas), _ = jax.lax.scan(
        body_fun,
        (v_basis, alphas, betas),
        jnp.arange(num_iter),
    )

    tridiag = jnp.diag(alphas)
    if num_iter > 1:
        tridiag = tridiag.at[jnp.arange(num_iter - 1), jnp.arange(1, num_iter)].set(
            betas[: num_iter - 1]
        )
        tridiag = tridiag.at[jnp.arange(1, num_iter), jnp.arange(num_iter - 1)].set(
            betas[: num_iter - 1]
        )

    evals, evecs_t = jnp.linalg.eigh(tridiag)
    if sort_by_abs:
        idx = jnp.argsort(jnp.abs(evals))[::-1]
    else:
        idx = jnp.argsort(evals)[::-1]

    evals = evals[idx]
    evecs_t = evecs_t[:, idx]

    v_k = v_basis[:-1]
    eigenvectors = _expand_from_basis(evecs_t.T, v_k).reshape(num_iter, dim)
    return evals, eigenvectors
