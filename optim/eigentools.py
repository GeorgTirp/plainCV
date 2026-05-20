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
    ritz_residual: Array
    extra_ritz_residual: Array
    g_proj: Array
    extra_g_proj: Array
    d_proj: Array
    extra_d_proj: Array
    update_energy_frac: Array
    extra_update_energy_frac: Array
    alpha: Array
    extra_alpha: Array
    phi: Array
    extra_phi: Array
    eos_rho: Array
    extra_eos_rho: Array
    alpha_valid: Array
    extra_alpha_valid: Array
    topk_eos_rho_max: Array
    topk_eos_rho_mean: Array
    topk_eos_rho_update_weighted: Array
    topk_eos_rho_over_2_max: Array
    topk_eos_rho_over_2_update_weighted: Array
    eff_cond: Array
    rng_key: Array
    rotation_diff: Array
    grad_norm: Array
    update_norm: Array
    tracked_update_energy_frac: Array
    tracked_grad_energy_frac: Array
    tracked_update_grad_cosine: Array
    pos_update_energy_frac: Array
    neg_update_energy_frac: Array
    pos_grad_energy_frac: Array
    neg_grad_energy_frac: Array
    pos_update_grad_cosine: Array
    neg_update_grad_cosine: Array
    effective_curvature_cond: Array
    actual_update_rayleigh: Array
    effective_curvature_eigenvalues: Array
    extra_effective_curvature_eigenvalues: Array
    damped_effective_curvature_eigenvalues: Array
    extra_damped_effective_curvature_eigenvalues: Array
    preconditioned_dir_gain: Array
    extra_preconditioned_dir_gain: Array


def _project_rows(matrix: Array, vector: Array) -> Array:
    """Project each row of a matrix onto a vector without a large GEMM."""
    return jax.lax.map(lambda row: jnp.vdot(row, vector), matrix)


def _expand_from_basis(coeffs_matrix: Array, basis_rows: Array) -> Array:
    """Form row-wise linear combinations of a basis without a fused k x dim GEMM."""
    return jax.lax.map(
        lambda coeffs: jnp.tensordot(coeffs, basis_rows, axes=1),
        coeffs_matrix,
    )


def apply_eigen_preconditioner(
    grad_flat: Array,
    eigenvalues: Array,
    eigenvectors: Array,
    damping: float = 1e-4,
    saddle_free_neg: bool = False,
) -> Array:
    """Apply a partial Newton-like preconditioner in a global eigenbasis."""
    if eigenvalues.size == 0:
        return grad_flat

    V = eigenvectors
    lambdas = eigenvalues

    proj = V @ grad_flat
    proj_vec = V.T @ proj

    if saddle_free_neg:
        lam_eff = jnp.abs(lambdas)
        scale = 1.0 / (lam_eff + damping)
    else:
        scale = 1.0 / (lambdas + damping)

    # Keep EigenAdam-style sqrt scaling used by the PN-S implementations.
    scale = jnp.sqrt(scale)
    new_subspace = V.T @ (proj * scale)
    g_perp = grad_flat - proj_vec
    return new_subspace + g_perp


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
    nan = jnp.asarray(jnp.nan, dtype=dtype)
    return EigenTrackingState(
        step=jnp.array(0, dtype=jnp.int32),
        eigenvalues=jnp.zeros((k,), dtype=dtype),
        eigenvectors=jnp.zeros((k, dim), dtype=dtype),
        extra_eigenvalues=jnp.zeros((extra_modes,), dtype=dtype),
        extra_eigenvectors=jnp.zeros((extra_modes, dim), dtype=dtype),
        ritz_residual=jnp.full((k,), jnp.nan, dtype=dtype),
        extra_ritz_residual=jnp.full((extra_modes,), jnp.nan, dtype=dtype),
        g_proj=jnp.full((k,), jnp.nan, dtype=dtype),
        extra_g_proj=jnp.full((extra_modes,), jnp.nan, dtype=dtype),
        d_proj=jnp.full((k,), jnp.nan, dtype=dtype),
        extra_d_proj=jnp.full((extra_modes,), jnp.nan, dtype=dtype),
        update_energy_frac=jnp.full((k,), jnp.nan, dtype=dtype),
        extra_update_energy_frac=jnp.full((extra_modes,), jnp.nan, dtype=dtype),
        alpha=jnp.full((k,), jnp.nan, dtype=dtype),
        extra_alpha=jnp.full((extra_modes,), jnp.nan, dtype=dtype),
        phi=jnp.full((k,), jnp.nan, dtype=dtype),
        extra_phi=jnp.full((extra_modes,), jnp.nan, dtype=dtype),
        eos_rho=jnp.full((k,), jnp.nan, dtype=dtype),
        extra_eos_rho=jnp.full((extra_modes,), jnp.nan, dtype=dtype),
        alpha_valid=jnp.zeros((k,), dtype=bool),
        extra_alpha_valid=jnp.zeros((extra_modes,), dtype=bool),
        topk_eos_rho_max=nan,
        topk_eos_rho_mean=nan,
        topk_eos_rho_update_weighted=nan,
        topk_eos_rho_over_2_max=nan,
        topk_eos_rho_over_2_update_weighted=nan,
        eff_cond=jnp.array(0.0, dtype=dtype),
        rng_key=jax.random.PRNGKey(seed),
        rotation_diff=jnp.array(0.0, dtype=dtype),
        grad_norm=nan,
        update_norm=nan,
        tracked_update_energy_frac=nan,
        tracked_grad_energy_frac=nan,
        tracked_update_grad_cosine=nan,
        pos_update_energy_frac=nan,
        neg_update_energy_frac=nan,
        pos_grad_energy_frac=nan,
        neg_grad_energy_frac=nan,
        pos_update_grad_cosine=nan,
        neg_update_grad_cosine=nan,
        effective_curvature_cond=nan,
        actual_update_rayleigh=nan,
        effective_curvature_eigenvalues=jnp.full((k,), jnp.nan, dtype=dtype),
        extra_effective_curvature_eigenvalues=jnp.full(
            (extra_modes,), jnp.nan, dtype=dtype
        ),
        damped_effective_curvature_eigenvalues=jnp.full((k,), jnp.nan, dtype=dtype),
        extra_damped_effective_curvature_eigenvalues=jnp.full(
            (extra_modes,), jnp.nan, dtype=dtype
        ),
        preconditioned_dir_gain=jnp.full((k,), jnp.nan, dtype=dtype),
        extra_preconditioned_dir_gain=jnp.full((extra_modes,), jnp.nan, dtype=dtype),
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


def _condition_number_from_abs(values: Array, eps: float, nan: Array) -> Array:
    abs_values = jnp.abs(values)
    valid = jnp.logical_and(jnp.isfinite(abs_values), abs_values > eps)
    max_abs = jnp.max(jnp.where(valid, abs_values, 0.0))
    min_abs = jnp.min(jnp.where(valid, abs_values, jnp.inf))
    return jnp.where(jnp.any(valid), max_abs / jnp.maximum(min_abs, eps), nan)


def _sort_desc(values: Array) -> Array:
    return values[jnp.argsort(values)[::-1]]


def _effective_curvature_metrics(
    basis_rows: Array,
    update_flat: Array,
    *,
    matvec_flat: Callable[[Array], Array],
    effective_transform_flat: Callable[[Array], Array],
    eps: float,
    gram_ridge: float,
) -> Tuple[Array, Array, Array, Array, Array]:
    """Measure curvature after pushing tracked directions through an update map.

    Returns:
      generalized_eigs: eigenvalues of K a = mu G a, where
        K_ij = z_i^T H z_j and G_ij = z_i^T z_j.
      damped_eigs: eigenvalues of K itself, retaining update-map scale.
      gains: ||z_i|| for each raw tracked direction.
      effective_cond: abs-condition number of generalized_eigs.
      update_rayleigh: u^T H u / ||u||^2 for the actual/probe update.
    """
    metric_dtype = basis_rows.dtype
    nan = jnp.asarray(jnp.nan, dtype=metric_dtype)
    n = basis_rows.shape[0]

    def transform_row(row: Array) -> Array:
        return effective_transform_flat(row).astype(metric_dtype)

    z_rows = jax.lax.map(transform_row, basis_rows)
    hz_rows = jax.lax.map(matvec_flat, z_rows)
    gains = jnp.sqrt(jnp.sum(jnp.square(z_rows), axis=1))

    gram = z_rows @ z_rows.T
    curv = z_rows @ hz_rows.T
    gram = 0.5 * (gram + gram.T)
    curv = 0.5 * (curv + curv.T)

    eye = jnp.eye(n, dtype=metric_dtype)
    gram_scale = jnp.maximum(jnp.mean(jnp.diag(gram)), eps)
    ridge = jnp.asarray(gram_ridge, dtype=metric_dtype) * gram_scale
    gram_reg = gram + ridge * eye

    # Symmetric generalized eigensolve via whitening.  This is intentionally
    # ridge-stabilized because sign-like optimizers can collapse directions.
    chol = jnp.linalg.cholesky(gram_reg)
    whitened_left = jnp.linalg.solve(chol, curv)
    whitened = jnp.linalg.solve(chol, whitened_left.T).T
    whitened = 0.5 * (whitened + whitened.T)

    generalized_eigs = _sort_desc(jnp.linalg.eigvalsh(whitened))
    damped_eigs = _sort_desc(jnp.linalg.eigvalsh(curv))
    effective_cond = _condition_number_from_abs(generalized_eigs, eps, nan)

    hu_flat = matvec_flat(update_flat)
    update_energy = jnp.sum(jnp.square(update_flat))
    update_rayleigh = jnp.vdot(update_flat, hu_flat) / (update_energy + eps)

    return (
        generalized_eigs,
        damped_eigs,
        gains,
        effective_cond,
        update_rayleigh,
    )


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
    learning_rate: float = 1.0,
    signed_split_enabled: bool = False,
    eps: float = 1e-12,
    alpha_grad_tol_abs: float = 1e-10,
    alpha_grad_tol_rel: float = 1e-3,
    effective_transform_fn: Optional[Callable[[PyTree], PyTree]] = None,
    effective_curvature_gram_ridge: float = 1e-8,
) -> EigenTrackingState:
    flat_params, unravel_params = ravel_pytree(params)
    dim = flat_params.shape[0]
    grad_flat, _ = ravel_pytree(grads)
    upd_flat, _ = ravel_pytree(updates)
    metric_dtype = grad_flat.dtype
    nan = jnp.asarray(jnp.nan, dtype=metric_dtype)
    grad_energy = jnp.sum(jnp.square(grad_flat))
    update_energy = jnp.sum(jnp.square(upd_flat))
    grad_norm = jnp.sqrt(grad_energy)
    update_norm = jnp.sqrt(update_energy)

    rng_key, lanczos_key = jax.random.split(eigen_state.rng_key)
    k = eigen_state.eigenvalues.shape[0]
    extra_k = eigen_state.extra_eigenvalues.shape[0]
    total_keep = k + extra_k
    if total_keep == 0:
        return eigen_state._replace(
            step=step,
            rng_key=rng_key,
            grad_norm=grad_norm,
            update_norm=update_norm,
            tracked_update_energy_frac=jnp.array(0.0, dtype=metric_dtype),
            tracked_grad_energy_frac=jnp.array(0.0, dtype=metric_dtype),
            tracked_update_grad_cosine=nan,
            topk_eos_rho_max=nan,
            topk_eos_rho_mean=nan,
            topk_eos_rho_update_weighted=nan,
            topk_eos_rho_over_2_max=nan,
            topk_eos_rho_over_2_update_weighted=nan,
            pos_update_energy_frac=nan,
            neg_update_energy_frac=nan,
            pos_grad_energy_frac=nan,
            neg_grad_energy_frac=nan,
            pos_update_grad_cosine=nan,
            neg_update_grad_cosine=nan,
            effective_curvature_cond=nan,
            actual_update_rayleigh=nan,
        )

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

    evals, evecs, ritz_residuals = lanczos(
        matvec=matvec_flat,
        dim=dim,
        num_iter=lanczos_steps,
        key=lanczos_key,
        eps=eps,
        sort_by_abs=sort_by_abs,
        init_v=warm_start_v,
        use_light_ortho=use_light_ortho,
        light_ortho_every=light_ortho_every,
        return_residuals=True,
        max_return_vectors=total_keep,
    )

    eigenvalues = evals[:k]
    eigenvectors = evecs[:k, :]
    extra_eigenvalues = evals[k : k + extra_k]
    extra_eigenvectors = evecs[k : k + extra_k, :]
    # Relative Ritz residual: ||A v - lambda v|| / (|lambda| + eps).
    ritz_residual = ritz_residuals[:k] / (jnp.abs(eigenvalues) + eps)
    extra_ritz_residual = ritz_residuals[k : k + extra_k] / (
        jnp.abs(extra_eigenvalues) + eps
    )

    prev_vecs = eigen_state.eigenvectors
    eigenvectors = _align_eigenvector_rows(prev_vecs, eigenvectors)

    prev_extra_vecs = eigen_state.extra_eigenvectors
    extra_eigenvectors = _align_eigenvector_rows(prev_extra_vecs, extra_eigenvectors)

    rotation_diff = _subspace_rotation_diff(prev_vecs, eigenvectors, eps)

    all_eigenvalues = jnp.concatenate([eigenvalues, extra_eigenvalues], axis=0)
    all_eigenvectors = jnp.concatenate([eigenvectors, extra_eigenvectors], axis=0)

    effective_curvature_cond = nan
    actual_update_rayleigh = nan
    effective_curvature_eigs_all = jnp.full((total_keep,), jnp.nan, dtype=metric_dtype)
    damped_effective_curvature_eigs_all = jnp.full(
        (total_keep,), jnp.nan, dtype=metric_dtype
    )
    preconditioned_dir_gain_all = jnp.full((total_keep,), jnp.nan, dtype=metric_dtype)

    if effective_transform_fn is not None:

        def effective_transform_flat(v_flat: Array) -> Array:
            z_pytree = effective_transform_fn(unravel_params(v_flat))
            z_flat, _ = ravel_pytree(z_pytree)
            return z_flat

        (
            effective_curvature_eigs_all,
            damped_effective_curvature_eigs_all,
            preconditioned_dir_gain_all,
            effective_curvature_cond,
            actual_update_rayleigh,
        ) = _effective_curvature_metrics(
            all_eigenvectors,
            upd_flat,
            matvec_flat=matvec_flat,
            effective_transform_flat=effective_transform_flat,
            eps=eps,
            gram_ridge=effective_curvature_gram_ridge,
        )

    if total_keep > 0:
        g_proj = _project_rows(all_eigenvectors, grad_flat)
        d_proj = _project_rows(all_eigenvectors, upd_flat)
        g_proj_energy = jnp.sum(jnp.square(g_proj))
        d_proj_energy = jnp.sum(jnp.square(d_proj))
        update_energy_frac_all = jnp.square(d_proj) / (update_energy + eps)
        tracked_update_energy_frac = d_proj_energy / (update_energy + eps)
        tracked_grad_energy_frac = g_proj_energy / (grad_energy + eps)
        tracked_update_grad_cosine = -jnp.sum(d_proj * g_proj) / (
            jnp.sqrt(d_proj_energy) * jnp.sqrt(g_proj_energy) + eps
        )

        def _split_metrics(mask: Array) -> tuple[Array, Array, Array]:
            masked_g_proj = jnp.where(mask, g_proj, 0.0)
            masked_d_proj = jnp.where(mask, d_proj, 0.0)
            masked_g_energy = jnp.sum(jnp.square(masked_g_proj))
            masked_d_energy = jnp.sum(jnp.square(masked_d_proj))
            update_frac = masked_d_energy / (update_energy + eps)
            grad_frac = masked_g_energy / (grad_energy + eps)
            cosine = -jnp.sum(masked_d_proj * masked_g_proj) / (
                jnp.sqrt(masked_d_energy) * jnp.sqrt(masked_g_energy) + eps
            )
            has_modes = jnp.any(mask)
            return (
                jnp.where(has_modes, update_frac, nan),
                jnp.where(has_modes, grad_frac, nan),
                jnp.where(has_modes, cosine, nan),
            )

        if signed_split_enabled:
            pos_mask = all_eigenvalues > eps
            neg_mask = all_eigenvalues < -eps
            (
                pos_update_energy_frac,
                pos_grad_energy_frac,
                pos_update_grad_cosine,
            ) = _split_metrics(pos_mask)
            (
                neg_update_energy_frac,
                neg_grad_energy_frac,
                neg_update_grad_cosine,
            ) = _split_metrics(neg_mask)
        else:
            pos_update_energy_frac = nan
            neg_update_energy_frac = nan
            pos_grad_energy_frac = nan
            neg_grad_energy_frac = nan
            pos_update_grad_cosine = nan
            neg_update_grad_cosine = nan

        # Relative threshold is taken against the largest projected gradient in
        # the tracked modes, with an absolute floor to keep near-zero ratios sane.
        g_ref = jnp.maximum(jnp.max(jnp.abs(g_proj)), eps)
        g_tol = jnp.maximum(
            jnp.asarray(alpha_grad_tol_abs, dtype=g_proj.dtype),
            jnp.asarray(alpha_grad_tol_rel, dtype=g_proj.dtype) * g_ref,
        )
        alpha_valid = jnp.abs(g_proj) > g_tol

        safe_g_proj = jnp.where(alpha_valid, g_proj, 1.0)
        alpha_raw = -d_proj / safe_g_proj
        alpha_all = jnp.where(alpha_valid, alpha_raw, jnp.nan)

        lr = jnp.asarray(learning_rate, dtype=alpha_all.dtype)
        safe_lr = jnp.where(jnp.abs(lr) > eps, lr, jnp.nan)
        phi_raw = alpha_raw * all_eigenvalues / safe_lr
        phi_all = jnp.where(alpha_valid, phi_raw, jnp.nan)
        eos_rho_raw = jnp.abs(alpha_raw * all_eigenvalues)
        eos_rho_all = jnp.where(alpha_valid, eos_rho_raw, jnp.nan)

        alpha = alpha_all[:k]
        extra_alpha = alpha_all[k : k + extra_k]
        phi = phi_all[:k]
        extra_phi = phi_all[k : k + extra_k]
        eos_rho = eos_rho_all[:k]
        extra_eos_rho = eos_rho_all[k : k + extra_k]
        top_g_proj = g_proj[:k]
        extra_g_proj = g_proj[k : k + extra_k]
        top_d_proj = d_proj[:k]
        extra_d_proj = d_proj[k : k + extra_k]
        top_update_energy_frac = update_energy_frac_all[:k]
        extra_update_energy_frac = update_energy_frac_all[k : k + extra_k]
        top_alpha_valid = alpha_valid[:k]
        extra_alpha_valid = alpha_valid[k : k + extra_k]

        valid_top_rho = jnp.logical_and(
            top_alpha_valid,
            jnp.isfinite(eos_rho),
        )
        top_rho_count = jnp.sum(valid_top_rho.astype(metric_dtype))
        top_rho_sum = jnp.sum(jnp.where(valid_top_rho, eos_rho, 0.0))
        topk_eos_rho_mean = jnp.where(
            top_rho_count > 0.0,
            top_rho_sum / jnp.maximum(top_rho_count, 1.0),
            nan,
        )
        topk_eos_rho_max = jnp.where(
            jnp.any(valid_top_rho),
            jnp.max(jnp.where(valid_top_rho, eos_rho, 0.0)),
            nan,
        )
        rho_weights = jnp.where(valid_top_rho, top_update_energy_frac, 0.0)
        rho_weight_sum = jnp.sum(rho_weights)
        topk_eos_rho_update_weighted = jnp.where(
            rho_weight_sum > eps,
            jnp.sum(rho_weights * jnp.where(valid_top_rho, eos_rho, 0.0))
            / rho_weight_sum,
            nan,
        )
        topk_eos_rho_over_2_max = topk_eos_rho_max / 2.0
        topk_eos_rho_over_2_update_weighted = (
            topk_eos_rho_update_weighted / 2.0
        )

        phi_abs = jnp.abs(jnp.where(top_alpha_valid, phi_raw[:k], 0.0))
        valid_for_cond = jnp.logical_and(top_alpha_valid, phi_abs > eps)

        max_abs = jnp.max(jnp.where(valid_for_cond, phi_abs, 0.0))
        min_abs = jnp.min(
            jnp.where(
                valid_for_cond,
                phi_abs,
                jnp.full_like(phi_abs, jnp.inf),
            )
        )
        eff_cond = jnp.where(
            jnp.any(valid_for_cond),
            max_abs / jnp.maximum(min_abs, eps),
            jnp.array(0.0, dtype=phi.dtype),
        )
    else:
        rotation_diff = jnp.array(0.0, dtype=eigenvalues.dtype)
        alpha = eigen_state.alpha
        extra_alpha = eigen_state.extra_alpha
        phi = eigen_state.phi
        extra_phi = eigen_state.extra_phi
        eos_rho = eigen_state.eos_rho
        extra_eos_rho = eigen_state.extra_eos_rho
        top_g_proj = eigen_state.g_proj
        extra_g_proj = eigen_state.extra_g_proj
        top_d_proj = eigen_state.d_proj
        extra_d_proj = eigen_state.extra_d_proj
        top_update_energy_frac = eigen_state.update_energy_frac
        extra_update_energy_frac = eigen_state.extra_update_energy_frac
        top_alpha_valid = eigen_state.alpha_valid
        extra_alpha_valid = eigen_state.extra_alpha_valid
        topk_eos_rho_max = eigen_state.topk_eos_rho_max
        topk_eos_rho_mean = eigen_state.topk_eos_rho_mean
        topk_eos_rho_update_weighted = eigen_state.topk_eos_rho_update_weighted
        topk_eos_rho_over_2_max = eigen_state.topk_eos_rho_over_2_max
        topk_eos_rho_over_2_update_weighted = (
            eigen_state.topk_eos_rho_over_2_update_weighted
        )
        eff_cond = jnp.array(0.0, dtype=eigenvalues.dtype)
        tracked_update_energy_frac = eigen_state.tracked_update_energy_frac
        tracked_grad_energy_frac = eigen_state.tracked_grad_energy_frac
        tracked_update_grad_cosine = eigen_state.tracked_update_grad_cosine
        pos_update_energy_frac = eigen_state.pos_update_energy_frac
        neg_update_energy_frac = eigen_state.neg_update_energy_frac
        pos_grad_energy_frac = eigen_state.pos_grad_energy_frac
        neg_grad_energy_frac = eigen_state.neg_grad_energy_frac
        pos_update_grad_cosine = eigen_state.pos_update_grad_cosine
        neg_update_grad_cosine = eigen_state.neg_update_grad_cosine

    return eigen_state._replace(
        step=step,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        extra_eigenvalues=extra_eigenvalues,
        extra_eigenvectors=extra_eigenvectors,
        ritz_residual=ritz_residual,
        extra_ritz_residual=extra_ritz_residual,
        g_proj=top_g_proj,
        extra_g_proj=extra_g_proj,
        d_proj=top_d_proj,
        extra_d_proj=extra_d_proj,
        update_energy_frac=top_update_energy_frac,
        extra_update_energy_frac=extra_update_energy_frac,
        alpha=alpha,
        extra_alpha=extra_alpha,
        phi=phi,
        extra_phi=extra_phi,
        eos_rho=eos_rho,
        extra_eos_rho=extra_eos_rho,
        alpha_valid=top_alpha_valid,
        extra_alpha_valid=extra_alpha_valid,
        topk_eos_rho_max=topk_eos_rho_max,
        topk_eos_rho_mean=topk_eos_rho_mean,
        topk_eos_rho_update_weighted=topk_eos_rho_update_weighted,
        topk_eos_rho_over_2_max=topk_eos_rho_over_2_max,
        topk_eos_rho_over_2_update_weighted=topk_eos_rho_over_2_update_weighted,
        eff_cond=eff_cond,
        rng_key=rng_key,
        rotation_diff=rotation_diff,
        grad_norm=grad_norm,
        update_norm=update_norm,
        tracked_update_energy_frac=tracked_update_energy_frac,
        tracked_grad_energy_frac=tracked_grad_energy_frac,
        tracked_update_grad_cosine=tracked_update_grad_cosine,
        pos_update_energy_frac=pos_update_energy_frac,
        neg_update_energy_frac=neg_update_energy_frac,
        pos_grad_energy_frac=pos_grad_energy_frac,
        neg_grad_energy_frac=neg_grad_energy_frac,
        pos_update_grad_cosine=pos_update_grad_cosine,
        neg_update_grad_cosine=neg_update_grad_cosine,
        effective_curvature_cond=effective_curvature_cond,
        actual_update_rayleigh=actual_update_rayleigh,
        effective_curvature_eigenvalues=effective_curvature_eigs_all[:k],
        extra_effective_curvature_eigenvalues=effective_curvature_eigs_all[
            k : k + extra_k
        ],
        damped_effective_curvature_eigenvalues=damped_effective_curvature_eigs_all[:k],
        extra_damped_effective_curvature_eigenvalues=damped_effective_curvature_eigs_all[
            k : k + extra_k
        ],
        preconditioned_dir_gain=preconditioned_dir_gain_all[:k],
        extra_preconditioned_dir_gain=preconditioned_dir_gain_all[k : k + extra_k],
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
    return_residuals: bool = False,
    max_return_vectors: Optional[int] = None,
) -> tuple[Array, ...]:
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
    residuals = jnp.abs(betas[num_iter - 1] * evecs_t[-1, :])

    if max_return_vectors is not None:
        keep = min(int(max_return_vectors), num_iter)
        evecs_t = evecs_t[:, :keep]

    v_k = v_basis[:-1]
    eigenvectors = _expand_from_basis(evecs_t.T, v_k).reshape(evecs_t.shape[1], dim)
    if return_residuals:
        return evals, eigenvectors, residuals
    return evals, eigenvectors
