from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
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
    full_update_grad_cosine: Array   # full-space update–gradient cosine (Eq. 5 first-order alignment)
    full_update_grad_cos2: Array     # its square
    pos_update_energy_frac: Array
    neg_update_energy_frac: Array
    pos_grad_energy_frac: Array
    neg_grad_energy_frac: Array
    pos_update_grad_cosine: Array
    neg_update_grad_cosine: Array
    effective_curvature_cond: Array
    actual_update_rayleigh: Array
    # A4.1: the Lanczos tridiagonal (~2*num_iter floats). The leading j x j
    # principal submatrix of T gives the exact j-iteration Ritz values for every
    # j <= num_iter, so "is num_iter more than we need" is answerable offline
    # per optimizer. It also exposes Lanczos breakdown: a beta collapsing to ~0
    # makes the reported Ritz residual tiny while the Ritz value is garbage.
    lanczos_alpha: Array
    lanczos_beta: Array
    effective_curvature_eigenvalues: Array
    extra_effective_curvature_eigenvalues: Array
    damped_effective_curvature_eigenvalues: Array
    extra_damped_effective_curvature_eigenvalues: Array
    preconditioned_dir_gain: Array
    extra_preconditioned_dir_gain: Array
    # Momentum-reference projections (NaN when measurement_momentum=False)
    m_proj: Array
    extra_m_proj: Array
    m_norm: Array
    # Experiment 1b — preconditioner-basis vs curvature alignment
    A_P_believed: Array       # Σ_i p̂_i a_i  (preconditioner-weighted curvature)
    A_Muon_block: Array       # (1/M) Σ_i a_i  (uniform / Muon reference)
    crit_resid: Array         # (A_P − A_Muon) − M·Cov(p̂, a) ≈ 0 sanity check
    precond_basis_sin2: Array # 1 − mean(σ²) of V @ Q_hat^T; 0 = aligned
    a_per_dir: Array          # (num_precond_dirs,) per-direction curvatures
    phat_per_dir: Array       # (num_precond_dirs,) per-direction p̂ weights


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
    num_precond_dirs: int = 0,
    lanczos_iters: int = 0,
    seed: int = 0,
) -> EigenTrackingState:
    flat_params, _ = ravel_pytree(params)
    dim = flat_params.shape[0]
    dtype = flat_params.dtype
    # Eigenvectors are (k, n_params) — at 162M params they are ~3.85 GiB each in fp32.
    # Use bf16 to halve that; scalar metrics remain in dtype (fp32).
    vec_dtype = jnp.bfloat16
    nan = jnp.asarray(jnp.nan, dtype=dtype)
    return EigenTrackingState(
        step=jnp.array(0, dtype=jnp.int32),
        lanczos_alpha=jnp.full((lanczos_iters,), jnp.nan, dtype=dtype),
        lanczos_beta=jnp.full((lanczos_iters,), jnp.nan, dtype=dtype),
        eigenvalues=jnp.zeros((k,), dtype=dtype),
        eigenvectors=jnp.zeros((k, dim), dtype=vec_dtype),
        extra_eigenvalues=jnp.zeros((extra_modes,), dtype=dtype),
        extra_eigenvectors=jnp.zeros((extra_modes, dim), dtype=vec_dtype),
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
        full_update_grad_cosine=nan,
        full_update_grad_cos2=nan,
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
        m_proj=jnp.full((k,), jnp.nan, dtype=dtype),
        extra_m_proj=jnp.full((extra_modes,), jnp.nan, dtype=dtype),
        m_norm=nan,
        A_P_believed=nan,
        A_Muon_block=nan,
        crit_resid=nan,
        precond_basis_sin2=nan,
        a_per_dir=jnp.full((num_precond_dirs,), jnp.nan, dtype=dtype),
        phat_per_dir=jnp.full((num_precond_dirs,), jnp.nan, dtype=dtype),
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
    # Gram/curvature matrices go through potrf (Cholesky); bf16 is unsupported.
    metric_dtype = jnp.float32
    nan = jnp.asarray(jnp.nan, dtype=metric_dtype)
    n = basis_rows.shape[0]

    def transform_row(row: Array) -> Array:
        return effective_transform_flat(row).astype(metric_dtype)

    # Stream rows instead of materializing z_rows and H z_rows.  For LM-scale
    # models, each row is a full parameter vector, so keeping transformed basis
    # matrices resident can dominate eigentracking memory.
    gram0 = jnp.zeros((n, n), dtype=metric_dtype)
    curv0 = jnp.zeros((n, n), dtype=metric_dtype)
    gains0 = jnp.zeros((n,), dtype=metric_dtype)

    def outer_body(i: int, carry):
        gram, curv, gains = carry
        z_i = transform_row(basis_rows[i])
        hz_i = matvec_flat(z_i).astype(metric_dtype)
        gains = gains.at[i].set(jnp.sqrt(jnp.sum(jnp.square(z_i))))

        gram_row0 = jnp.zeros((n,), dtype=metric_dtype)
        curv_row0 = jnp.zeros((n,), dtype=metric_dtype)

        def inner_body(j: int, row_carry):
            gram_row, curv_row = row_carry
            z_j = transform_row(basis_rows[j])
            gram_row = gram_row.at[j].set(jnp.vdot(z_i, z_j))
            # H is intended to be symmetric; using z_j^T H z_i lets us reuse the
            # single streamed matvec H z_i for this row. We symmetrize below.
            curv_row = curv_row.at[j].set(jnp.vdot(z_j, hz_i))
            return gram_row, curv_row

        gram_row, curv_row = jax.lax.fori_loop(
            0,
            n,
            inner_body,
            (gram_row0, curv_row0),
        )
        gram = gram.at[i, :].set(gram_row)
        curv = curv.at[i, :].set(curv_row)
        return gram, curv, gains

    gram, curv, gains = jax.lax.fori_loop(
        0,
        n,
        outer_body,
        (gram0, curv0, gains0),
    )
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


def _iter_state_children(state):
    """Yield direct children of an optax-style state node for structure traversal."""
    if hasattr(state, '_fields'):
        for f in state._fields:
            yield getattr(state, f)
    elif isinstance(state, dict):
        yield from state.values()
    elif isinstance(state, (list, tuple)):
        yield from state


def _find_first_of_type(state, target_type):
    """Return the first node in the state pytree that is an instance of target_type."""
    if isinstance(state, target_type):
        return state
    if isinstance(state, (dict, list, tuple)) or hasattr(state, '_fields'):
        for child in _iter_state_children(state):
            result = _find_first_of_type(child, target_type)
            if result is not None:
                return result
    return None


def _combine_masked_pytrees(primary, fallback):
    """Walk two pytrees with identical structure; for each leaf use primary unless it is a MaskedNode."""
    try:
        masked_cls = optax.MaskedNode
    except AttributeError:
        return primary
    if isinstance(primary, masked_cls):
        return fallback
    if isinstance(primary, dict):
        return {k: _combine_masked_pytrees(primary[k], fallback[k]) for k in primary}
    if hasattr(primary, '_fields'):
        return type(primary)(
            *[_combine_masked_pytrees(getattr(primary, f), getattr(fallback, f))
              for f in primary._fields]
        )
    if isinstance(primary, (list, tuple)):
        merged = [_combine_masked_pytrees(p, f) for p, f in zip(primary, fallback)]
        return type(primary)(merged)
    return primary


def extract_momentum_pytree(opt_state, optim_name: str):
    """
    Extract the post-momentum / pre-preconditioning intermediate vector from the
    optimizer state.  Returns a pytree matching the params structure, or None if
    the optimizer is not recognised or has no applicable momentum buffer.

    Concretely:
      adam / adamw      : ScaleByAdamState.mu  (first-moment EMA)
      soap / ni_soap    : SoapPerParamState.m / NiSoapPerParamState.m  (momentum before Kronecker projection)
      muon              : MuonState.mu (Muon params) combined with ScaleByAdamState.mu (Adam params)
      signum            : SignumState.momentum_buffer  (before sign)
      sgd               : TraceState.trace  (velocity / momentum buffer; equals grad when momentum=0)
    """
    name = optim_name.lower().strip()

    # ---- Adam / AdamW --------------------------------------------------------
    if name in {"adam", "adamw"}:
        ScaleByAdamState = getattr(optax, "ScaleByAdamState", None)
        if ScaleByAdamState is None:
            try:
                from optax._src.transform import ScaleByAdamState
            except ImportError:
                ScaleByAdamState = None
        if ScaleByAdamState is not None:
            node = _find_first_of_type(opt_state, ScaleByAdamState)
            if node is not None:
                return node.mu
        return None

    # ---- SOAP / NI-SOAP -------------------------------------------------------
    if name in {"soap", "ni_soap", "ni-soap", "nisoap"}:
        from optim.soap import SoapState, SoapPerParamState
        from optim.ni_soap import NiSoapState, NiSoapPerParamState

        soap_node = _find_first_of_type(opt_state, SoapState)
        if soap_node is not None:
            # per_param is a pytree[SoapPerParamState]; extract .m from each leaf
            return jax.tree_util.tree_map(
                lambda s: s.m,
                soap_node.per_param,
                is_leaf=lambda x: isinstance(x, SoapPerParamState),
            )
        ni_soap_node = _find_first_of_type(opt_state, NiSoapState)
        if ni_soap_node is not None:
            return jax.tree_util.tree_map(
                lambda s: s.m,
                ni_soap_node.per_param,
                is_leaf=lambda x: isinstance(x, NiSoapPerParamState),
            )
        return None

    # ---- Muon -----------------------------------------------------------------
    if name == "muon":
        try:
            from optax.contrib._muon import MuonState
        except ImportError:
            MuonState = None
        ScaleByAdamState = getattr(optax, "ScaleByAdamState", None)
        if ScaleByAdamState is None:
            try:
                from optax._src.transform import ScaleByAdamState
            except ImportError:
                ScaleByAdamState = None

        if MuonState is None or ScaleByAdamState is None:
            return None

        muon_node = _find_first_of_type(opt_state, MuonState)
        adam_node = _find_first_of_type(opt_state, ScaleByAdamState)
        if muon_node is None and adam_node is None:
            return None
        if muon_node is None:
            return adam_node.mu if adam_node is not None else None
        if adam_node is None:
            return muon_node.mu

        # Both exist: combine, giving priority to muon momentum where it's real.
        return _combine_masked_pytrees(muon_node.mu, adam_node.mu)

    # ---- Signum ---------------------------------------------------------------
    if name in {"signum", "sign_sgd", "sign-sgd", "signsgd"}:
        from optim.signum import SignumState
        node = _find_first_of_type(opt_state, SignumState)
        if node is not None:
            return node.momentum_buffer
        return None

    # ---- SGD (with or without momentum) --------------------------------------
    if name in {"sgd", "momentum", "sgdm", "sgd_momentum", "sgd-momentum"}:
        TraceState = getattr(optax, "TraceState", None)
        if TraceState is None:
            try:
                from optax._src.transform import TraceState
            except ImportError:
                TraceState = None
        if TraceState is not None:
            node = _find_first_of_type(opt_state, TraceState)
            if node is not None:
                return node.trace
        return None

    return None


# ---------------------------------------------------------------------------
# Experiment 1b helpers: preconditioner-basis vs curvature alignment
# ---------------------------------------------------------------------------

def _precond_criterion_metrics(
    Q_hat: Array,
    lambda_hat: Array,
    V: Array,
    matvec_flat: Callable[[Array], Array],
    num_store: int,
    *,
    damping: float,
    eps: float,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Compute Exp 1b criterion metrics.

    Args:
        Q_hat: (M, dim) bfloat16 — believed preconditioner eigenvectors (unit rows).
        lambda_hat: (M,) float32 — believed preconditioner eigenvalues.
        V: (k, dim) bfloat16 — measured curvature eigenvectors from Lanczos.
        matvec_flat: curvature matrix-vector product in flat param space.
        num_store: number of per-direction entries to store (0 → empty arrays).
        damping: ridge added to lambda_hat before inverting for p̂.
        eps: numerical floor.

    Returns (A_P, A_Muon, crit_resid, precond_basis_sin2, a_store, phat_store).
    """
    M = Q_hat.shape[0]
    metric_dtype = jnp.float32
    nan = jnp.asarray(jnp.nan, dtype=metric_dtype)

    # a_i = q_i^T H q_i  (one Hvp per believed direction)
    def rayleigh(q_bf16: Array) -> Array:
        q = q_bf16.astype(metric_dtype)
        hq = matvec_flat(q).astype(metric_dtype)
        return jnp.vdot(q, hq)

    a_per_dir = jax.lax.map(rayleigh, Q_hat)  # (M,) float32

    # p̂_i = (1/(λ̂_i+δ)) / Σ_j 1/(λ̂_j+δ)
    lam = lambda_hat.astype(metric_dtype)
    inv_lam = 1.0 / (lam + jnp.asarray(damping, metric_dtype))
    phat = inv_lam / jnp.maximum(jnp.sum(inv_lam), eps)  # (M,)

    A_P = jnp.sum(phat * a_per_dir)
    M_f = jnp.asarray(M, dtype=metric_dtype)
    A_Muon = jnp.sum(a_per_dir) / M_f

    # crit_resid = (A_P − A_Muon) − M·Cov(p̂, a)  ≈ 0 by identity (sanity check)
    cov_phat_a = jnp.mean(phat * a_per_dir) - jnp.mean(phat) * jnp.mean(a_per_dir)
    crit_resid = (A_P - A_Muon) - M_f * cov_phat_a

    # precond_basis_sin2 = 1 − mean(σ²) where σ = SVD(V @ Q_hat^T)
    overlap = V.astype(metric_dtype) @ Q_hat.astype(metric_dtype).T  # (k, M)
    sing_vals = jnp.linalg.svd(overlap, compute_uv=False)            # min(k,M)
    precond_basis_sin2 = jnp.maximum(0.0, 1.0 - jnp.mean(jnp.square(sing_vals)))

    # Store per-direction arrays, truncated to num_store
    store_len = min(M, num_store)
    a_store = jnp.full((num_store,), jnp.nan, dtype=metric_dtype).at[:store_len].set(
        a_per_dir[:store_len]
    )
    phat_store = jnp.full((num_store,), jnp.nan, dtype=metric_dtype).at[:store_len].set(
        phat[:store_len]
    )
    return A_P, A_Muon, crit_resid, precond_basis_sin2, a_store, phat_store


class PrecondDirections(NamedTuple):
    """Compact description of the preconditioner's believed eigendirections.

    A dense ``(M, dim)`` basis is not representable at LM scale — M=32 over
    162M params is 20 GB in float32, which is what used to OOM-hold the Adam and
    SOAP seed-sweep jobs — and it is also entirely redundant: Adam's rows are
    one-hot, and SOAP's rows are rank-1 Kronecker products supported on a single
    2-D leaf. This keeps the generators instead, and
    :func:`precond_criterion_from_directions` materializes one direction at a
    time as a params-shaped pytree (peak cost: one extra copy of the params).

    Row ``d`` of the implied basis is a unit vector, zero outside leaf
    ``leaf_idx[d]``, equal to ``outer(u[d], v[d])`` (SOAP) or to the one-hot at
    ``coord[d]`` (Adam) inside it.
    """

    kind: str                # "soap" | "adam"
    lambda_hat: np.ndarray   # (M,) believed eigenvalue
    stratum: np.ndarray      # (M,) 0 = top half, 1 = bottom half
    leaf_idx: np.ndarray     # (M,) index into jax.tree_util.tree_leaves(params)
    leaf_offset: np.ndarray  # (M,) flat-space offset of that leaf
    rows: np.ndarray         # (M,)
    cols: np.ndarray         # (M,)
    coord: np.ndarray        # (M,) Adam: flat index inside the leaf; SOAP: -1
    u: tuple = ()            # SOAP: per-direction (r,) float32 left factor
    v: tuple = ()            # SOAP: per-direction (c,) float32 right factor

    @property
    def num_dirs(self) -> int:
        return int(self.lambda_hat.shape[0])


def _stratified_indices(values: np.ndarray, max_dirs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of the largest and smallest ``values``, half the budget each.

    Returns ``(idx, stratum)`` with ``stratum`` 0 for the top half and 1 for the
    bottom half, descending by ``values`` inside each half. A top-only selection
    only ever samples the stiff end of the believed spectrum, which is where a
    preconditioner is least wrong; the calibration question needs both ends.
    An odd budget gives the extra direction to the top half.
    """
    n = int(values.shape[0])
    M = int(min(max_dirs, n))
    if M <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)
    n_top = M - M // 2
    n_bot = M - n_top
    order = np.argsort(-values, kind="stable")
    idx = np.concatenate([order[:n_top], order[n - n_bot:]]) if n_bot else order[:n_top]
    stratum = np.concatenate(
        [np.zeros(n_top, dtype=np.int32), np.ones(n_bot, dtype=np.int32)]
    )
    return idx.astype(np.int64), stratum


def _extract_soap_precond_directions(
    opt_state,
    is_ni_soap: bool,
    param_leaves: list,
    leaf_offsets: list,
    max_dirs: int,
    stratified: bool = True,
) -> Optional[PrecondDirections]:
    """Kronecker eigendirections of the SOAP / NI-SOAP preconditioner.

    ``lambda_hat[d] = |eL[i] * eR[j]|`` is the believed curvature along
    ``outer(QL[:, i], QR[:, j])``. Candidates are pre-selected per leaf (a leaf
    contributes at most ``max_dirs`` from each end) so the global selection never
    materializes the full ``sum_leaves r*c`` candidate list.
    """
    from optim.soap import SoapState, SoapPerParamState
    from optim.ni_soap import NiSoapState, NiSoapPerParamState

    if is_ni_soap:
        StateClass, PerParamClass = NiSoapState, NiSoapPerParamState
    else:
        StateClass, PerParamClass = SoapState, SoapPerParamState

    node = _find_first_of_type(opt_state, StateClass)
    if node is None:
        return None

    per_param_leaves = jax.tree_util.tree_leaves(
        node.per_param, is_leaf=lambda x: isinstance(x, PerParamClass)
    )
    if len(per_param_leaves) != len(param_leaves):
        return None

    cand_lam: list = []
    cand_leaf: list = []
    cand_qi: list = []
    cand_qj: list = []
    factors: dict = {}  # leaf_idx -> (QL, QR, r, c)

    for leaf_idx, (p_leaf, s_leaf) in enumerate(zip(param_leaves, per_param_leaves)):
        p_np = np.asarray(jax.device_get(p_leaf))
        if p_np.ndim != 2:
            continue
        r, c = p_np.shape

        # Transfer only the small factor matrices, not the large m/v buffers.
        QL = np.asarray(jax.device_get(s_leaf.QL), dtype=np.float32)
        QR = np.asarray(jax.device_get(s_leaf.QR), dtype=np.float32)
        if QL.shape[0] <= 1 or QR.shape[0] <= 1 or QL.shape[0] != r or QR.shape[0] != c:
            continue  # preconditioner not initialised or mismatched

        if is_ni_soap:
            eL = np.asarray(jax.device_get(s_leaf.eL), dtype=np.float32)
            eR = np.asarray(jax.device_get(s_leaf.eR), dtype=np.float32)
        else:
            L = np.asarray(jax.device_get(s_leaf.L), dtype=np.float32)
            R = np.asarray(jax.device_get(s_leaf.R), dtype=np.float32)
            eL = np.diag(QL.T @ L @ QL)
            eR = np.diag(QR.T @ R @ QR)

        factors[leaf_idx] = (QL, QR, r, c)

        kron = np.abs(np.outer(eL, eR)).ravel()  # (r*c,)
        # Per-leaf pre-selection: the global extremes are a subset of the
        # per-leaf extremes, so this is exact and caps peak memory at one leaf.
        keep = min(int(max_dirs), kron.size)
        if 2 * keep < kron.size:
            part = np.argpartition(kron, (keep - 1, kron.size - keep))
            # np.unique also guards the 2*keep >= size case, where the two
            # slices would overlap and offer the same coordinate twice.
            sel = np.unique(np.concatenate([part[:keep], part[-keep:]]))
        else:
            sel = np.arange(kron.size)
        cand_lam.append(kron[sel])
        cand_leaf.append(np.full(sel.size, leaf_idx, dtype=np.int32))
        cand_qi.append((sel // c).astype(np.int32))
        cand_qj.append((sel % c).astype(np.int32))

    if not cand_lam:
        return None

    lam_all = np.concatenate(cand_lam).astype(np.float32)
    leaf_all = np.concatenate(cand_leaf)
    qi_all = np.concatenate(cand_qi)
    qj_all = np.concatenate(cand_qj)

    if stratified:
        idx, stratum = _stratified_indices(lam_all, max_dirs)
    else:
        M = int(min(max_dirs, lam_all.size))
        idx = np.argsort(-lam_all, kind="stable")[:M]
        stratum = np.zeros(M, dtype=np.int32)

    u_list, v_list = [], []
    rows = np.zeros(idx.size, dtype=np.int64)
    cols = np.zeros(idx.size, dtype=np.int64)
    offs = np.zeros(idx.size, dtype=np.int64)
    for k, j in enumerate(idx):
        leaf_idx = int(leaf_all[j])
        QL, QR, r, c = factors[leaf_idx]
        u = QL[:, int(qi_all[j])].astype(np.float32)
        v = QR[:, int(qj_all[j])].astype(np.float32)
        # QL/QR are orthonormal so ||outer(u, v)|| == 1 already; renormalize
        # defensively so a drifted factor cannot silently rescale a_d.
        u = u / (np.linalg.norm(u) + 1e-12)
        v = v / (np.linalg.norm(v) + 1e-12)
        u_list.append(u)
        v_list.append(v)
        rows[k], cols[k] = r, c
        offs[k] = leaf_offsets[leaf_idx]

    return PrecondDirections(
        kind="soap",
        lambda_hat=lam_all[idx].astype(np.float32),
        stratum=stratum,
        leaf_idx=leaf_all[idx].astype(np.int64),
        leaf_offset=offs,
        rows=rows,
        cols=cols,
        coord=np.full(idx.size, -1, dtype=np.int64),
        u=tuple(u_list),
        v=tuple(v_list),
    )


def _extract_adam_precond_directions(
    opt_state,
    param_leaves: list,
    leaf_offsets: list,
    max_dirs: int,
    stratified: bool = True,
) -> Optional[PrecondDirections]:
    """Coordinate directions of the Adam / AdamW preconditioner.

    Rows are standard-basis vectors; ``lambda_hat`` is the second moment nu_i.
    """
    ScaleByAdamState = getattr(optax, "ScaleByAdamState", None)
    if ScaleByAdamState is None:
        try:
            from optax._src.transform import ScaleByAdamState  # type: ignore[assignment]
        except ImportError:
            return None

    node = _find_first_of_type(opt_state, ScaleByAdamState)
    if node is None:
        return None

    nu_leaves = jax.tree_util.tree_leaves(node.nu)
    if len(nu_leaves) != len(param_leaves):
        return None

    # Per-leaf pre-selection keeps this O(sum_leaves leaf_size) in time but
    # allocates only per-leaf temporaries.
    cand_val: list = []
    cand_leaf: list = []
    cand_coord: list = []
    for leaf_idx, nu_leaf in enumerate(nu_leaves):
        nu_np = np.asarray(jax.device_get(nu_leaf), dtype=np.float32).ravel()
        if nu_np.size == 0:
            continue
        keep = min(int(max_dirs), nu_np.size)
        if 2 * keep < nu_np.size:
            part = np.argpartition(nu_np, (keep - 1, nu_np.size - keep))
            # np.unique also guards the 2*keep >= size case, where the two
            # slices would overlap and offer the same coordinate twice.
            sel = np.unique(np.concatenate([part[:keep], part[-keep:]]))
        else:
            sel = np.arange(nu_np.size)
        cand_val.append(nu_np[sel])
        cand_leaf.append(np.full(sel.size, leaf_idx, dtype=np.int64))
        cand_coord.append(sel.astype(np.int64))

    if not cand_val:
        return None

    val_all = np.concatenate(cand_val).astype(np.float32)
    leaf_all = np.concatenate(cand_leaf)
    coord_all = np.concatenate(cand_coord)

    if stratified:
        idx, stratum = _stratified_indices(val_all, max_dirs)
    else:
        M = int(min(max_dirs, val_all.size))
        idx = np.argsort(-val_all, kind="stable")[:M]
        stratum = np.zeros(M, dtype=np.int32)

    sel_leaf = leaf_all[idx]
    return PrecondDirections(
        kind="adam",
        lambda_hat=val_all[idx].astype(np.float32),
        stratum=stratum,
        leaf_idx=sel_leaf,
        leaf_offset=np.array([leaf_offsets[int(i)] for i in sel_leaf], dtype=np.int64),
        rows=np.array(
            [int(np.asarray(param_leaves[int(i)]).size) for i in sel_leaf], dtype=np.int64
        ),
        cols=np.ones(idx.size, dtype=np.int64),
        coord=coord_all[idx],
    )


def _leaf_offsets_of(params: Params) -> Tuple[list, list, int]:
    """(leaves, flat offset of each leaf, total dim) in ravel_pytree order."""
    leaves = jax.tree_util.tree_leaves(params)
    offsets: list = []
    offset = 0
    for leaf in leaves:
        offsets.append(offset)
        offset += int(np.prod(np.shape(leaf)))
    return leaves, offsets, offset


def extract_precond_directions(
    opt_state,
    optim_name: str,
    params: Params,
    max_dirs: int,
    stratified: bool = True,
) -> Optional[PrecondDirections]:
    """Compact believed eigenbasis of the optimizer's preconditioner.

    Returns None when the optimizer has no structured parameter-level eigenbasis
    (Muon, SGD, Signum) or when its preconditioner is not initialised yet.

    Must be called *outside* JIT (inspects optimizer state with numpy).
    """
    if max_dirs <= 0:
        return None

    name = optim_name.lower().strip()
    leaves, offsets, _ = _leaf_offsets_of(params)

    if name in {"soap", "ni_soap", "ni-soap", "nisoap"}:
        is_ni = name in {"ni_soap", "ni-soap", "nisoap"}
        return _extract_soap_precond_directions(
            opt_state, is_ni, leaves, offsets, max_dirs, stratified
        )

    if name in {"adam", "adamw"}:
        return _extract_adam_precond_directions(
            opt_state, leaves, offsets, max_dirs, stratified
        )

    return None


def extract_precond_eigsystem(
    opt_state,
    optim_name: str,
    params: Params,
    max_dirs: int,
    stratified: bool = True,
) -> Optional[Tuple[Array, Array]]:
    """Dense ``(Q_hat, lambda_hat)`` form of :func:`extract_precond_directions`.

    ``Q_hat`` is ``(M, dim)`` bfloat16 with unit rows, ``lambda_hat`` is ``(M,)``
    float32; returns None when the optimizer has no structured eigenbasis.

    Memory: M x dim x 2 bytes for ``Q_hat``, plus an M x dim float32 staging
    buffer. That is 20 GB at M=32 over 162M params, so LM-scale callers must use
    :func:`extract_precond_directions` with
    :func:`precond_criterion_from_directions` instead; this dense form is for
    ViT-scale models and the in-JIT :func:`track_eigenstate` path.

    Must be called *outside* JIT.
    """
    dirs = extract_precond_directions(
        opt_state, optim_name, params, max_dirs, stratified
    )
    if dirs is None:
        return None
    _, _, dim = _leaf_offsets_of(params)
    return densify_precond_directions(dirs, dim)


def densify_precond_directions(
    dirs: PrecondDirections, dim: int
) -> Tuple[Array, Array]:
    """Expand compact directions into a dense ``(M, dim)`` bfloat16 basis."""
    M = dirs.num_dirs
    Q_hat = np.zeros((M, dim), dtype=np.float32)
    for d in range(M):
        off = int(dirs.leaf_offset[d])
        if dirs.kind == "soap":
            n_leaf = int(dirs.rows[d]) * int(dirs.cols[d])
            Q_hat[d, off : off + n_leaf] = np.outer(dirs.u[d], dirs.v[d]).ravel()
        else:
            Q_hat[d, off + int(dirs.coord[d])] = 1.0

    return (
        jnp.asarray(Q_hat, dtype=jnp.bfloat16),
        jnp.asarray(dirs.lambda_hat, dtype=jnp.float32),
    )


def precond_criterion_from_directions(
    directions: PrecondDirections,
    params: Params,
    eigenvectors_flat: Array,
    hvp_pytree_fn: Callable[[PyTree], PyTree],
    *,
    damping: float,
    eps: float = 1e-12,
) -> dict:
    """Host-driven Exp-1b criterion metrics from a compact believed eigenbasis.

    Equivalent to :func:`_precond_criterion_metrics` but never forms the dense
    ``(M, dim)`` basis: each direction is materialized as a params-shaped pytree
    (zero except on its own leaf), pushed through ``hvp_pytree_fn`` once, and
    discarded. Costs M curvature matvecs, the same as the in-JIT path.

    Args:
        directions: from :func:`extract_precond_directions`.
        params: parameter pytree defining the leaf structure (unreplicated).
        eigenvectors_flat: ``(K, dim)`` measured curvature eigenvectors.
        hvp_pytree_fn: maps a params-shaped pytree ``q`` to ``C @ q``.
        damping: ridge added to lambda_hat before inverting for phat.

    Returns a dict of host floats / numpy arrays, keyed as the CSV columns.
    """
    leaves, treedef = jax.tree_util.tree_flatten(params)
    zero_leaves = [jnp.zeros(np.shape(l), dtype=jnp.float32) for l in leaves]
    M = directions.num_dirs
    K = int(eigenvectors_flat.shape[0])

    a_per_dir = np.full(M, np.nan, dtype=np.float64)
    overlap = np.zeros((K, M), dtype=np.float64)

    for d in range(M):
        li = int(directions.leaf_idx[d])
        off = int(directions.leaf_offset[d])
        if directions.kind == "soap":
            u = jnp.asarray(directions.u[d], dtype=jnp.float32)
            v = jnp.asarray(directions.v[d], dtype=jnp.float32)
            q_leaf = jnp.outer(u, v)
        else:
            flat = zero_leaves[li].reshape(-1).at[int(directions.coord[d])].set(1.0)
            q_leaf = flat.reshape(zero_leaves[li].shape)

        q = jax.tree_util.tree_unflatten(
            treedef,
            [q_leaf if i == li else z for i, z in enumerate(zero_leaves)],
        )
        hq_leaf = jax.tree_util.tree_leaves(hvp_pytree_fn(q))[li].astype(jnp.float32)
        # q is zero off leaf li, so q^T C q reduces to the one-leaf inner product.
        a_per_dir[d] = float(jnp.vdot(q_leaf, hq_leaf))

        n_leaf = int(q_leaf.size)
        if directions.kind == "adam":
            # Cast on device: eigenvectors_flat is bfloat16, and numpy has no
            # native bfloat16, so converting host-side would go through
            # ml_dtypes rather than a plain float cast.
            col = eigenvectors_flat[:, off + int(directions.coord[d])].astype(
                jnp.float32
            )
            overlap[:, d] = np.asarray(jax.device_get(col), dtype=np.float64)
        else:
            V_blk = eigenvectors_flat[:, off : off + n_leaf].astype(jnp.float32)
            overlap[:, d] = np.asarray(
                jax.device_get(V_blk @ q_leaf.reshape(-1)), dtype=np.float64
            )

    lam = directions.lambda_hat.astype(np.float64)
    inv_lam = 1.0 / (lam + float(damping))
    phat = inv_lam / max(float(np.sum(inv_lam)), eps)

    A_P = float(np.sum(phat * a_per_dir))
    A_Muon = float(np.mean(a_per_dir)) if M else float("nan")
    cov = float(np.mean(phat * a_per_dir) - np.mean(phat) * np.mean(a_per_dir))
    crit_resid = (A_P - A_Muon) - M * cov

    if M and K:
        sing = np.linalg.svd(overlap, compute_uv=False)
        sin2 = float(max(0.0, 1.0 - np.mean(np.square(sing))))
    else:
        sin2 = float("nan")

    return {
        "A_P_believed": A_P,
        "A_Muon_block": A_Muon,
        "crit_resid": float(crit_resid),
        "precond_basis_sin2": sin2,
        # M*Cov(phat, a) / AM is the normalized direction-level certificate (W1);
        # AM is a Section-1 quantity, so only the numerator is formed here.
        "precond_cov_phat_a": cov,
        "a_per_dir": a_per_dir,
        "phat_per_dir": phat,
        "lambda_hat_per_dir": lam,
        "stratum_per_dir": directions.stratum.astype(np.float64),
    }


def soap_steps_since_refresh(
    opt_state,
    is_ni_soap: bool,
    precondition_frequency: int,
) -> float:
    """Steps elapsed since SOAP last recomputed its eigenbasis (B3).

    SOAP refreshes QL/QR when ``step % precondition_frequency == 0``, so the
    staleness of the believed basis at a measurement is ``step %
    precondition_frequency``. Returns NaN when the state is unavailable or the
    preconditioner never refreshes.
    """
    from optim.soap import SoapState, SoapPerParamState
    from optim.ni_soap import NiSoapState, NiSoapPerParamState

    if is_ni_soap:
        StateClass, PerParamClass = NiSoapState, NiSoapPerParamState
    else:
        StateClass, PerParamClass = SoapState, SoapPerParamState

    node = _find_first_of_type(opt_state, StateClass)
    if node is None or int(precondition_frequency) <= 0:
        return float("nan")

    steps = [
        int(jax.device_get(s.step))
        for s in jax.tree_util.tree_leaves(
            node.per_param, is_leaf=lambda x: isinstance(x, PerParamClass)
        )
        if getattr(s, "step", None) is not None
    ]
    steps = [s for s in steps if s >= 0]
    if not steps:
        return float("nan")
    return float(max(steps) % int(precondition_frequency))


def compute_soap_perlayer_sin2(
    opt_state,
    is_ni_soap: bool,
    params: Params,
    eigenvectors: Array,
) -> Tuple[float, list]:
    """Per-layer sin² between SOAP's believed top-k and true top-k subspaces.

    Uses the bilinear form trick: P_i = Q_L^T V_i Q_R avoids forming
    D-dimensional Kronecker vectors.  O(m²n + mn²) per Ritz vector per layer.

    Returns (layer_mean_sin2, list_of_per_layer_sin2) where the list has one
    entry per 2D param leaf (NaN for leaves where SOAP state is unavailable).
    Must be called outside JIT.
    """
    from optim.soap import SoapState, SoapPerParamState
    from optim.ni_soap import NiSoapState, NiSoapPerParamState

    if is_ni_soap:
        StateClass, PerParamClass = NiSoapState, NiSoapPerParamState
    else:
        StateClass, PerParamClass = SoapState, SoapPerParamState

    node = _find_first_of_type(opt_state, StateClass)
    if node is None:
        return float("nan"), []

    param_leaves = jax.tree_util.tree_leaves(params)
    per_param_leaves = jax.tree_util.tree_leaves(
        node.per_param, is_leaf=lambda x: isinstance(x, PerParamClass)
    )
    if len(per_param_leaves) != len(param_leaves):
        return float("nan"), []

    k_ritz = eigenvectors.shape[0]
    if k_ritz == 0:
        return float("nan"), []

    offsets: list = []
    offset = 0
    for p_leaf in param_leaves:
        offsets.append(offset)
        offset += int(np.asarray(jax.device_get(p_leaf)).size)

    per_layer_sin2: list = []
    valid_sin2s: list = []

    for leaf_idx, (p_leaf, s_leaf) in enumerate(zip(param_leaves, per_param_leaves)):
        p_np = np.asarray(jax.device_get(p_leaf))
        if p_np.ndim != 2:
            continue

        m, n = p_np.shape

        QL = np.asarray(jax.device_get(s_leaf.QL), dtype=np.float32)
        QR = np.asarray(jax.device_get(s_leaf.QR), dtype=np.float32)
        if QL.shape[0] <= 1 or QR.shape[0] <= 1 or QL.shape[0] != m or QR.shape[0] != n:
            per_layer_sin2.append(float("nan"))
            continue

        if is_ni_soap:
            eL = np.asarray(jax.device_get(s_leaf.eL), dtype=np.float32)
            eR = np.asarray(jax.device_get(s_leaf.eR), dtype=np.float32)
        else:
            L_mat = np.asarray(jax.device_get(s_leaf.L), dtype=np.float32)
            R_mat = np.asarray(jax.device_get(s_leaf.R), dtype=np.float32)
            eL = np.diag(QL.T @ L_mat @ QL)
            eR = np.diag(QR.T @ R_mat @ QR)

        Lam_ab = np.abs(np.outer(eL, eR))

        k_use = min(k_ritz, m * n)
        flat_lam = Lam_ab.ravel()
        if k_use < len(flat_lam):
            top_flat_idx = np.argpartition(flat_lam, -k_use)[-k_use:]
            top_flat_idx = top_flat_idx[np.argsort(-flat_lam[top_flat_idx])]
        else:
            top_flat_idx = np.argsort(-flat_lam)[:k_use]
        idx_a = top_flat_idx // n
        idx_b = top_flat_idx % n

        off = offsets[leaf_idx]
        v_block = np.asarray(
            jax.device_get(eigenvectors[:, off : off + m * n]),
            dtype=np.float32,
        ).reshape(k_ritz, m, n)

        # P_k = Q_L^T V_k Q_R for each Ritz vector k. Computed as two batched
        # matmuls -> O(k*m*n*(m+n)); the previous np.einsum("ia,kij,jb->kab")
        # was unoptimized (O(k*m^2*n^2)) and hung for LM-sized layers. This is
        # memory-neutral (Section 4 of the revision plan / Q_L,Q_R-vs-GGN alignment).
        T = np.matmul(QL.T[None, :, :], v_block)  # (k, m, n) = Q_L^T V_k
        P = np.matmul(T, QR)                       # (k, m, n) = (Q_L^T V_k) Q_R
        M = P[:, idx_a, idx_b]  # (k_ritz, k_use)

        sv = np.linalg.svd(M, compute_uv=False)
        sv = np.clip(sv, 0.0, 1.0)
        sin2 = float(max(0.0, 1.0 - np.mean(sv ** 2)))
        per_layer_sin2.append(sin2)
        valid_sin2s.append(sin2)

    layer_mean = float(np.mean(valid_sin2s)) if valid_sin2s else float("nan")
    return layer_mean, per_layer_sin2


def compute_muon_topk_energy(
    updates: PyTree,
    params: Params,
    eigenvectors: Array,
) -> Tuple[float, float, list, list]:
    """Fraction of update energy in the true top-k sharp subspace per 2D layer.

    If e_top ≈ k/D the update is incoherent w.r.t. curvature (exposure ≈ AM).
    If e_top < k/D the optimizer actively vacates the sharp subspace (sub-AM).

    Returns (mean_e_top, mean_baseline, per_layer_e_top, per_layer_baseline)
    where the lists have one entry per 2D param leaf (NaN for zero-energy layers).
    Must be called outside JIT.
    """
    param_leaves = jax.tree_util.tree_leaves(params)
    update_leaves = jax.tree_util.tree_leaves(updates)

    k = eigenvectors.shape[0]
    if k == 0:
        return float("nan"), float("nan"), [], []

    offsets: list = []
    offset = 0
    for p_leaf in param_leaves:
        offsets.append(offset)
        offset += int(np.asarray(jax.device_get(p_leaf)).size)

    per_layer_etop: list = []
    per_layer_baseline: list = []
    valid_etop: list = []
    valid_baseline: list = []

    for leaf_idx, (p_leaf, u_leaf) in enumerate(zip(param_leaves, update_leaves)):
        p_np = np.asarray(jax.device_get(p_leaf))
        if p_np.ndim != 2:
            continue

        delta = np.asarray(jax.device_get(u_leaf), dtype=np.float32).ravel()
        d_energy = float(np.sum(delta ** 2))
        D_layer = delta.shape[0]
        baseline = float(k / D_layer)

        if d_energy < 1e-30:
            per_layer_etop.append(float("nan"))
            per_layer_baseline.append(baseline)
            continue

        off = offsets[leaf_idx]
        V_layer = np.asarray(
            jax.device_get(eigenvectors[:, off : off + D_layer]),
            dtype=np.float32,
        )

        e_top = float(np.sum((V_layer @ delta) ** 2)) / d_energy
        per_layer_etop.append(e_top)
        per_layer_baseline.append(baseline)
        valid_etop.append(e_top)
        valid_baseline.append(baseline)

    mean_etop = float(np.mean(valid_etop)) if valid_etop else float("nan")
    mean_baseline = float(np.mean(valid_baseline)) if valid_baseline else float("nan")
    return mean_etop, mean_baseline, per_layer_etop, per_layer_baseline


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
    momentum_flat: Optional[Array] = None,
    precond_eigsys: Optional[Tuple[Array, Array]] = None,
    precond_criterion_damping: float = 1e-6,
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
    # Full-space update–gradient cosine (Eq. 5 first-order alignment factor).  Same leading-sign
    # convention as tracked_update_grad_cosine so a descent step gives a positive value.  Computed on
    # the whole raveled vectors (not the tracked subspace, which holds ~ppm of the update energy).
    full_update_grad_cosine = -jnp.vdot(upd_flat, grad_flat) / (update_norm * grad_norm + eps)
    full_update_grad_cos2 = jnp.square(full_update_grad_cosine)

    rng_key, lanczos_key = jax.random.split(eigen_state.rng_key)
    k = eigen_state.eigenvalues.shape[0]
    extra_k = eigen_state.extra_eigenvalues.shape[0]
    total_keep = k + extra_k
    if total_keep == 0:
        m_norm_val = (
            jnp.sqrt(jnp.sum(jnp.square(momentum_flat.astype(metric_dtype))))
            if momentum_flat is not None else nan
        )
        return eigen_state._replace(
            step=step,
            rng_key=rng_key,
            grad_norm=grad_norm,
            update_norm=update_norm,
            tracked_update_energy_frac=jnp.array(0.0, dtype=metric_dtype),
            tracked_grad_energy_frac=jnp.array(0.0, dtype=metric_dtype),
            tracked_update_grad_cosine=nan,
            full_update_grad_cosine=full_update_grad_cosine,
            full_update_grad_cos2=full_update_grad_cos2,
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
            m_norm=m_norm_val,
            A_P_believed=nan,
            A_Muon_block=nan,
            crit_resid=nan,
            precond_basis_sin2=nan,
        )

    lanczos_steps = max(total_keep, total_keep if num_iter is None else int(num_iter))

    def matvec_flat(v_flat: Array) -> Array:
        v_pytree = unravel_params(v_flat.astype(jnp.float32))
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

    evals, evecs, ritz_residuals, lanczos_alphas, lanczos_betas = lanczos(
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
        return_tridiag=True,
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
            z_pytree = effective_transform_fn(unravel_params(v_flat.astype(jnp.float32)))
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

    # Experiment 1b: preconditioner criterion metrics
    _metric_nan = jnp.asarray(jnp.nan, dtype=metric_dtype)
    _num_precond_dirs = eigen_state.a_per_dir.shape[0]
    if precond_eigsys is not None:
        _Q_hat, _lambda_hat = precond_eigsys
        (
            A_P_believed_val,
            A_Muon_block_val,
            crit_resid_val,
            precond_basis_sin2_val,
            a_per_dir_val,
            phat_per_dir_val,
        ) = _precond_criterion_metrics(
            _Q_hat,
            _lambda_hat,
            all_eigenvectors,
            matvec_flat,
            _num_precond_dirs,
            damping=precond_criterion_damping,
            eps=eps,
        )
    else:
        A_P_believed_val = _metric_nan
        A_Muon_block_val = _metric_nan
        crit_resid_val = _metric_nan
        precond_basis_sin2_val = _metric_nan
        a_per_dir_val = eigen_state.a_per_dir
        phat_per_dir_val = eigen_state.phat_per_dir

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

        # Momentum-reference projections
        if momentum_flat is not None:
            mom_flat = momentum_flat.astype(metric_dtype)
            m_proj_all = _project_rows(all_eigenvectors, mom_flat)
            top_m_proj = m_proj_all[:k]
            extra_m_proj = m_proj_all[k : k + extra_k]
            m_norm_val = jnp.sqrt(jnp.sum(jnp.square(mom_flat)))
        else:
            top_m_proj = jnp.full((k,), jnp.nan, dtype=metric_dtype)
            extra_m_proj = jnp.full((extra_k,), jnp.nan, dtype=metric_dtype)
            m_norm_val = nan
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
        top_m_proj = eigen_state.m_proj
        extra_m_proj = eigen_state.extra_m_proj
        m_norm_val = eigen_state.m_norm

    return eigen_state._replace(
        step=step,
        lanczos_alpha=lanczos_alphas,
        lanczos_beta=lanczos_betas,
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
        full_update_grad_cosine=full_update_grad_cosine,
        full_update_grad_cos2=full_update_grad_cos2,
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
        m_proj=top_m_proj,
        extra_m_proj=extra_m_proj,
        m_norm=m_norm_val,
        A_P_believed=A_P_believed_val,
        A_Muon_block=A_Muon_block_val,
        crit_resid=crit_resid_val,
        precond_basis_sin2=precond_basis_sin2_val,
        a_per_dir=a_per_dir_val,
        phat_per_dir=phat_per_dir_val,
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
    return_tridiag: bool = False,
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
        # Read as fp32 for numerically stable orthogonalization; v_basis stored as bf16.
        v = v_basis[i].astype(jnp.float32)

        w = matvec(v)
        alpha = jnp.vdot(v, w)
        w = w - alpha * v

        def ortho_against_prev(current_w, basis_idx):
            prev_v = v_basis[basis_idx].astype(jnp.float32)
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

        # Store back as bf16 to halve the 8 GiB Krylov basis memory.
        v_basis = v_basis.at[i + 1].set(next_v.astype(jnp.bfloat16))
        alphas = alphas.at[i].set(alpha)
        betas = betas.at[i].set(beta)
        return (v_basis, alphas, betas), None

    # bf16 storage: (num_iter+1, dim) fp32 would be ~8 GiB for 162M-param models.
    v_basis = jnp.zeros((num_iter + 1, dim), dtype=jnp.bfloat16)
    v_basis = v_basis.at[0].set(v0.astype(jnp.bfloat16))
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
    # _expand_from_basis promotes bf16 basis × fp32 coeffs → fp32; cast back to bf16.
    eigenvectors = _expand_from_basis(evecs_t.T, v_k).reshape(evecs_t.shape[1], dim).astype(jnp.bfloat16)
    if return_residuals and return_tridiag:
        return evals, eigenvectors, residuals, alphas, betas
    if return_residuals:
        return evals, eigenvectors, residuals
    if return_tridiag:
        return evals, eigenvectors, alphas, betas
    return evals, eigenvectors


# ===========================================================================
# Revision Section 1 — probe measures (additive; consumes the eigenvectors
# already produced by track_eigenstate, does not modify it).
#
#   1a  participation ratio per tracked eigenvector          (PR, PR/d)
#   1b  Hutchinson curvature trace -> arithmetic-mean ref     (Tr(C), AM, AM/lammax)
#   1c  per-block covariance decomposition                    (w_b, A_b, AM_b)
#   1d  layer-type / governance attribution + restricted A    (A_restricted, AM_S)
#
# All heavy reductions are done on-device; only small per-block/per-type arrays
# are returned to the host for CSV logging. The block partition is a *contiguous*
# partition of the raveled parameter vector (ravel_pytree leaf order), which is
# exactly the order of jax.tree_util.tree_leaves_with_path used below.
# ===========================================================================

# Ordered layer taxonomy used for the by-type attribution of Section 1d.
LAYER_TYPES: Tuple[str, ...] = (
    "embed",
    "head_unembed",
    "attn_qkv",
    "attn_out",
    "mlp_up",
    "mlp_down",
    "norm",
    "bias",
    "other",
)


def classify_layer_type(name: str, ndim: int) -> str:
    """Map a Flax parameter path to one of :data:`LAYER_TYPES`."""
    n = name.lower()
    if "lm_head" in n:
        return "head_unembed"
    if ("embed_tokens" in n) or ("embedding" in n):
        return "embed"
    if "w_qkv" in n:
        return "attn_qkv"
    if "w_out" in n:
        return "attn_out"
    if ("fc_gate" in n) or ("fc_up" in n) or ("fc1" in n):
        return "mlp_up"
    if "fc2" in n:
        return "mlp_down"
    if "norm" in n:
        return "norm"
    if ndim <= 1:
        return "bias"
    return "other"


def build_block_metadata(params: Params) -> dict:
    """Contiguous per-leaf block partition of the raveled parameter vector.

    Returns a dict of host-side lists/arrays. ``namesake`` marks the routed 2-D
    matrices (Muon/SOAP/Kaon/Hadamard-Signum apply their namesake rule there and
    fall back to AdamW elsewhere); it is optimizer-independent here so the same
    restricted subspace is applied to every arm (the decisive control of 1d).
    """
    from .matrix_routing import should_use_matrix_preconditioner, path_to_name

    leaves_with_path = jax.tree_util.tree_leaves_with_path(params)
    names, types, is_matrix, namesake = [], [], [], []
    starts, sizes, rows, cols = [], [], [], []
    polar_r, gamma2 = [], []
    off = 0
    for path, leaf in leaves_with_path:
        name = path_to_name(path)
        numel = int(np.prod(leaf.shape))
        ndim = int(getattr(leaf, "ndim", np.ndim(leaf)))
        ism = ndim == 2
        names.append(name)
        types.append(classify_layer_type(name, ndim))
        is_matrix.append(ism)
        namesake.append(bool(should_use_matrix_preconditioner(path, leaf)))
        starts.append(off)
        sizes.append(numel)
        rows.append(int(leaf.shape[0]) if ism else numel)
        cols.append(int(leaf.shape[1]) if ism else 1)
        # A8: polar rank r_b = min(fan_in, fan_out) and the width-transfer
        # multiplier gamma_b^2 = max(1, fan_out/fan_in). Verified against the
        # installed optax 0.2.8 scale_by_muon, which applies
        # sqrt(max(1, fan_out/fan_in)) -- identical to optim/kaon.py
        # _shape_factor. Undefined for non-matrix leaves.
        if ism:
            fi, fo = int(leaf.shape[0]), int(leaf.shape[1])
            polar_r.append(min(fi, fo))
            gamma2.append(max(1.0, fo / fi))
        else:
            polar_r.append(0)
            gamma2.append(float("nan"))
        off += numel

    n_blocks = len(names)
    # A0: the blocks must tile the flattened optimization variable exactly --
    # no gaps, no overlaps, no leaf in two blocks. Checked on the integers here
    # so a mis-tiling fails at setup rather than as a silent mass leak later.
    if int(np.sum(sizes)) != int(off):
        raise ValueError(
            f"block partition does not tile the parameter vector: "
            f"sum_b d_b = {int(np.sum(sizes))} but d = {int(off)}"
        )
    for b in range(1, n_blocks):
        if starts[b] != starts[b - 1] + sizes[b - 1]:
            raise ValueError(
                f"block partition is not contiguous at block {b} ({names[b]}): "
                f"start {starts[b]} != previous start {starts[b - 1]} + size "
                f"{sizes[b - 1]}"
            )
    type_ids = np.array([LAYER_TYPES.index(t) for t in types], dtype=np.int32)
    # Flat segment map: coordinate -> block id (contiguous). Used both for
    # segment reductions and for the per-block matvec masks (seg == b).
    seg = np.repeat(np.arange(n_blocks, dtype=np.int32), np.array(sizes, dtype=np.int64))
    return {
        "names": names,
        "types": types,
        "is_matrix": is_matrix,
        "namesake": namesake,
        "type_ids": type_ids,
        "starts": np.array(starts, dtype=np.int64),
        "sizes": np.array(sizes, dtype=np.int32),
        "rows": np.array(rows, dtype=np.int32),
        "cols": np.array(cols, dtype=np.int32),
        "polar_r": np.array(polar_r, dtype=np.int32),
        "gamma2": np.array(gamma2, dtype=np.float64),
        "seg": seg,
        # Static (start, size) pairs for the exact per-block tree reduction.
        "slices": tuple((int(a), int(b)) for a, b in zip(starts, sizes)),
        "dim": off,
        "n_blocks": n_blocks,
        "n_types": len(LAYER_TYPES),
    }


def _spectral_norm_power(mat: Array, key: Array, iters: int = 10) -> Array:
    """Largest singular value of ``mat`` by ``iters`` power iterations on M^T M."""
    r, c = mat.shape
    v = jax.random.normal(key, (c,), dtype=jnp.float32)
    v = v / (jnp.linalg.norm(v) + 1e-12)

    def _body(v, _):
        u = mat @ v
        u = u / (jnp.linalg.norm(u) + 1e-12)
        w = mat.T @ u
        sigma = jnp.linalg.norm(w)
        return w / (sigma + 1e-12), sigma

    v, sigmas = jax.lax.scan(_body, v, None, length=iters)
    return sigmas[-1]


def _randomized_svdvals(mat: Array, k: int, key: Array, iters: int = 2) -> Array:
    """Top-``k`` singular values via randomized subspace iteration (descending)."""
    r, c = mat.shape
    k = int(min(k, r, c))
    omega = jax.random.normal(key, (c, k), dtype=jnp.float32)
    Y = mat @ omega
    Q, _ = jnp.linalg.qr(Y)
    for _ in range(iters):
        Q, _ = jnp.linalg.qr(mat.T @ Q)
        Q, _ = jnp.linalg.qr(mat @ Q)
    return jnp.linalg.svd(Q.T @ mat, compute_uv=False)


def namesake_matrix_norms(
    grads: PyTree,
    updates: PyTree,
    block_meta: dict,
    *,
    seed: int = 0,
    power_iters: int = 10,
    svd_full_max: int = 4096,
    svd_topk: int = 256,
) -> dict:
    """A4: matrix norms on the namesake (matrix-routed) blocks, host-driven.

    For every block flagged ``namesake`` in ``block_meta`` returns ``||G_b||_F``,
    ``||G_b||_nuc``, ``||Delta_b||_sigma`` (``power_iters`` power iterations) and
    ``r_b = min(rows, cols)``. The nuclear norm uses a full ``svdvals`` while
    ``min(rows, cols) <= svd_full_max`` and a randomized top-``svd_topk``
    otherwise; ``nuc_k`` records how many singular values were actually summed so
    a truncated nuclear norm is never mistaken for the exact one.

    Called outside JIT — shapes differ per leaf, and each leaf's ops compile once
    and are then cached by shape.
    """
    namesake = np.asarray(block_meta["namesake"], dtype=bool)
    g_leaves = jax.tree_util.tree_leaves(grads)
    u_leaves = jax.tree_util.tree_leaves(updates)
    n_blocks = int(block_meta["n_blocks"])
    if len(g_leaves) != n_blocks or len(u_leaves) != n_blocks:
        raise ValueError(
            f"namesake_matrix_norms: leaf count {len(g_leaves)}/{len(u_leaves)} "
            f"does not match block count {n_blocks}; block metadata is stale."
        )

    out = {
        "block_g_fro": np.full(n_blocks, np.nan, dtype=np.float64),
        "block_g_nuc": np.full(n_blocks, np.nan, dtype=np.float64),
        "block_g_nuc_k": np.full(n_blocks, np.nan, dtype=np.float64),
        "block_d_sigma": np.full(n_blocks, np.nan, dtype=np.float64),
        "block_r": np.full(n_blocks, np.nan, dtype=np.float64),
    }

    key = jax.random.PRNGKey(int(seed))
    for b in range(n_blocks):
        if not namesake[b]:
            continue
        G = jnp.asarray(g_leaves[b], dtype=jnp.float32)
        D = jnp.asarray(u_leaves[b], dtype=jnp.float32)
        if G.ndim != 2:
            continue
        key, k_svd, k_pow = jax.random.split(key, 3)
        r_b = int(min(G.shape))
        if r_b <= int(svd_full_max):
            sv = jnp.linalg.svd(G, compute_uv=False)
        else:
            sv = _randomized_svdvals(G, int(svd_topk), k_svd)
        out["block_g_fro"][b] = float(jnp.linalg.norm(G))
        out["block_g_nuc"][b] = float(jnp.sum(sv))
        out["block_g_nuc_k"][b] = float(sv.shape[0])
        out["block_d_sigma"][b] = float(
            _spectral_norm_power(D, k_pow, iters=int(power_iters))
        )
        out["block_r"][b] = float(r_b)
    return out


def _contiguous_block_sums(x: Array, slices) -> Array:
    """Per-block sums over a *contiguous* partition, as exact tree reductions.

    ``jax.ops.segment_sum`` lowers to a scatter-add that accumulates every
    coordinate of a block into a single float32 accumulator. Once the running
    total exceeds 2**24 times the typical term, further adds round away and the
    block total silently stagnates. With d_b = 50257*768 = 38.6M that costs the
    embedding and unembedding roughly half their mass (measured: a factor
    0.5343 on both, identical because the deficit depends only on d_b), while
    every block below 2**24 = 16.8M coordinates is unaffected. The full-space
    denominator uses ``jnp.sum``, which is a tree reduction and stays exact, so
    the damage shows up as ``sum_b block_w_b`` falling to 0.785-0.984 instead of
    1. Slicing and reducing per block keeps the tree reduction and is exact.

    ``slices`` is a host-side sequence of ``(start, size)`` int pairs; it must be
    static so the slice shapes are known at trace time. Reduces the last axis.
    """
    outs = [
        jnp.sum(jax.lax.dynamic_slice_in_dim(x, int(st), int(sz), axis=-1), axis=-1)
        for st, sz in slices
    ]
    return jnp.stack(outs, axis=-1)


def native_frame_decomposition(
    upd_flat: Array,
    matvec_flat: Callable[[Array], Array],
    block_meta: dict,
    *,
    eta_blocks=None,
    rng_key: Array,
    n_draws: int = 8,
    rank_tol: float = 1e-6,
    eps: float = 1e-12,
) -> dict:
    """A6/A8-exact: realized native-frame coefficients on the namesake blocks.

    The frame is ``q_i = vec(u_i v_i^T)`` from the SVD of ``Delta_b`` itself.
    Newton-Schulz in the form ``X <- aX + bX(X^TX) + cX(X^TX)^2`` is a polynomial
    in ``X^T X``, so it preserves U and V exactly and acts only on the singular
    values; taking the SVD of the realized update therefore recovers the same
    frame as the momentum without touching optimizer state. The realized
    coefficients are the singular values, so ``p_i = sigma_i^2 / sum_j sigma_j^2``.

    ``p_i = 1/r`` holds only for the *exact* polar factor. optax 0.2.8 uses 5
    Newton-Schulz steps with the Jordan coefficients, which deliberately do not
    converge to 1, so p_i is not uniform and the allocation term does not vanish.
    That departure is measured here rather than assumed away.

    Two probe distributions are drawn over the pooled namesake support:
      equal-magnitude  eps/sqrt(r)  ->  E[A] = lambda_bar_Q   (frame mean)
      weighted         eps * |z|    ->  E[A] = A_diag^Q       (allocation)
    Both randomize every namesake block within a single draw, so cross-block
    terms vanish only *in expectation*: the returned SEs are across draws and
    must be reported rather than treated as exact.

    Three-way split (telescopes exactly):
        A_actual - lambda_bar_b = (lambda_bar_Q - lambda_bar_b)   subspace
                                + (A_diag_Q    - lambda_bar_Q)    allocation
                                + (A_actual    - A_diag_Q)        coupling
    """
    f32 = jnp.float32
    upd_flat = upd_flat.astype(f32)
    slices = block_meta["slices"]
    namesake = np.asarray(block_meta["namesake"], dtype=bool)
    rows = np.asarray(block_meta["rows"], dtype=np.int64)
    cols = np.asarray(block_meta["cols"], dtype=np.int64)
    gamma2 = np.asarray(block_meta["gamma2"], dtype=np.float64)
    n_blocks = int(block_meta["n_blocks"])
    idx = [b for b in range(n_blocks) if namesake[b]]

    nan = jnp.asarray(jnp.nan, f32)
    per_block = {k: [nan] * n_blocks for k in
                 ("numerical_rank", "kappa_pred", "eps_NS", "p_spread", "p_max_r")}
    svd_cache = {}
    for b in idx:
        st, sz = slices[b]
        mat = jax.lax.dynamic_slice(upd_flat, (st,), (sz,)).reshape(
            int(rows[b]), int(cols[b])
        )
        U, sig, Vt = jnp.linalg.svd(mat, full_matrices=False)
        sig = sig.astype(f32)
        svd_cache[b] = (U, sig, Vt)
        r_b = jnp.asarray(float(min(int(rows[b]), int(cols[b]))), f32)
        smax = jnp.max(sig)
        # r_b = min(m,n) presumes full rank; a rank-deficient block gives
        # kappa_b < 1 for a reason unrelated to Newton-Schulz, and only this
        # column disambiguates the two.
        per_block["numerical_rank"][b] = jnp.sum(
            (sig > rank_tol * smax).astype(f32)
        )
        p = jnp.square(sig) / (jnp.sum(jnp.square(sig)) + eps)
        per_block["p_spread"][b] = jnp.max(p) / (jnp.min(p) + eps)
        per_block["p_max_r"][b] = jnp.max(p) * r_b
        if eta_blocks is not None and np.isfinite(gamma2[b]):
            scale = jnp.asarray(eta_blocks[b], f32) * jnp.sqrt(
                jnp.asarray(float(gamma2[b]), f32)
            )
            sn = sig / (scale + eps)
            per_block["kappa_pred"][b] = jnp.sum(jnp.square(sn)) / r_b
            # what "finite Newton-Schulz error" actually means: rms(sigma_hat-1).
            # kappa == 1 is necessary but not sufficient for phi == 1, so this is
            # the honest quantity and kappa alone is not.
            per_block["eps_NS"][b] = jnp.sqrt(
                jnp.sum(jnp.square(sn - 1.0)) / r_b
            )

    def _draw(key, weighted: bool):
        v = jnp.zeros_like(upd_flat)
        keys = jax.random.split(key, max(len(idx), 1))
        for j, b in enumerate(idx):
            U, sig, Vt = svd_cache[b]
            r_b = sig.shape[0]
            sgn = jax.random.rademacher(keys[j], (r_b,), dtype=f32)
            coef = sgn * sig if weighted else sgn / jnp.sqrt(jnp.asarray(r_b, f32))
            blk = (U * coef[None, :]) @ Vt
            st, _sz = slices[b]
            v = jax.lax.dynamic_update_slice(v, blk.reshape(-1), (st,))
        return v

    def _rayleigh(v):
        Cv = matvec_flat(v).astype(f32)
        den = jnp.vdot(v, v)
        return jnp.where(den > eps, jnp.vdot(v, Cv) / (den + eps), jnp.nan)

    out = {f"block_{k}": jnp.stack(v) for k, v in per_block.items()}
    k_lb, k_ad = jax.random.split(rng_key)
    for tag, weighted, key in (
        ("lambda_bar_Q", False, k_lb), ("A_diag_Q", True, k_ad)
    ):
        draws = jnp.stack(
            [_rayleigh(_draw(k, weighted)) for k in jax.random.split(key, n_draws)]
        )
        out[tag] = jnp.mean(draws)
        out[tag + "_se"] = jnp.std(draws) / jnp.sqrt(jnp.asarray(n_draws, f32))

    # realized pooled namesake update on the same support
    nm_coord = jnp.asarray(namesake)[jnp.asarray(block_meta["seg"])]
    out["A_actual_Q"] = _rayleigh(jnp.where(nm_coord, upd_flat, 0.0))
    # frame mean of the block itself: (1/d_b) tr(C_b) pooled over namesake
    return out


def section1_measures(
    updates: PyTree,
    eigenvectors: Array,
    extra_eigenvectors: Array,
    eigenvalues: Array,
    matvec_flat: Callable[[Array], Array],
    *,
    seg: Array,
    block_namesake: Array,
    type_ids: Array,
    sizes: Array,
    n_blocks: int,
    n_types: int,
    m_hutchinson: int,
    rng_key: Array,
    grads: Optional[PyTree] = None,
    compute_block_A: bool = True,
    block_slices=None,
    eta_blocks: Optional[Array] = None,
    block_gamma2: Optional[Array] = None,
    block_polar_r: Optional[Array] = None,
    n_adiag_probes: int = 8,
    eps: float = 1e-12,
) -> dict:
    """Compute the Section-1 probe measures for one checkpoint.

    Args mirror :func:`track_eigenstate`: ``matvec_flat`` maps a flat vector to
    ``C @ v`` (same curvature operator used by the probe). ``seg`` is the flat
    coordinate->block map from :func:`build_block_metadata`. Returns a dict of
    JAX arrays (small except ``d_hat`` is reduced away internally).

    ``grads`` (same PyTree structure as ``updates``) enables the per-block
    gradient channel (A1/A2): ``block_gnorm2``, the signed per-block descent
    cosine ``block_cos`` and the raw inner product ``block_grad_update``. Pass
    the *same* (clipped) gradients handed to :func:`track_eigenstate` so
    ``sum_b block_gnorm2 == grad_norm**2``.

    The Section-5 block ledger is ``sizes`` (d_b, static, in
    ``section1_block_tags.csv``), ``block_energy`` (||Delta_b||^2),
    ``block_gnorm2`` (||g_b||^2), ``block_grad_update`` (-<g_b, Delta_b>),
    ``block_cos`` (c_b) and ``block_A`` (A_b). ``block_cos`` is signed on
    purpose: a globally-descent update does not guarantee every block is locally
    descent-aligned, and the nonnegative-scale optimum is
    ``t_b* = max(beta_b, 0) / A_b``, which needs the sign to be recoverable.
    ``id_*_rel`` are the three closure identities against the full-space
    quantities and are meant to be hard acceptance checks on the rerun.
    """
    upd_flat, _ = ravel_pytree(updates)
    upd_flat = upd_flat.astype(jnp.float32)
    dim = upd_flat.shape[0]
    # Raw ||Delta||^2 is reported as-is; the +eps form is only ever a denominator,
    # so the logged identity sum_b block_energy_b == update_energy_full is exact.
    update_energy_raw = jnp.sum(jnp.square(upd_flat))
    update_energy = update_energy_raw + eps
    f32 = jnp.float32

    # Every reduction over the coordinate axis goes through this. With
    # ``block_slices`` it is an exact per-block tree reduction; without it we
    # fall back to segment_sum, which loses mass on blocks above 2**24
    # coordinates (see _contiguous_block_sums).
    if block_slices is None:
        def _seg_sum(x):
            return jax.ops.segment_sum(x, seg, num_segments=n_blocks)
    else:
        def _seg_sum(x):
            return _contiguous_block_sums(x, block_slices)

    # ---- 1a: participation ratio of each tracked eigenvector ----
    V = jnp.concatenate([eigenvectors, extra_eigenvectors], axis=0).astype(f32)
    v2 = jnp.square(V)                                  # (total_keep, dim)
    row_norm2 = jnp.sum(v2, axis=1) + eps               # ~1 for unit vectors
    sum4 = jnp.sum(jnp.square(v2), axis=1) + eps
    pr = jnp.square(row_norm2) / sum4
    pr_norm = pr / dim

    # eigenvector mass by block then by layer type (fraction of ||v_i||^2)
    def _block_mass(row_v2):
        return _seg_sum(row_v2)

    block_mass = jax.vmap(_block_mass)(v2)              # (total_keep, n_blocks)

    def _type_from_block(bvals):
        return jax.ops.segment_sum(bvals, type_ids, num_segments=n_types)

    evec_mass_type = jax.vmap(_type_from_block)(block_mass)   # (total_keep, n_types)
    evec_mass_type = evec_mass_type / row_norm2[:, None]

    # ---- 1b: Hutchinson trace + diagonal estimator (shared with 1c/1d) ----
    hutch_key, adiag_key = jax.random.split(rng_key)
    keys = jax.random.split(hutch_key, m_hutchinson)

    # p_i = Delta_i^2 / ||Delta||^2, the coordinate-frame allocation weights (A5)
    p_coord = jnp.square(upd_flat) / update_energy

    def _hutch_body(d_sum, key):
        z = jax.random.rademacher(key, (dim,), dtype=f32)
        Cz = matvec_flat(z).astype(f32)
        # per-probe: (z^T C z, ||C z||^2). E[z z^T] = I gives E[||Cz||^2] =
        # tr(C^T C) = tr(C^2) for symmetric C — the second trace moment (A3).
        # d_hat^(s) = z * Cz is the per-probe unbiased diag(C) estimate; A5 needs
        # A_diag^(s) and lambda_bar^(s) formed from the *same* probe so their
        # shared Hutchinson noise cancels in the difference.
        dh = z * Cz
        return d_sum + dh, (
            jnp.vdot(z, Cz),
            jnp.sum(jnp.square(Cz)),
            jnp.vdot(p_coord, dh),          # A_diag^(s)   = sum_i p_i d_hat^(s)_i
            jnp.sum(dh) / jnp.asarray(dim, f32),   # lambda_bar^(s)
        )

    d_sum, (per_probe, per_probe_c2, per_probe_adiag, per_probe_lbar) = jax.lax.scan(
        _hutch_body, jnp.zeros((dim,), f32), keys
    )
    d_hat = d_sum / jnp.asarray(m_hutchinson, f32)      # unbiased diag(C) estimate
    tr_C = jnp.mean(per_probe)
    tr_C_se = jnp.std(per_probe) / jnp.sqrt(jnp.asarray(m_hutchinson, f32))
    tr_C2 = jnp.mean(per_probe_c2)
    tr_C2_se = jnp.std(per_probe_c2) / jnp.sqrt(jnp.asarray(m_hutchinson, f32))
    AM = tr_C / dim
    # Contraharmonic mean of the spectrum, tr(C^2)/tr(C) — the GD rung of the
    # mean ladder, in the full space. CH >= AM for any PSD C (Cauchy-Schwarz).
    CH = tr_C2 / (tr_C + eps)
    lambda_max = eigenvalues[0].astype(f32)
    AM_over_lammax = AM / (lambda_max + eps)
    CH_over_lammax = CH / (lambda_max + eps)

    # ---- full exposure A = Δ^T C Δ / ||Δ||^2 ----
    CDelta = matvec_flat(upd_flat).astype(f32)
    A_full = jnp.vdot(upd_flat, CDelta) / update_energy
    A_over_lammax = A_full / (lambda_max + eps)

    # ---- 1c: per-block energy share, block trace (AM_b), block Rayleigh A_b ----
    block_energy = _seg_sum(jnp.square(upd_flat))
    block_w = block_energy / update_energy
    block_diag_C = _seg_sum(d_hat)                                         # Tr(C_b)
    block_AM = block_diag_C / jnp.maximum(sizes.astype(f32), 1.0)          # Tr(C_b)/r_b

    # ---- A1/A2: per-block gradient energy and per-block descent cosine ----
    # Sign convention matches full_update_grad_cosine: positive on descent, so
    # the -<g, Delta> numerator. beta_b^2 = block_gnorm2_b * block_cos_b^2 is the
    # measured per-block first-order gain coefficient.
    if grads is not None:
        g_flat, _ = ravel_pytree(grads)
        g_flat = g_flat.astype(f32)
        block_gnorm2 = _seg_sum(jnp.square(g_flat))
        block_gd = _seg_sum(g_flat * upd_flat)
        # s_b := -<g_b, Delta_b>, the raw per-block first-order gain numerator.
        # Logged directly (algebraically redundant) so the identity
        # s_b == sqrt(block_gnorm2_b * block_energy_b) * block_cos_b is an
        # independent runtime check on the block segmentation itself.
        block_grad_update = -block_gd
        block_cos = -block_gd / jnp.sqrt(block_gnorm2 * block_energy + eps)
        # Guard the empty/degenerate-block case rather than reporting a hard 0.
        block_cos = jnp.where(
            (block_gnorm2 > eps) & (block_energy > eps), block_cos, jnp.nan
        )
        grad_norm2_full = jnp.sum(jnp.square(g_flat))
        grad_update_full = -jnp.vdot(g_flat, upd_flat)
    else:
        block_gnorm2 = jnp.full((n_blocks,), jnp.nan, dtype=f32)
        block_cos = jnp.full((n_blocks,), jnp.nan, dtype=f32)
        block_grad_update = jnp.full((n_blocks,), jnp.nan, dtype=f32)
        grad_norm2_full = jnp.asarray(jnp.nan, dtype=f32)
        grad_update_full = jnp.asarray(jnp.nan, dtype=f32)

    # ---- block-ledger closure identities (hard acceptance checks) ----
    # sum_b ||g_b||^2 == ||g||^2, sum_b ||Delta_b||^2 == ||Delta||^2 and
    # sum_b <g_b, Delta_b> == <g, Delta> must hold to numerical precision. A
    # partition that drops or double-counts coordinates -- the embedding/head
    # segmentation failure mode -- breaks these and nothing else, so they are
    # logged per row rather than assumed.
    def _rel_gap(part_sum, full):
        return jnp.abs(part_sum - full) / (jnp.abs(full) + eps)

    id_gnorm2_rel = _rel_gap(jnp.sum(block_gnorm2), grad_norm2_full)
    id_update_energy_rel = _rel_gap(jnp.sum(block_energy), update_energy_raw)
    id_grad_update_rel = _rel_gap(jnp.sum(block_grad_update), grad_update_full)

    if compute_block_A:
        def _a_body(_carry, b):
            mask = seg == b
            e = jnp.where(mask, upd_flat, 0.0)
            Ce = matvec_flat(e).astype(f32)
            den = jnp.vdot(e, e)
            a_b = jnp.where(den > eps, jnp.vdot(e, Ce) / (den + eps), jnp.nan)
            return _carry, a_b

        _, block_A = jax.lax.scan(_a_body, None, jnp.arange(n_blocks, dtype=jnp.int32))
    else:
        block_A = jnp.full((n_blocks,), jnp.nan, dtype=f32)

    # ---- A7/A8: per-block step-size coefficients (norms only, no products) ----
    # A7  s_b   = ||Delta_b|| / (eta_b sqrt(d_b))    so ||Delta_b||^2 = eta_b^2 d_b s_b^2
    # A8  kappa_b = ||Delta_b||^2 / (eta_b^2 gamma_b^2 r_b)
    # kappa_b is NOT predicted to be 1: the 5-step Newton-Schulz map undershoots,
    # giving ~0.72-0.84 depending on the momentum spectrum. The testable content
    # is whether kappa_b is *constant across blocks*, so it is logged raw and the
    # shape test is done in analysis. Kaon normalises its random singular values
    # to unit RMS and so sits at kappa = 1 by construction; that offset is a
    # confound for any Muon-vs-Kaon step-magnitude comparison.
    block_norm = jnp.sqrt(jnp.maximum(block_energy, 0.0))
    if eta_blocks is not None:
        eta_b = jnp.asarray(eta_blocks, f32)
        d_b = sizes.astype(f32)
        block_s_rms = block_norm / (eta_b * jnp.sqrt(jnp.maximum(d_b, 1.0)) + eps)
        if block_gamma2 is not None and block_polar_r is not None:
            g2 = jnp.asarray(block_gamma2, f32)
            rb = jnp.asarray(block_polar_r, f32)
            kappa_b = block_energy / (jnp.square(eta_b) * g2 * rb + eps)
            kappa_b = jnp.where(rb > 0, kappa_b, jnp.nan)
        else:
            kappa_b = jnp.full((n_blocks,), jnp.nan, dtype=f32)
    else:
        block_s_rms = jnp.full((n_blocks,), jnp.nan, dtype=f32)
        kappa_b = jnp.full((n_blocks,), jnp.nan, dtype=f32)

    # block-diagonal exposure and its residual against the full operator
    sum_wb_Ab = jnp.nansum(block_w * block_A)
    sum_wb_AMb = jnp.sum(block_w * block_AM)
    cross_block_resid = A_full - sum_wb_Ab

    # ---- A5: coordinate-frame allocation / coupling split ----
    # A_diag_coord = sum_i p_i C_ii, and since sum_i p_i = 1,
    #   A_diag_coord - lambda_bar = d * Cov_i(p_i, C_ii).
    # Paired per probe: the difference is taken inside the scan output, so the
    # SE is the SE *of the difference*, which is far tighter than differencing
    # two separately-averaged terms. Gamma_coord is the residual coupling.
    m_f32 = jnp.asarray(m_hutchinson, f32)
    A_diag_coord = jnp.mean(per_probe_adiag)
    cov_term_probe = per_probe_adiag - per_probe_lbar
    cov_term = jnp.mean(cov_term_probe)
    cov_term_se = jnp.std(cov_term_probe) / jnp.sqrt(m_f32)
    Gamma_coord = A_full - A_diag_coord
    # Signum sanity: an exact sign step makes |Delta_i| constant, so p_i = 1/nnz
    # and A_diag_coord == lambda_bar identically -- a pipeline assertion, not a
    # curvature result. jnp.sign(0) == 0, so report the support fraction.
    nnz_frac = jnp.mean((upd_flat != 0).astype(f32))

    # Independent cross-check through C rather than through the Hutchinson
    # diagonal: E_eps[A(eps * |Delta|)] = A_diag_coord exactly, because
    # ||eps * |Delta|| || = ||Delta|| deterministically and E[eps_i eps_j] = 0.
    if n_adiag_probes > 0:
        def _adiag_body(_carry, key):
            sgn = jax.random.rademacher(key, (dim,), dtype=f32)
            dr = sgn * jnp.abs(upd_flat)
            Cdr = matvec_flat(dr).astype(f32)
            return _carry, jnp.vdot(dr, Cdr) / update_energy

        _, adiag_draws = jax.lax.scan(
            _adiag_body, None, jax.random.split(adiag_key, n_adiag_probes)
        )
        A_diag_coord_check = jnp.mean(adiag_draws)
        A_diag_coord_check_se = jnp.std(adiag_draws) / jnp.sqrt(
            jnp.asarray(n_adiag_probes, f32)
        )
    else:
        A_diag_coord_check = jnp.asarray(jnp.nan, f32)
        A_diag_coord_check_se = jnp.asarray(jnp.nan, f32)

    # ---- 1d: layer-type / governance attribution + restricted exposure ----
    type_energy = jax.ops.segment_sum(block_energy, type_ids, num_segments=n_types)
    type_energy_frac = type_energy / update_energy
    namesake_energy = jnp.sum(jnp.where(block_namesake, block_energy, 0.0))
    fallback_energy = update_energy - eps - namesake_energy
    namesake_energy_frac = namesake_energy / update_energy

    nm_coord = block_namesake[seg]                     # (dim,) bool over coordinates
    e_S = jnp.where(nm_coord, upd_flat, 0.0)
    CeS = matvec_flat(e_S).astype(f32)
    den_S = jnp.vdot(e_S, e_S)
    A_restricted = jnp.where(den_S > eps, jnp.vdot(e_S, CeS) / (den_S + eps), jnp.nan)
    r_S = jnp.sum(jnp.where(block_namesake, sizes.astype(f32), 0.0))
    tr_C_S = jnp.sum(jnp.where(nm_coord, d_hat, 0.0))
    AM_S = tr_C_S / jnp.maximum(r_S, 1.0)
    A_restricted_over_lammax = A_restricted / (lambda_max + eps)
    AM_S_over_lammax = AM_S / (lambda_max + eps)

    # ---- A9: two-group (namesake M vs fallback F) decomposition ----
    # Exact given disjoint support, which is why the A0 partition checks must
    # pass first. Gamma_MF = 2 Delta_M^T C Delta_F / ||Delta||^2 is then the
    # measured cross term, replacing the triangle-inequality bound on A_F.
    e_F = jnp.where(nm_coord, 0.0, upd_flat)
    CeF = matvec_flat(e_F).astype(f32)
    den_F = jnp.vdot(e_F, e_F)
    A_fallback = jnp.where(den_F > eps, jnp.vdot(e_F, CeF) / (den_F + eps), jnp.nan)
    w_M = namesake_energy_frac
    w_F = den_F / update_energy
    Gamma_MF = A_full - w_M * A_restricted - w_F * A_fallback

    return {
        # 1a
        "pr": pr,
        "pr_norm": pr_norm,
        "evec_mass_type": evec_mass_type,           # (total_keep, n_types)
        # 1b
        "hutch_m": jnp.asarray(m_hutchinson, jnp.int32),
        "tr_C": tr_C,
        "tr_C_se": tr_C_se,
        "AM": AM,
        "lambda_max": lambda_max,
        "AM_over_lammax": AM_over_lammax,
        "A_full": A_full,
        "A_over_lammax": A_over_lammax,
        # A3 — second trace moment and the contraharmonic (GD) rung
        "tr_C2": tr_C2,
        "tr_C2_se": tr_C2_se,
        "CH": CH,
        "CH_over_lammax": CH_over_lammax,
        # 1c
        "block_w": block_w,
        "block_A": block_A,
        "block_AM": block_AM,
        # A1/A2 — per-block gradient channel. block_energy and block_grad_update
        # are the un-normalised ledger entries: with block_gnorm2 and block_A they
        # reconstruct beta_b = sqrt(block_gnorm2_b) * block_cos_b, t_b* =
        # [beta_b]_+ / A_b and w_b* without any proxy.
        "block_gnorm2": block_gnorm2,
        "block_cos": block_cos,
        "block_norm": block_norm,
        # A7 / A8
        "block_s_rms": block_s_rms,
        "kappa_b": kappa_b,
        "block_energy": block_energy,
        "block_grad_update": block_grad_update,
        "grad_norm2_full": grad_norm2_full,
        "update_energy_full": update_energy_raw,
        "grad_update_full": grad_update_full,
        "id_gnorm2_rel": id_gnorm2_rel,
        "id_update_energy_rel": id_update_energy_rel,
        "id_grad_update_rel": id_grad_update_rel,
        "sum_wb_Ab": sum_wb_Ab,
        "sum_wb_AMb": sum_wb_AMb,
        "cross_block_resid": cross_block_resid,
        # 1d
        "type_energy_frac": type_energy_frac,       # (n_types,)
        "namesake_energy_frac": namesake_energy_frac,
        "A_restricted": A_restricted,
        "AM_S": AM_S,
        # A5 — coordinate-frame allocation / coupling (signed pair, paired SE)
        "A_diag_coord": A_diag_coord,
        "cov_term": cov_term,
        "cov_term_se": cov_term_se,
        "Gamma_coord": Gamma_coord,
        "nnz_frac": nnz_frac,
        "A_diag_coord_check": A_diag_coord_check,
        "A_diag_coord_check_se": A_diag_coord_check_se,
        # A9 — two-group split
        "A_fallback": A_fallback,
        "w_fallback": w_F,
        "Gamma_MF": Gamma_MF,
        "A_restricted_over_lammax": A_restricted_over_lammax,
        "AM_S_over_lammax": AM_S_over_lammax,
        "r_S": r_S,
    }
