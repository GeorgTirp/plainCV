# optim/pns_eigenadam.py
from typing import Any, Callable, NamedTuple, Optional, Tuple
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax

Array = jax.Array
Params = Any
PyTree = Any

# Type: given params, direction (same PyTree structure as params), rng -> matvec(direction)
GGNMatvecFn = Callable[[Params, PyTree, Array], PyTree]


# ---------------------------------------------------------------------------
# Global PN-S EigenAdam state
# ---------------------------------------------------------------------------

class PnsEigenAdamState(NamedTuple):
    """State for PN-S EigenAdam (global eigenbasis)."""
    adam_state: optax.OptState
    step: Array
    eigenvalues: Array
    eigenvectors: Array
    rng_key: Array
    rotation_diff: Array

    # Split-spaces moments (used when split_spaces=True)
    m_top: Array
    v_top: Array
    m_perp: Array
    v_perp: Array

    # Spectral LR gates (A) + complement curvature gate (D1)
    lr_gate_top: Array          # scalar
    lr_gate_perp: Array         # scalar
    lr_ref_top: Array           # scalar "reference" curvature (set on first refresh)
    lr_ref_perp: Array          # scalar "reference" curvature (set on first refresh)
    lam_max: Array              # scalar (last refreshed)
    lam_bar_perp: Array         # scalar (last refreshed)


# ---------------------------------------------------------------------------
# Lanczos iterative eigen solver on a matrix-vector product (un-sketched)
# ---------------------------------------------------------------------------

def lanczos(
    matvec: Callable[[Array], Array],
    dim: int,
    num_iter: int,
    key: Array,
    eps: float = 1e-6,
    sort_by_abs: bool = False,
) -> Tuple[Array, Array]:
    """Run Lanczos to approximate top eigenvalues/eigenvectors.

    Returns:
      eigenvalues: (num_iter,) approximated eigenvalues (sorted).
      eigenvectors: (num_iter, dim) corresponding eigenvectors (in flattened space).
                   Returned as rows (each row is an eigenvector).
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

    evals, evecs_T = jnp.linalg.eigh(T)  # ascending

    if sort_by_abs:
        idx = jnp.argsort(jnp.abs(evals))[::-1]
    else:
        idx = jnp.argsort(evals)[::-1]

    evals = evals[idx]
    evecs_T = evecs_T[:, idx]

    V_k = V[:-1]  # (k, dim)
    eigenvectors_flat = (evecs_T.T @ V_k).reshape(k, dim)  # rows

    return evals, eigenvectors_flat


# ---------------------------------------------------------------------------
# Preconditioner (whole-gradient mode)
# ---------------------------------------------------------------------------

def apply_eigen_preconditioner(
    grad_flat: Array,
    eigenvalues: Array,
    eigenvectors: Array,
    damping: float = 1e-4,
    saddle_free_neg: bool = False,
) -> Array:
    """Apply partial Newton-like sqrt-preconditioner in eigenbasis."""
    if eigenvalues.size == 0:
        return grad_flat

    V = eigenvectors  # (k, dim) rows
    lambdas = eigenvalues  # (k,)

    proj = V @ grad_flat          # (k,)
    proj_vec = V.T @ proj         # (dim,)

    if saddle_free_neg:
        lam_eff = jnp.abs(lambdas)
    else:
        lam_eff = lambdas

    scale = 1.0 / (lam_eff + damping)
    scale = jnp.sqrt(scale)

    scaled = proj * scale         # (k,)
    new_subspace = V.T @ scaled   # (dim,)

    g_perp = grad_flat - proj_vec
    return new_subspace + g_perp


def _apply_eigenadam_whole(
    grads: PyTree,
    params: Params,
    eigenvalues: Array,
    eigenvectors: Array,
    precond_damping: float,
    use_saddle_free: bool,
    base_opt: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> Tuple[PyTree, optax.OptState]:
    """Old behaviour: precondition full gradient in eigenbasis, then apply base optimizer."""
    flat_grads, unravel_grads = ravel_pytree(grads)
    precond_flat_grads = apply_eigen_preconditioner(
        grad_flat=flat_grads,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        damping=precond_damping,
        saddle_free_neg=use_saddle_free,
    )
    precond_grads = unravel_grads(precond_flat_grads)
    updates, new_state = base_opt.update(precond_grads, opt_state, params=params)
    return updates, new_state


# ---------------------------------------------------------------------------
# Split-spaces mode: Adam in top-k eigen-coordinates with eigenvalue normalization
# + Adam in complement, with spectral LR gates (A) and complement curvature LR gate (D1).
# ---------------------------------------------------------------------------

def _apply_split_adam_topk(
    grads: PyTree,
    params: Params,
    *,
    eigenvalues: Array,
    eigenvectors: Array,
    m_top: Array,
    v_top: Array,
    m_perp: Array,
    v_perp: Array,
    step: Array,
    lr_top_eff: float,
    lr_perp_eff: float,
    beta1: float,
    beta2: float,
    eps: float,
    precond_damping: float,
    use_saddle_free: bool,
    weight_decay: float,
) -> Tuple[PyTree, Array, Array, Array, Array]:
    """
    - Top-k: Adam on projected gradient coefficients, multiplied by 1/(λ+δ) ("normalize in eigendirections")
    - Perp: standard diagonal Adam
    - Decoupled weight decay applied once (tied to lr_perp_eff)
    """
    flat_grads, unravel_grads = ravel_pytree(grads)
    flat_params, _ = ravel_pytree(params)

    V = eigenvectors        # (k_max, dim), rows
    lambdas = eigenvalues   # (k_max,)

    # Project gradient into tracked subspace (k_max,)
    g_top = V @ flat_grads
    g_par = V.T @ g_top
    g_perp = flat_grads - g_par

    # Bias correction
    t = step.astype(jnp.float32) + 1.0
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t

    # ---- Top space: Adam in eigen-coordinates + eigenvalue normalization
    m_top = beta1 * m_top + (1.0 - beta1) * g_top
    v_top = beta2 * v_top + (1.0 - beta2) * (g_top * g_top)

    m_top_hat = m_top / bc1
    v_top_hat = v_top / bc2

    if use_saddle_free:
        lam_eff = jnp.abs(lambdas)
    else:
        # Safe for GGN-like PSD: avoid negative (can happen from numerics)
        lam_eff = jnp.maximum(lambdas, 0.0)

    lam_eff = lam_eff + precond_damping
    s_scale = 1.0 / (lam_eff + 1e-12)  # "normalize in eigendirections"

    step_top_coords = -lr_top_eff * s_scale * m_top_hat / (jnp.sqrt(v_top_hat) + eps)
    step_top_flat = V.T @ step_top_coords  # (dim,)

    # ---- Complement: standard diagonal Adam
    m_perp = beta1 * m_perp + (1.0 - beta1) * g_perp
    v_perp = beta2 * v_perp + (1.0 - beta2) * (g_perp * g_perp)

    m_perp_hat = m_perp / bc1
    v_perp_hat = v_perp / bc2

    step_perp_flat = -lr_perp_eff * m_perp_hat / (jnp.sqrt(v_perp_hat) + eps)

    # Combine
    step_flat = step_top_flat + step_perp_flat

    # Decoupled weight decay (tie to lr_perp_eff to keep semantics stable)
    if weight_decay != 0.0:
        step_flat = step_flat - lr_perp_eff * weight_decay * flat_params

    return unravel_grads(step_flat), m_top, v_top, m_perp, v_perp


# ---------------------------------------------------------------------------
# Base optimizer construction (whole-gradient mode only)
# ---------------------------------------------------------------------------

def _make_pns_base_optimizer(
    *,
    base_optimizer: str,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    sgd_momentum: float,
    sgd_nesterov: bool,
    rmsprop_decay: Optional[float],
    rmsprop_momentum: float,
    rmsprop_centered: bool,
) -> optax.GradientTransformation:
    base_name = base_optimizer.lower().replace("-", "_")

    if base_name in {"adam", "adamw"}:
        return optax.adamw(
            learning_rate=learning_rate,
            b1=beta1,
            b2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )

    if base_name == "nadamw":
        return optax.nadamw(
            learning_rate=learning_rate,
            b1=beta1,
            b2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )

    if base_name in {"nesterovw", "nagw", "nadamw_no_rms", "nadam_no_rms"}:
        tx_parts = [optax.trace(decay=beta1, nesterov=True)]
        if weight_decay != 0.0:
            tx_parts.append(optax.add_decayed_weights(weight_decay))
        tx_parts.append(optax.scale(-learning_rate))
        return optax.chain(*tx_parts)

    if base_name == "sgd":
        momentum = None if sgd_momentum <= 0.0 else sgd_momentum
        sgd_tx = optax.sgd(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=sgd_nesterov,
        )
        if weight_decay != 0.0:
            return optax.chain(optax.add_decayed_weights(weight_decay), sgd_tx)
        return sgd_tx

    if base_name in {"rmsprop", "rms_prop"}:
        decay = beta2 if rmsprop_decay is None else rmsprop_decay
        momentum = None if rmsprop_momentum <= 0.0 else rmsprop_momentum
        rmsprop_tx = optax.rmsprop(
            learning_rate=learning_rate,
            decay=decay,
            eps=eps,
            centered=rmsprop_centered,
            momentum=momentum,
        )
        if weight_decay != 0.0:
            return optax.chain(optax.add_decayed_weights(weight_decay), rmsprop_tx)
        return rmsprop_tx

    raise ValueError(
        f"Unknown pns base optimizer '{base_optimizer}'. "
        "Use one of: adam, adamw, nadamw, nesterovw, sgd, rmsprop."
    )


# ---------------------------------------------------------------------------
# Global PN-S EigenAdam wrapper
# ---------------------------------------------------------------------------

def pns_eigenadam(
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
    base_optimizer: str = "adamw",
    sgd_momentum: float = 0.0,
    sgd_nesterov: bool = False,
    rmsprop_decay: Optional[float] = None,
    rmsprop_momentum: float = 0.0,
    rmsprop_centered: bool = False,
    # --- A/D: spectral LR controllers ---
    lr_gate_ema: float = 0.1,         # smoothing of gates at refresh
    lr_gate_clip_min: float = 0.25,   # clamp gate multipliers
    lr_gate_clip_max: float = 4.0,
    hutchinson_probes: int = 1,       # D1: how many trace probes at refresh
) -> optax.GradientTransformation:
    """
    PN-S EigenAdam as an Optax gradient transformation (global eigenbasis).

    - split_spaces == False:
        * Precondition full gradient in eigenbasis, then apply chosen base optimizer.

    - split_spaces == True:
        * Top-k: Adam in eigen-coordinates with eigenvalue normalization 1/(λ+δ).
        * Perp: standard diagonal Adam.
        * Additionally: spectral LR gates:
            (A) lr_top_eff = lr_top * gate_top, where gate_top ~ (ref)/(λmax+δ)
            (D1) lr_perp_eff = lr_perp * gate_perp, where gate_perp ~ (ref)/(λbar_perp+δ)
          Gates are updated only at curvature refresh steps and smoothed by EMA.
    """
    if lanczos_iters is None:
        lanczos_iters = max_eigenvectors
    k_top = min(max_eigenvectors, lanczos_iters)

    use_saddle_free = ((backend == "hessian") or (backend == "fisher"))
    sort_by_abs = ((backend == "hessian") or (backend == "fisher"))

    base_name = base_optimizer.lower().replace("-", "_")
    base_opt = _make_pns_base_optimizer(
        base_optimizer=base_name,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        sgd_momentum=sgd_momentum,
        sgd_nesterov=sgd_nesterov,
        rmsprop_decay=rmsprop_decay,
        rmsprop_momentum=rmsprop_momentum,
        rmsprop_centered=rmsprop_centered,
    )

    def init_fn(params: Params) -> PnsEigenAdamState:
        flat_params, _ = ravel_pytree(params)
        dim = flat_params.shape[0]
        dtype = flat_params.dtype

        eigenvalues = jnp.zeros((max_eigenvectors,), dtype=dtype)
        eigenvectors = jnp.zeros((max_eigenvectors, dim), dtype=dtype)

        # We keep base optimizer state even if split_spaces=True (harmless).
        adam_state = base_opt.init(params)
        rng_key = jax.random.PRNGKey(0)
        step = jnp.array(0, dtype=jnp.int32)
        rotation_diff = jnp.array(0.0, dtype=dtype)

        m_top = jnp.zeros((max_eigenvectors,), dtype=dtype)
        v_top = jnp.zeros((max_eigenvectors,), dtype=dtype)
        m_perp = jnp.zeros((dim,), dtype=dtype)
        v_perp = jnp.zeros((dim,), dtype=dtype)

        # Gates & references (0 means "unset"; set on first curvature refresh)
        lr_gate_top = jnp.array(1.0, dtype=dtype)
        lr_gate_perp = jnp.array(1.0, dtype=dtype)
        lr_ref_top = jnp.array(0.0, dtype=dtype)
        lr_ref_perp = jnp.array(0.0, dtype=dtype)
        lam_max = jnp.array(0.0, dtype=dtype)
        lam_bar_perp = jnp.array(0.0, dtype=dtype)

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
            lr_gate_top=lr_gate_top,
            lr_gate_perp=lr_gate_perp,
            lr_ref_top=lr_ref_top,
            lr_ref_perp=lr_ref_perp,
            lam_max=lam_max,
            lam_bar_perp=lam_bar_perp,
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

        lr_gate_top = state.lr_gate_top
        lr_gate_perp = state.lr_gate_perp
        lr_ref_top = state.lr_ref_top
        lr_ref_perp = state.lr_ref_perp
        lam_max = state.lam_max
        lam_bar_perp = state.lam_bar_perp

        lr_top_base = learning_rate if lr_top is None else lr_top
        lr_perp_base = learning_rate if lr_perp is None else lr_perp

        # ---------------------------------------------------------------
        # 1) Curvature refresh (eigenpairs) + update LR gates (A & D1)
        # ---------------------------------------------------------------
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
                    lr_gate_top,
                    lr_gate_perp,
                    lr_ref_top,
                    lr_ref_perp,
                    lam_max,
                    lam_bar_perp,
                ) = carry

                flat_params, unravel_params = ravel_pytree(params)
                dim = flat_params.shape[0]
                dtype = flat_params.dtype

                def matvec_flat(v_flat: Array) -> Array:
                    v_pytree = unravel_params(v_flat)
                    Hv_pytree = ggn_matvec_fn(params, v_pytree, rng_key)
                    Hv_flat, _ = ravel_pytree(Hv_pytree)
                    return Hv_flat

                # Lanczos eigenpairs
                rng_key, lanczos_key = jax.random.split(rng_key)
                evals, evecs = lanczos(
                    matvec=matvec_flat,
                    dim=dim,
                    num_iter=lanczos_iters,
                    key=lanczos_key,
                    sort_by_abs=sort_by_abs,
                )

                # For Hessian/Fisher backends: reorder by |λ| descending
                if use_saddle_free:
                    order = jnp.argsort(jnp.abs(evals))[::-1]
                    evals = evals[order]
                    evecs = evecs[order, :]

                # Static number of modes: k_top
                prev_vecs_k = eigenvectors[:k_top, :]
                new_vecs_k = evecs[:k_top, :]

                # Rotation distance (normalized Frobenius)
                diff = new_vecs_k - prev_vecs_k
                frob_num = jnp.linalg.norm(diff)
                frob_den = jnp.linalg.norm(prev_vecs_k)
                frob_den_safe = jnp.where(frob_den > 1e-8, frob_den, 1.0)
                rotation_diff_new = jnp.where(
                    frob_den > 1e-8,
                    frob_num / frob_den_safe,
                    jnp.array(0.0, dtype=dtype),
                )

                # Store eigenpairs into fixed-size arrays
                eigenvalues_new = jnp.zeros_like(eigenvalues)
                eigenvalues_new = eigenvalues_new.at[:k_top].set(evals[:k_top])

                eigenvectors_new = jnp.zeros_like(eigenvectors)
                eigenvectors_new = eigenvectors_new.at[:k_top, :].set(new_vecs_k)

                # Moment transport for top-subspace moments (keeps Adam-top stable)
                if split_spaces and k_top > 0:
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

                # -------- A) spectral radius gate from λmax
                lam_max_new = jnp.max(jnp.abs(evals[:k_top])) if k_top > 0 else jnp.array(0.0, dtype=dtype)
                denom_top = lam_max_new + precond_damping

                # Set reference on first refresh so gate≈1 initially
                lr_ref_top_new = jnp.where(lr_ref_top > 0.0, lr_ref_top, denom_top)
                raw_gate_top = lr_ref_top_new / (denom_top + 1e-12)
                raw_gate_top = jnp.clip(raw_gate_top, lr_gate_clip_min, lr_gate_clip_max)
                lr_gate_top_new = (1.0 - lr_gate_ema) * lr_gate_top + lr_gate_ema * raw_gate_top

                # -------- D1) complement scalar curvature via Hutchinson trace
                # Estimate tr(H) with a few Rademacher probes at refresh.
                def one_probe_trace(key):
                    r = jax.random.choice(key, jnp.array([-1.0, 1.0], dtype=dtype), (dim,))
                    Hr = matvec_flat(r)
                    return jnp.vdot(r, Hr)

                if hutchinson_probes <= 1:
                    rng_key, k_tr = jax.random.split(rng_key)
                    tr_est = one_probe_trace(k_tr)
                else:
                    keys = jax.random.split(rng_key, hutchinson_probes + 1)
                    rng_key = keys[0]
                    tr_vals = jax.vmap(one_probe_trace)(keys[1:])
                    tr_est = jnp.mean(tr_vals)

                sum_top = jnp.sum(evals[:k_top]) if k_top > 0 else jnp.array(0.0, dtype=dtype)
                denom_dim = jnp.maximum(dim - k_top, 1)
                lam_bar_perp_new = (tr_est - sum_top) / denom_dim

                # For safety: keep nonnegative if using PSD curvature (GGN/Fisher).
                if not use_saddle_free:
                    lam_bar_perp_new = jnp.maximum(lam_bar_perp_new, 0.0)

                denom_perp = lam_bar_perp_new + precond_damping
                lr_ref_perp_new = jnp.where(lr_ref_perp > 0.0, lr_ref_perp, denom_perp)
                raw_gate_perp = lr_ref_perp_new / (denom_perp + 1e-12)
                raw_gate_perp = jnp.clip(raw_gate_perp, lr_gate_clip_min, lr_gate_clip_max)
                lr_gate_perp_new = (1.0 - lr_gate_ema) * lr_gate_perp + lr_gate_ema * raw_gate_perp

                return (
                    eigenvalues_new,
                    eigenvectors_new,
                    rng_key,
                    params,
                    rotation_diff_new,
                    m_top_new,
                    v_top_new,
                    lr_gate_top_new,
                    lr_gate_perp_new,
                    lr_ref_top_new,
                    lr_ref_perp_new,
                    lam_max_new,
                    lam_bar_perp_new,
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
                lr_gate_top,
                lr_gate_perp,
                lr_ref_top,
                lr_ref_perp,
                lam_max,
                lam_bar_perp,
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
                    lr_gate_top,
                    lr_gate_perp,
                    lr_ref_top,
                    lr_ref_perp,
                    lam_max,
                    lam_bar_perp,
                ),
            )

        # Effective learning rates (A + D1)
        lr_top_eff = lr_top_base * lr_gate_top
        lr_perp_eff = lr_perp_base * lr_gate_perp

        # ---------------------------------------------------------------
        # 2) Apply update
        # ---------------------------------------------------------------
        if not split_spaces:
            updates, new_adam_state = _apply_eigenadam_whole(
                grads=grads,
                params=params,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                precond_damping=precond_damping,
                use_saddle_free=use_saddle_free,
                base_opt=base_opt,
                opt_state=state.adam_state,
            )
            new_m_top, new_v_top = m_top, v_top
            new_m_perp, new_v_perp = m_perp, v_perp

        else:
            updates, new_m_top, new_v_top, new_m_perp, new_v_perp = _apply_split_adam_topk(
                grads=grads,
                params=params,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                m_top=m_top,
                v_top=v_top,
                m_perp=m_perp,
                v_perp=v_perp,
                step=step,
                lr_top_eff=lr_top_eff,
                lr_perp_eff=lr_perp_eff,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                precond_damping=precond_damping,
                use_saddle_free=use_saddle_free,
                weight_decay=weight_decay,
            )
            # base optimizer state unused in split mode
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
            lr_gate_top=lr_gate_top,
            lr_gate_perp=lr_gate_perp,
            lr_ref_top=lr_ref_top,
            lr_ref_perp=lr_ref_perp,
            lam_max=lam_max,
            lam_bar_perp=lam_bar_perp,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Profiling helper (manual, not used inside the JITted training loop)
# ---------------------------------------------------------------------------

def profile_pns_eigenadam_curvature(
    params: Params,
    ggn_matvec_fn: GGNMatvecFn,
    max_eigenvectors: int = 16,
    lanczos_iters: Optional[int] = None,
    rng: Optional[Array] = None,
    warmup: bool = True,
) -> None:
    """Run a single curvature update with line_profiler + wall-clock timing."""
    try:
        from line_profiler import LineProfiler
    except ImportError as exc:
        raise ImportError(
            "line_profiler is required for profiling. Install via `pip install line_profiler`."
        ) from exc

    if lanczos_iters is None:
        lanczos_iters = max_eigenvectors
    if rng is None:
        rng = jax.random.PRNGKey(0)

    flat_params, unravel_params = ravel_pytree(params)
    dim = flat_params.shape[0]
    rng, lanczos_key = jax.random.split(rng)

    def matvec_flat(v_flat: Array) -> Array:
        v_pytree = unravel_params(v_flat)
        Hv_pytree = ggn_matvec_fn(params, v_pytree, rng)
        Hv_flat, _ = ravel_pytree(Hv_pytree)
        return Hv_flat

    if warmup:
        _ = lanczos(matvec_flat, dim=dim, num_iter=lanczos_iters, key=lanczos_key)

    lp = LineProfiler()
    profiled_lanczos = lp(lanczos)
    profiled_matvec = lp(matvec_flat)

    start = time.perf_counter()
    evals, evecs = profiled_lanczos(
        profiled_matvec,
        dim=dim,
        num_iter=lanczos_iters,
        key=lanczos_key,
    )
    jax.block_until_ready((evals, evecs))
    elapsed = time.perf_counter() - start

    print(f"[PN-S EigenAdam] Curvature step wall-clock: {elapsed:.3f} s")
    print(f"Top eigenvalues (first 5): {jnp.asarray(evals)[:5]}")
    lp.print_stats()
