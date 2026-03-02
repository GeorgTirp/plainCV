# optim/pns_eigenadam.py
from typing import Any, Callable, NamedTuple, Optional, Tuple
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax

from .eigentools import PnsEigenAdamState, lanczos, apply_eigen_preconditioner

Array = jax.Array
Params = Any
PyTree = Any

# Type: given params, direction (same PyTree structure as params), rng -> matvec(direction)
GGNMatvecFn = Callable[[Params, PyTree, Array], PyTree]



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
# Split-spaces mode: dedicated eigenspace EMA/Adam moments in top-k coordinates
# + selected base optimizer in the orthogonal complement.
# ---------------------------------------------------------------------------

def _apply_split_adam_topk(
    grads: PyTree,
    params: Params,
    *,
    eigenvalues: Array,
    eigenvectors: Array,
    m_top: Array,
    v_top: Array,
    step: Array,
    top_modes_for_ema: int,
    lr_top_eff: float,
    beta1: float,
    beta2: float,
    eps: float,
    perp_lr_scale: float,
    base_opt: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> Tuple[PyTree, optax.OptState, Array, Array]:
    """
    - Top-(k-1) eigenspace: dedicated EMA/Adam moments in eigen-coordinates.
      The last tracked eigendirection is excluded from top-space updates.
    - Complement space: updates from the selected base optimizer only.
      We project base updates back onto the complement to keep the split strict.
    """
    flat_grads, unravel_grads = ravel_pytree(grads)
    V = eigenvectors  # (k_max, dim), rows

    k_max = V.shape[0]
    top_mask = (jnp.arange(k_max) < top_modes_for_ema).astype(flat_grads.dtype)
    V_top = V * top_mask[:, None]

    # Project gradient into top-(k-1) subspace
    g_top = V_top @ flat_grads
    g_par = V_top.T @ g_top
    g_perp = flat_grads - g_par

    # ---- Top space: EMA/Adam in top-(k-1) coordinates
    t = step.astype(jnp.float32) + 1.0
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t

    m_top = top_mask * (beta1 * m_top + (1.0 - beta1) * g_top)
    v_top = top_mask * (beta2 * jnp.maximum(v_top, 0.0) + (1.0 - beta2) * (g_top * g_top))

    m_top_hat = top_mask * (m_top / bc1)
    v_top_hat = top_mask * jnp.maximum(v_top / bc2, 0.0)
    step_top_coords = -lr_top_eff * m_top_hat / jnp.sqrt(v_top_hat + eps)
    step_top_flat = V_top.T @ step_top_coords  # (dim,)

    # ---- Complement: selected base optimizer, driven by g_perp only
    g_perp_tree = unravel_grads(g_perp)
    base_updates, new_opt_state = base_opt.update(g_perp_tree, opt_state, params=params)
    base_updates_flat, _ = ravel_pytree(base_updates)

    # Keep base optimizer contribution in complement only.
    base_top = V_top @ base_updates_flat
    base_parallel = V_top.T @ base_top
    step_perp_flat = perp_lr_scale * (base_updates_flat - base_parallel)

    step_flat = step_top_flat + step_perp_flat
    return unravel_grads(step_flat), new_opt_state, m_top, v_top


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
    # Streaming subspace tracking between full Lanczos refreshes.
    subspace_tracking_enabled: bool = False,
    subspace_tracking_every: int = 1,
    subspace_tracking_alpha: float = 0.1,
    subspace_tracking_power_iters: int = 1,
    # Faster curvature eigensolve options.
    lanczos_warm_start: bool = True,
    lanczos_light_ortho: bool = True,
    lanczos_light_ortho_every: int = 4,
    # EOS-style automatic complement LR from the last tracked top-k eigenvalue.
    perp_eos_enabled: bool = True,
    perp_eos_gamma: float = 1.0,
    perp_eos_ema: float = 0.1,
    perp_eos_min: float = 1.0e-6,
    perp_eos_max: float = 1.0,
) -> optax.GradientTransformation:
    """
    PN-S EigenAdam as an Optax gradient transformation (global eigenbasis).

    - split_spaces == False:
        * Precondition full gradient in eigenbasis, then apply chosen base optimizer.

    - split_spaces == True:
        * Top-(k-1): EMA/Adam smoothing of projected coefficients.
        * Perp: selected base optimizer on the complement space only.
        * Optional EOS-controlled lr_perp from the last tracked top-k eigenvalue.
    - Between full refreshes:
        * Optional Oja/PAST-style streaming subspace tracking using HVPs.
    - Curvature refresh acceleration:
        * Optional Lanczos warm-start + light reorthogonalization.
    """
    if lanczos_iters is None:
        lanczos_iters = max_eigenvectors
    k_top = min(max_eigenvectors, lanczos_iters)

    use_saddle_free = ((backend == "hessian") or (backend == "fisher"))
    sort_by_abs = ((backend == "hessian") or (backend == "fisher"))
    track_every = max(int(subspace_tracking_every), 1)
    track_alpha = float(jnp.clip(jnp.array(subspace_tracking_alpha, dtype=jnp.float32), 0.0, 1.0))
    track_power_iters = max(int(subspace_tracking_power_iters), 1)
    light_ortho_every = max(int(lanczos_light_ortho_every), 1)
    eos_gamma = float(jnp.clip(jnp.array(perp_eos_gamma, dtype=jnp.float32), 1e-6, 2.0))
    eos_ema = float(jnp.clip(jnp.array(perp_eos_ema, dtype=jnp.float32), 0.0, 1.0))
    eos_min = float(jnp.maximum(jnp.array(perp_eos_min, dtype=jnp.float32), 1e-12))
    eos_max = float(jnp.maximum(jnp.array(perp_eos_max, dtype=jnp.float32), eos_min))
    top_modes_for_ema = max(k_top - 1, 0)

    base_name = base_optimizer.lower().replace("-", "_")
    lr_perp_base = learning_rate if lr_perp is None else lr_perp
    base_lr = lr_perp_base if split_spaces else learning_rate
    base_opt = _make_pns_base_optimizer(
        base_optimizer=base_name,
        learning_rate=base_lr,
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
        opt_state = base_opt.init(params)
        rng_key = jax.random.PRNGKey(0)
        step = jnp.array(0, dtype=jnp.int32)
        rotation_diff = jnp.array(0.0, dtype=dtype)

        m_top = jnp.zeros((max_eigenvectors,), dtype=dtype)
        v_top = jnp.zeros((max_eigenvectors,), dtype=dtype)
        m_perp = jnp.zeros((dim,), dtype=dtype)
        v_perp = jnp.zeros((dim,), dtype=dtype)
        lr_perp_eff = jnp.array(lr_perp_base, dtype=dtype)

        return PnsEigenAdamState(
            opt_state=opt_state,
            step=step,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rng_key=rng_key,
            rotation_diff=rotation_diff,
            m_top=m_top,
            v_top=v_top,
            m_perp=m_perp,
            v_perp=v_perp,
            lr_perp_eff=lr_perp_eff,
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
        lr_perp_eff = state.lr_perp_eff

        lr_top_eff = learning_rate if lr_top is None else lr_top
        should_update = jnp.array(False)

        # ---------------------------------------------------------------
        # 1) Curvature refresh (eigenpairs)
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
                warm_start_v = eigenvectors[0] if lanczos_warm_start else None
                evals, evecs = lanczos(
                    matvec=matvec_flat,
                    dim=dim,
                    num_iter=lanczos_iters,
                    key=lanczos_key,
                    sort_by_abs=sort_by_abs,
                    init_v=warm_start_v,
                    use_light_ortho=lanczos_light_ortho,
                    light_ortho_every=light_ortho_every,
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
                    # Keep second moments nonnegative under basis change.
                    v_top_new_small = (R * R) @ jnp.maximum(v_top_small, 0.0)

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

            # Optional streaming subspace tracking between full Lanczos refreshes.
            if subspace_tracking_enabled and k_top > 0:
                basis_ready = jnp.linalg.norm(eigenvectors[:k_top, :]) > 1e-8
                should_track = jnp.logical_and(
                    jnp.logical_and(jnp.logical_not(should_update), basis_ready),
                    (step % track_every) == 0,
                )

                def do_track(carry):
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
                    dtype = flat_params.dtype
                    prev_vecs_k = eigenvectors[:k_top, :]   # (k_top, dim)
                    U_prev_cols = prev_vecs_k.T             # (dim, k_top)

                    def hvp_cols(U_cols: Array, rng_in: Array) -> Tuple[Array, Array]:
                        keys = jax.random.split(rng_in, k_top + 1)
                        rng_out = keys[0]

                        def one_hvp(v_col: Array, key_col: Array) -> Array:
                            v_pytree = unravel_params(v_col)
                            Hv_pytree = ggn_matvec_fn(params, v_pytree, key_col)
                            Hv_flat, _ = ravel_pytree(Hv_pytree)
                            return Hv_flat

                        HU_cols = jax.vmap(one_hvp, in_axes=(1, 0), out_axes=1)(U_cols, keys[1:])
                        return HU_cols, rng_out

                    def power_step(_i, power_carry):
                        U_cols, rng_step = power_carry
                        HU_cols, rng_step = hvp_cols(U_cols, rng_step)
                        Q_cols, _ = jnp.linalg.qr(HU_cols, mode="reduced")
                        return (Q_cols, rng_step)

                    U_tilde_cols, rng_key = jax.lax.fori_loop(
                        0,
                        track_power_iters,
                        power_step,
                        (U_prev_cols, rng_key),
                    )

                    U_mix_cols = (1.0 - track_alpha) * U_prev_cols + track_alpha * U_tilde_cols
                    U_new_cols, _ = jnp.linalg.qr(U_mix_cols, mode="reduced")
                    U_new_rows = U_new_cols.T  # (k_top, dim)

                    HU_new_cols, rng_key = hvp_cols(U_new_cols, rng_key)
                    evals_k = jnp.sum(U_new_cols * HU_new_cols, axis=0)
                    if not use_saddle_free:
                        evals_k = jnp.maximum(evals_k, 0.0)

                    diff = U_new_rows - prev_vecs_k
                    frob_num = jnp.linalg.norm(diff)
                    frob_den = jnp.linalg.norm(prev_vecs_k)
                    frob_den_safe = jnp.where(frob_den > 1e-8, frob_den, 1.0)
                    rotation_diff_new = jnp.where(
                        frob_den > 1e-8,
                        frob_num / frob_den_safe,
                        jnp.array(0.0, dtype=dtype),
                    )

                    eigenvalues_new = jnp.zeros_like(eigenvalues)
                    eigenvalues_new = eigenvalues_new.at[:k_top].set(evals_k)

                    eigenvectors_new = jnp.zeros_like(eigenvectors)
                    eigenvectors_new = eigenvectors_new.at[:k_top, :].set(U_new_rows)

                    if split_spaces:
                        R = U_new_rows @ prev_vecs_k.T
                        m_top_small = m_top[:k_top]
                        v_top_small = v_top[:k_top]
                        m_top_new_small = R @ m_top_small
                        v_top_new_small = (R * R) @ jnp.maximum(v_top_small, 0.0)

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

                (
                    eigenvalues,
                    eigenvectors,
                    rng_key,
                    _,
                    rotation_diff,
                    m_top,
                    v_top,
                ) = jax.lax.cond(
                    should_track,
                    do_track,
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

        if split_spaces and perp_eos_enabled:
            dtype = eigenvalues.dtype
            if k_top > 0:
                lam_proxy = eigenvalues[k_top - 1]
                if use_saddle_free:
                    lam_proxy = jnp.abs(lam_proxy)
                else:
                    lam_proxy = jnp.maximum(lam_proxy, 0.0)
            else:
                lam_proxy = jnp.array(0.0, dtype=dtype)

            eta_target = eos_gamma / (lam_proxy + precond_damping)
            eta_target = jnp.clip(eta_target, eos_min, eos_max)
            lr_perp_eff = jax.lax.cond(
                should_update,
                lambda prev: (1.0 - eos_ema) * prev + eos_ema * eta_target,
                lambda prev: prev,
                lr_perp_eff,
            )
        else:
            lr_perp_eff = jnp.array(lr_perp_base, dtype=eigenvalues.dtype)

        # ---------------------------------------------------------------
        # 2) Apply update
        # ---------------------------------------------------------------
        if not split_spaces:
            updates, new_opt_state = _apply_eigenadam_whole(
                grads=grads,
                params=params,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                precond_damping=precond_damping,
                use_saddle_free=use_saddle_free,
                base_opt=base_opt,
                opt_state=state.opt_state,
            )
            new_m_top, new_v_top = m_top, v_top
            new_m_perp, new_v_perp = m_perp, v_perp
            new_lr_perp_eff = lr_perp_eff

        else:
            perp_lr_scale = lr_perp_eff / (lr_perp_base + 1e-12)
            updates, new_opt_state, new_m_top, new_v_top = _apply_split_adam_topk(
                grads=grads,
                params=params,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                m_top=m_top,
                v_top=v_top,
                step=step,
                top_modes_for_ema=top_modes_for_ema,
                lr_top_eff=lr_top_eff,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                perp_lr_scale=perp_lr_scale,
                base_opt=base_opt,
                opt_state=state.opt_state,
            )
            # Complement moments are handled by the base optimizer state.
            new_m_perp, new_v_perp = m_perp, v_perp
            new_lr_perp_eff = lr_perp_eff

        new_state = PnsEigenAdamState(
            opt_state=new_opt_state,
            step=step,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rng_key=rng_key,
            rotation_diff=rotation_diff,
            m_top=new_m_top,
            v_top=new_v_top,
            m_perp=new_m_perp,
            v_perp=new_v_perp,
            lr_perp_eff=new_lr_perp_eff,
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
