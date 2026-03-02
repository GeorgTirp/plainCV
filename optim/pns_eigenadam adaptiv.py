# optim/pns_eigenadam.py
from typing import Any, Callable, NamedTuple, Optional, Tuple
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import tree_util as jtu
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
    base_adam: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> Tuple[PyTree, optax.OptState]:
    """Precondition full gradient in eigenbasis, then apply base optimizer."""
    flat_grads, unravel_grads = ravel_pytree(grads)
    precond_flat_grads = apply_eigen_preconditioner(
        grad_flat=flat_grads,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        damping=precond_damping,
        saddle_free_neg=use_saddle_free,
    )
    precond_grads = unravel_grads(precond_flat_grads)

    updates, new_adam_state = base_adam.update(
        precond_grads,
        opt_state,
        params=params,
    )
    return updates, new_adam_state


def _apply_eigenadam_split_spaces(
    grads: PyTree,
    params: Params,
    eigenvalues: Array,
    eigenvectors: Array,
    m_top: Array,    # unused for pure Newton, but kept for state compatibility
    v_top: Array,    # unused
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
    """
    Split-space behaviour:

      - Top-k eigen-subspace: damped truncated Newton step with lr_top.
      - Orthogonal complement: standard Adam with lr_perp.

    m_top, v_top are currently not used (pure Newton), but kept for future
    extensions and for state shape compatibility.
    """
    flat_grads, unravel_grads = ravel_pytree(grads)
    flat_params, _ = ravel_pytree(params)

    V = eigenvectors           # (k_max, dim)
    lambdas = eigenvalues      # (k_max,)

    # 1) Split gradient into top-k and complement
    proj = V @ flat_grads          # α (k_max,)
    g_par = V.T @ proj             # (dim,)
    g_perp = flat_grads - g_par    # (dim,)

    # 2) Adam bias correction factors
    t = step.astype(jnp.float32) + 1.0
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t

    # 3) Complement Adam on g_perp
    m_perp = beta1 * m_perp + (1.0 - beta1) * g_perp
    v_perp = beta2 * v_perp + (1.0 - beta2) * (g_perp * g_perp)

    m_perp_hat = m_perp / bc1
    v_perp_hat = v_perp / bc2

    step_perp_flat = -lr_perp * m_perp_hat / (jnp.sqrt(v_perp_hat) + eps)

    # 4) Top-k Newton-style step (no Adam, just H^{-1} g)
    if use_saddle_free:
        lam_eff = jnp.abs(lambdas)
    else:
        # Clamp negatives to zero for GGN safety
        lam_eff = jnp.maximum(lambdas, 0.0)

    lam_eff = lam_eff + precond_damping
    newton_coeffs = proj / (lam_eff + 1e-12)     # α_i / (λ_eff_i + δ)

    step_top_flat = -lr_top * (V.T @ newton_coeffs)   # (dim,)

    # 5) Combine
    step_flat = step_top_flat + step_perp_flat

    # 6) Decoupled weight decay (tied to lr_perp)
    if weight_decay != 0.0:
        step_flat = step_flat - lr_perp * weight_decay * flat_params

    updates = unravel_grads(step_flat)

    # m_top, v_top passed through unchanged (pure Newton on top-k)
    return updates, m_top, v_top, m_perp, v_perp


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
    """Build base optimizer used by PN-S after curvature preconditioning."""
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

    # "NadamW without RMSProp part":
    # Nesterov momentum + decoupled-style weight decay.
    if base_name in {"nesterovw", "nagw", "nadamw_no_rms", "nadam_no_rms"}:
        tx_parts = [
            optax.trace(decay=beta1, nesterov=True),
        ]
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
# Global PN-S EigenAdam wrapper (single global eigenbasis, optional sketching,
# and optional saddle-free treatment when curvature_is_hessian=True)
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
    innovation_enabled: bool = True,
    innovation_threshold: float = 0.2,
    innovation_check_every: int = 1,
    innovation_num_probes: int = 1,
    innovation_probe: str = "gradient",  # "gradient" or "random"
    innovation_eps: float = 1e-8,
    innovation_use_damping: bool = True,
    subspace_tracking_enabled: bool = True,
    subspace_tracking_every: int = 1,
    subspace_tracking_alpha: float = 0.1,
    subspace_tracking_power_iters: int = 1,
    subspace_tracking_on_probe_steps: bool = True,
    eigenvalue_keep_threshold: float = 5.0,
    freeze_subspace_after_threshold: bool = True,
    keep_at_least_one_eigenpair: bool = False,
    base_optimizer: str = "adamw",
    sgd_momentum: float = 0.0,
    sgd_nesterov: bool = False,
    rmsprop_decay: Optional[float] = None,
    rmsprop_momentum: float = 0.0,
    rmsprop_centered: bool = False,
) -> optax.GradientTransformation:

    """
    PN-S EigenAdam as an Optax gradient transformation (global eigenbasis).

    - split_spaces == False:
        * EXACTLY your old behaviour:
            - estimate top-k curvature eigenpairs via Lanczos,
            - apply eigenbasis square-root preconditioner,
            - run selected base optimizer on the preconditioned gradient.

    - split_spaces == True:
        * Same eigenbasis, but:
            - perform Adam in the top-k subspace in eigen coordinates,
              with curvature-based scaling,
            - perform standard diagonal Adam in the orthogonal complement,
            - combine both steps + decoupled weight decay.

    Additional adaptive behavior:
      - Curvature refreshes are event-driven via an innovation test.
      - Eigenmodes with eigenvalue < eigenvalue_keep_threshold are dropped.
      - If freeze_subspace_after_threshold=True, active subspace size is
        fixed after first refresh based on that threshold.
      - Optional online subspace tracking between Lanczos refreshes.
    """
    if innovation_check_every < 1:
        raise ValueError("innovation_check_every must be >= 1")
    if innovation_num_probes < 1:
        raise ValueError("innovation_num_probes must be >= 1")
    if subspace_tracking_every < 1:
        raise ValueError("subspace_tracking_every must be >= 1")
    if subspace_tracking_power_iters < 1:
        raise ValueError("subspace_tracking_power_iters must be >= 1")
    if not (0.0 <= subspace_tracking_alpha <= 1.0):
        raise ValueError("subspace_tracking_alpha must be in [0, 1]")

    if lanczos_iters is None:
        lanczos_iters = max_eigenvectors
    # Static number of tracked modes (no JAX tracing here).
    k_top = min(max_eigenvectors, lanczos_iters)

    base_name = base_optimizer.lower().replace("-", "_")
    if split_spaces and base_name != "adamw":
        raise ValueError(
            "split_spaces=True currently supports only pns_base_optimizer='adamw'."
        )

    base_adam = _make_pns_base_optimizer(
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

        opt_state = base_adam.init(params)
        rng_key = jax.random.PRNGKey(0)
        step = jnp.array(0, dtype=jnp.int32)
        rotation_diff = jnp.array(0.0, dtype=dtype)
        last_refresh_step = jnp.array(-1, dtype=jnp.int32)
        active_k = jnp.array(-1, dtype=jnp.int32)
        innovation_residual = jnp.array(0.0, dtype=dtype)

        # Extra state for split_spaces mode (harmless otherwise).
        m_top = jnp.zeros((max_eigenvectors,), dtype=dtype)
        v_top = jnp.zeros((max_eigenvectors,), dtype=dtype)
        m_perp = jnp.zeros((dim,), dtype=dtype)
        v_perp = jnp.zeros((dim,), dtype=dtype)

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
            last_refresh_step=last_refresh_step,
            active_k=active_k,
            innovation_residual=innovation_residual,
        )

    def update_fn(
        grads: PyTree,
        state: PnsEigenAdamState,
        params: Optional[Params] = None,
    ):
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
        last_refresh_step = state.last_refresh_step
        active_k = state.active_k
        innovation_residual = state.innovation_residual
        lr_top_eff = learning_rate if lr_top is None else lr_top
        lr_perp_eff = learning_rate if lr_perp is None else lr_perp

        use_saddle_free = ((backend == "hessian") or (backend == "fisher"))
        sort_by_abs = ((backend == "hessian") or (backend == "fisher"))

        # ---------------------------------------------------------------
        # 1. Curvature update:
        #    - event-driven (innovation residual)
        #    - optional periodic fallback (max staleness)
        #    - eigenvalue thresholding + optional frozen active subspace size
        # ---------------------------------------------------------------
        if ggn_matvec_fn is not None:
            flat_grads, _ = ravel_pytree(grads)
            flat_params, unravel_params = ravel_pytree(params)
            dim = flat_params.shape[0]
            dtype = flat_params.dtype

            def matvec_flat_with_key(v_flat: Array, matvec_key: Array) -> Array:
                v_pytree = unravel_params(v_flat)
                Hv_pytree = ggn_matvec_fn(params, v_pytree, matvec_key)
                Hv_flat, _ = ravel_pytree(Hv_pytree)
                return Hv_flat

            if curvature_update_every > 0:
                periodic_due = jnp.logical_or(
                    last_refresh_step < 0,
                    (step - last_refresh_step) >= curvature_update_every,
                )
            else:
                periodic_due = last_refresh_step < 0

            if innovation_check_every > 1:
                innovation_check_due = (step % innovation_check_every) == 0
            else:
                innovation_check_due = jnp.array(True)

            innovation_mode = innovation_probe.lower()
            use_gradient_probe = innovation_mode in {"gradient", "grad", "g"}
            should_run_innovation = jnp.logical_and(
                jnp.logical_and(
                    innovation_enabled and (innovation_threshold >= 0.0),
                    innovation_check_due,
                ),
                jnp.logical_and(
                    last_refresh_step >= 0,
                    jnp.logical_not(periodic_due),
                ),
            )

            def run_innovation_test(carry):
                eigenvalues, eigenvectors, rng_key = carry
                grad_norm = jnp.linalg.norm(flat_grads)

                def probe_once(i, probe_carry):
                    rho_sum, probe_rng = probe_carry
                    probe_rng, vec_key = jax.random.split(probe_rng)
                    probe_rng, hvp_key = jax.random.split(probe_rng)

                    if use_gradient_probe:
                        base_vec = jax.lax.cond(
                            jnp.logical_and(i == 0, grad_norm > innovation_eps),
                            lambda _: flat_grads,
                            lambda _: jax.random.normal(vec_key, (dim,), dtype=dtype),
                            operand=None,
                        )
                    else:
                        base_vec = jax.random.normal(vec_key, (dim,), dtype=dtype)

                    probe_vec = base_vec / (jnp.linalg.norm(base_vec) + innovation_eps)
                    Hv = matvec_flat_with_key(probe_vec, hvp_key)

                    coeffs = eigenvectors @ probe_vec
                    Hv_hat = eigenvectors.T @ (eigenvalues * coeffs)
                    if innovation_use_damping:
                        Hv_hat = Hv_hat + precond_damping * probe_vec

                    rho = jnp.linalg.norm(Hv - Hv_hat) / (
                        jnp.linalg.norm(Hv) + innovation_eps
                    )
                    return rho_sum + rho, probe_rng

                rho0 = jnp.array(0.0, dtype=dtype)
                rho_sum, rng_key = jax.lax.fori_loop(
                    0,
                    innovation_num_probes,
                    probe_once,
                    (rho0, rng_key),
                )
                rho = rho_sum / float(innovation_num_probes)
                return rho, rng_key

            innovation_residual, rng_key = jax.lax.cond(
                should_run_innovation,
                run_innovation_test,
                lambda carry: (innovation_residual, carry[2]),
                operand=(eigenvalues, eigenvectors, rng_key),
            )

            innovation_due = jnp.logical_and(
                should_run_innovation,
                innovation_residual > innovation_threshold,
            )
            should_update = jnp.logical_or(periodic_due, innovation_due)

            def do_update(carry):
                (
                    eigenvalues,
                    eigenvectors,
                    rng_key,
                    params,
                    rotation_diff,
                    m_top,
                    v_top,
                    last_refresh_step,
                    active_k,
                    innovation_residual,
                ) = carry

                def matvec_flat(v_flat: Array) -> Array:
                    return matvec_flat_with_key(v_flat, rng_key)

                rng_key, lanczos_key = jax.random.split(rng_key)
                evals, evecs = lanczos(
                    matvec=matvec_flat,
                    dim=dim,
                    num_iter=lanczos_iters,  # STATIC int
                    key=lanczos_key,
                    sort_by_abs=sort_by_abs,
                )  # evecs: (lanczos_iters, dim)

                # For Hessian/Fisher backend, keep |λ|-ordering safety.
                if (backend == "hessian") or (backend == "fisher"):
                    order = jnp.argsort(jnp.abs(evals))[::-1]
                    evals = evals[order]
                    evecs = evecs[order, :]

                prev_vecs_k = eigenvectors[:k_top, :]  # (k_top, dim)
                new_vecs_k = evecs[:k_top, :]          # (k_top, dim)
                evals_k = evals[:k_top]

                # Decide active subspace size from threshold.
                kept_now = jnp.sum(
                    evals_k >= eigenvalue_keep_threshold,
                    dtype=jnp.int32,
                )
                if keep_at_least_one_eigenpair:
                    kept_now = jnp.maximum(kept_now, jnp.array(1, dtype=jnp.int32))
                kept_now = jnp.minimum(kept_now, jnp.array(k_top, dtype=jnp.int32))

                if freeze_subspace_after_threshold:
                    active_k_new = jax.lax.cond(
                        active_k < 0,
                        lambda _: kept_now,
                        lambda _: active_k,
                        operand=None,
                    )
                else:
                    active_k_new = kept_now

                mode_mask = jnp.arange(k_top) < active_k_new
                mode_mask_f = mode_mask.astype(evals_k.dtype)

                evals_k = evals_k * mode_mask_f
                new_vecs_k = jnp.where(
                    mode_mask[:, None],
                    new_vecs_k,
                    jnp.zeros_like(new_vecs_k),
                )

                # Rotation distance (normalized Frobenius) on active basis.
                diff = new_vecs_k - prev_vecs_k
                frob_num = jnp.linalg.norm(diff)
                frob_den = jnp.linalg.norm(prev_vecs_k)
                frob_den_safe = jnp.where(frob_den > 1e-8, frob_den, 1.0)
                rotation_diff_new = jnp.where(
                    frob_den > 1e-8,
                    frob_num / frob_den_safe,
                    jnp.array(0.0, dtype=dtype),
                )

                # Store eigenvalues/eigenvectors in fixed-size arrays.
                eigenvalues_new = jnp.zeros_like(eigenvalues)
                eigenvalues_new = eigenvalues_new.at[:k_top].set(evals_k)

                eigenvectors_new = jnp.zeros_like(eigenvectors)
                eigenvectors_new = eigenvectors_new.at[:k_top, :].set(new_vecs_k)

                # Optional moment transport for top-subspace moments.
                if split_spaces and k_top > 0:
                    R = new_vecs_k @ prev_vecs_k.T  # (k_top, k_top)

                    m_top_small = m_top[:k_top]
                    v_top_small = v_top[:k_top]

                    m_top_new_small = (R @ m_top_small) * mode_mask_f
                    v_top_new_small = (R @ v_top_small) * mode_mask_f  # heuristic

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
                    step,
                    active_k_new,
                    jnp.array(0.0, dtype=dtype),
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
                last_refresh_step,
                active_k,
                innovation_residual,
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
                    last_refresh_step,
                    active_k,
                    innovation_residual,
                ),
            )

            # ---------------------------------------------------------------
            # 1b. Streaming subspace tracking between full Lanczos refreshes:
            #     U <- qf((1-alpha) U + alpha H(U))
            # ---------------------------------------------------------------
            if subspace_tracking_every > 1:
                tracking_every_due = (step % subspace_tracking_every) == 0
            else:
                tracking_every_due = jnp.array(True)

            if subspace_tracking_on_probe_steps:
                tracking_probe_due = innovation_check_due
            else:
                tracking_probe_due = jnp.array(True)

            should_track = jnp.logical_and(
                jnp.logical_and(
                    subspace_tracking_enabled,
                    jnp.logical_and(
                        tracking_every_due,
                        tracking_probe_due,
                    ),
                ),
                jnp.logical_and(
                    jnp.logical_and(last_refresh_step >= 0, active_k > 0),
                    jnp.logical_not(should_update),
                ),
            )

            def do_track(carry):
                (
                    eigenvalues,
                    eigenvectors,
                    rng_key,
                    rotation_diff,
                    m_top,
                    v_top,
                ) = carry

                active_k_eff = jnp.minimum(
                    jnp.maximum(active_k, 0),
                    jnp.array(k_top, dtype=jnp.int32),
                )
                mode_mask = jnp.arange(k_top) < active_k_eff
                mode_mask_f = mode_mask.astype(dtype)

                U_prev = eigenvectors[:k_top, :]
                U_prev = jnp.where(
                    mode_mask[:, None],
                    U_prev,
                    jnp.zeros_like(U_prev),
                )

                def apply_H_to_rows(U_rows: Array, key_in: Array) -> tuple[Array, Array]:
                    HU_init = jnp.zeros_like(U_rows)

                    def hvp_body(i, hvp_carry):
                        HU_rows, key_loop = hvp_carry
                        key_loop, hvp_key = jax.random.split(key_loop)
                        u_i = U_rows[i]
                        hv_i = jax.lax.cond(
                            mode_mask[i],
                            lambda _: matvec_flat_with_key(u_i, hvp_key),
                            lambda _: jnp.zeros((dim,), dtype=dtype),
                            operand=None,
                        )
                        HU_rows = HU_rows.at[i].set(hv_i)
                        return HU_rows, key_loop

                    return jax.lax.fori_loop(0, k_top, hvp_body, (HU_init, key_in))

                def power_body(_i, power_carry):
                    U_rows, key_power = power_carry
                    HU_rows, key_power = apply_H_to_rows(U_rows, key_power)

                    mixed_cols = (
                        (1.0 - subspace_tracking_alpha) * U_rows.T
                        + subspace_tracking_alpha * HU_rows.T
                    )  # (dim, k_top)
                    Q, _ = jnp.linalg.qr(mixed_cols)
                    U_next = Q.T
                    U_next = jnp.where(
                        mode_mask[:, None],
                        U_next,
                        jnp.zeros_like(U_next),
                    )
                    return U_next, key_power

                U_tracked, rng_key = jax.lax.fori_loop(
                    0,
                    subspace_tracking_power_iters,
                    power_body,
                    (U_prev, rng_key),
                )

                diff = U_tracked - U_prev
                frob_num = jnp.linalg.norm(diff)
                frob_den = jnp.linalg.norm(U_prev)
                frob_den_safe = jnp.where(frob_den > 1e-8, frob_den, 1.0)
                rotation_diff_new = jnp.where(
                    frob_den > 1e-8,
                    frob_num / frob_den_safe,
                    jnp.array(0.0, dtype=dtype),
                )

                eigenvectors_new = jnp.zeros_like(eigenvectors)
                eigenvectors_new = eigenvectors_new.at[:k_top, :].set(U_tracked)

                eigenvalues_new = jnp.zeros_like(eigenvalues)
                eigenvalues_new = eigenvalues_new.at[:k_top].set(
                    eigenvalues[:k_top] * mode_mask_f
                )

                if split_spaces and k_top > 0:
                    R = U_tracked @ U_prev.T  # (k_top, k_top)
                    m_top_small = m_top[:k_top]
                    v_top_small = v_top[:k_top]

                    m_top_new_small = (R @ m_top_small) * mode_mask_f
                    v_top_new_small = (R @ v_top_small) * mode_mask_f  # heuristic

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
                    rotation_diff_new,
                    m_top_new,
                    v_top_new,
                )

            def dont_track(carry):
                return carry

            (
                eigenvalues,
                eigenvectors,
                rng_key,
                rotation_diff,
                m_top,
                v_top,
            ) = jax.lax.cond(
                should_track,
                do_track,
                dont_track,
                operand=(
                    eigenvalues,
                    eigenvectors,
                    rng_key,
                    rotation_diff,
                    m_top,
                    v_top,
                ),
            )

        # ---------------------------------------------------------------
        # 2. Apply update: either old behaviour or split-spaces version.
        #    The choice is a PURE Python flag → compile-time for JAX.
        # ---------------------------------------------------------------
        if not split_spaces:
            # Old behaviour (exactly your previous code path)
            updates, new_adam_state = _apply_eigenadam_whole(
                grads=grads,
                params=params,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                precond_damping=precond_damping,
                use_saddle_free=use_saddle_free,
                base_adam=base_adam,
                opt_state=state.opt_state,
            )
            # Moments unchanged in this mode
            new_m_top = m_top
            new_v_top = v_top
            new_m_perp = m_perp
            new_v_perp = v_perp

        else:
            # New two-space behaviour: Newton on top-k, Adam on complement
            updates, new_m_top, new_v_top, new_m_perp, new_v_perp = (
                _apply_eigenadam_split_spaces(
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
            )
            new_adam_state = state.opt_state

        new_state = PnsEigenAdamState(
            opt_state=new_adam_state,
            step=step,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rng_key=rng_key,
            rotation_diff=rotation_diff,
            m_top=new_m_top,
            v_top=new_v_top,
            m_perp=new_m_perp,
            v_perp=new_v_perp,
            last_refresh_step=last_refresh_step,
            active_k=active_k,
            innovation_residual=innovation_residual,
        )

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)




# ---------------------------------------------------------------------------
# Profiling helpers (manual, not used inside the JITted training loop)
# ---------------------------------------------------------------------------

def profile_pns_eigenadam_curvature(
    params: Params,
    ggn_matvec_fn: GGNMatvecFn,
    max_eigenvectors: int = 16,
    lanczos_iters: Optional[int] = None,
    rng: Optional[Array] = None,
    warmup: bool = True,
) -> None:
    """
    Run a single curvature update with line_profiler + wall-clock timing.

    This is meant for manual debugging outside the JITted training loop.
    Usage example:
        from optim.pns_eigenadam import profile_pns_eigenadam_curvature
        profile_pns_eigenadam_curvature(params, ggn_matvec_fn, max_eigenvectors=8)
    """
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
        # Trigger compilation so the profiled run focuses on execution time.
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
    # Wait for device execution so timing reflects the full runtime.
    jax.block_until_ready((evals, evecs))
    elapsed = time.perf_counter() - start

    print(f"[PN-S EigenAdam] Curvature step wall-clock: {elapsed:.3f} s")
    print(f"Top eigenvalues (first 5): {jnp.asarray(evals)[:5]}")
    lp.print_stats()
