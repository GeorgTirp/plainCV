from __future__ import annotations

import math
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax

from .matrix_routing import path_to_name, should_use_matrix_preconditioner

Array = jax.Array
PyTree = Any


class NiSoapPerParamState(NamedTuple):
    m: Array
    v: Array
    L: Array
    R: Array
    QL: Array
    QR: Array
    eL: Array  # diag(QL^T L QL), updated at same frequency as QL/QR
    eR: Array  # diag(QR^T R QR), updated at same frequency as QL/QR
    step: Array


class NiSoapState(NamedTuple):
    per_param: PyTree
    rng_key: Array  # advanced each step to provide fresh noise


def _is_ni_soap_state(x: Any) -> bool:
    return isinstance(x, NiSoapPerParamState)


def _ni_soap_is_matrix(p: Array) -> bool:
    return p.ndim == 2 and p.shape[0] > 1 and p.shape[1] > 1


def _should_use_ni_soap(path, p: Array) -> bool:
    return should_use_matrix_preconditioner(path, p)


def _has_kronecker_precond(L: Array, R: Array, QL: Array, QR: Array) -> bool:
    return (L.shape[0] > 1) and (R.shape[0] > 1) and (QL.shape[0] > 1) and (QR.shape[0] > 1)


def _init_per_param(p: Array, *, use_soap: bool) -> NiSoapPerParamState:
    p_arr = jnp.asarray(p)
    m0 = jnp.zeros_like(p_arr)
    v0 = jnp.zeros_like(p_arr)
    if _ni_soap_is_matrix(p_arr) and use_soap:
        rows, cols = p_arr.shape
        return NiSoapPerParamState(
            m=m0,
            v=v0,
            L=jnp.zeros((rows, rows), dtype=jnp.float32),
            R=jnp.zeros((cols, cols), dtype=jnp.float32),
            QL=jnp.eye(rows, dtype=jnp.float32),
            QR=jnp.eye(cols, dtype=jnp.float32),
            eL=jnp.zeros(rows, dtype=jnp.float32),
            eR=jnp.zeros(cols, dtype=jnp.float32),
            step=jnp.array(-1, dtype=jnp.int32),
        )
    one = jnp.eye(1, dtype=jnp.float32)
    return NiSoapPerParamState(
        m=m0, v=v0,
        L=one, R=one, QL=one, QR=one,
        eL=jnp.zeros(1, dtype=jnp.float32),
        eR=jnp.zeros(1, dtype=jnp.float32),
        step=jnp.array(0, dtype=jnp.int32),
    )


def _kron_second_moments(g2d: Array) -> tuple[Array, Array]:
    g = jnp.asarray(g2d, dtype=jnp.float32)
    return g @ g.T, g.T @ g


def _project_2d(g2d: Array, QL: Array, QR: Array) -> Array:
    return QL.T @ g2d @ QR


def _project_back_2d(g2d: Array, QL: Array, QR: Array) -> Array:
    return QL @ g2d @ QR.T


def _eigh_desc(mat: Array) -> Array:
    mat32 = jnp.asarray(mat, dtype=jnp.float32)
    mat_sym = 0.5 * (mat32 + mat32.T)
    eye = jnp.eye(mat_sym.shape[0], dtype=mat_sym.dtype)
    _, q = jnp.linalg.eigh(mat_sym + 1e-30 * eye)
    return jnp.flip(q, axis=1)


def _refresh_qr_and_reindex_v(
    L: Array, R: Array, QL: Array, QR: Array, v: Array
) -> tuple[Array, Array, Array, Array, Array]:
    L32 = jnp.asarray(L, dtype=jnp.float32)
    R32 = jnp.asarray(R, dtype=jnp.float32)
    QL32 = jnp.asarray(QL, dtype=jnp.float32)
    QR32 = jnp.asarray(QR, dtype=jnp.float32)

    est_l = jnp.diag(QL32.T @ L32 @ QL32)
    idx_l = jnp.argsort(-est_l)
    v_new = jnp.take(v, idx_l, axis=0)
    QL_sorted = jnp.take(QL32, idx_l, axis=1)
    QL_new, _ = jnp.linalg.qr(L32 @ QL_sorted, mode="reduced")
    eL_new = jnp.diag(QL_new.T @ L32 @ QL_new)

    est_r = jnp.diag(QR32.T @ R32 @ QR32)
    idx_r = jnp.argsort(-est_r)
    v_new = jnp.take(v_new, idx_r, axis=1)
    QR_sorted = jnp.take(QR32, idx_r, axis=1)
    QR_new, _ = jnp.linalg.qr(R32 @ QR_sorted, mode="reduced")
    eR_new = jnp.diag(QR_new.T @ R32 @ QR_new)

    return QL_new, QR_new, eL_new, eR_new, v_new


def _kron_sqrt_noise(Z: Array, eL: Array, eR: Array, QL: Array, QR: Array) -> Array:
    """Compute L^{1/2} Z R^{1/2} using cached Kronecker factor eigenvalues/vectors.

    eL and eR are diag(QL^T L QL) and diag(QR^T R QR), updated at the same
    frequency as QL/QR so noise and gradient preconditioner stay in sync.
    """
    QL32 = jnp.asarray(QL, dtype=jnp.float32)
    QR32 = jnp.asarray(QR, dtype=jnp.float32)
    Z32 = jnp.asarray(Z, dtype=jnp.float32)

    eL32 = jnp.nan_to_num(jnp.asarray(eL, dtype=jnp.float32), nan=0.0, posinf=0.0, neginf=0.0)
    eR32 = jnp.nan_to_num(jnp.asarray(eR, dtype=jnp.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sqrt_eL = jnp.sqrt(jnp.maximum(eL32, 0.0))
    sqrt_eR = jnp.sqrt(jnp.maximum(eR32, 0.0))

    Z_rot = QL32.T @ Z32 @ QR32
    Z_scaled = sqrt_eL[:, None] * Z_rot * sqrt_eR[None, :]
    return QL32 @ Z_scaled @ QR32.T


def scale_by_ni_soap(
    b1: float = 0.95,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    precondition_frequency: int = 10,
    shampoo_beta2: Optional[float] = None,
    log_skipped: bool = False,
    correct_bias: bool = True,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    alpha: float = 1.0,
    seed: int = 0,
) -> optax.GradientTransformation:
    """NI-SOAP core transform (without learning-rate scaling).

    For 2D Kronecker-preconditioned parameters, after computing the SOAP update n,
    subtracts noise_coeff * L^{1/2} Z R^{1/2} so that after the downstream
    scale_by_learning_rate(-lr) step the full parameter update is:

        θ_{t+1} = θ_t − η P⁻¹ g̃_t + α √(η/m) · L^{1/2} Z R^{1/2}

    The noise_coeff = α / sqrt(η * m) cancels the -η factor to produce the
    energy-matched SGD noise term (Zhu et al. trace constraint at fixed energy).
    For AdamW-fallback parameters, isotropic noise is injected with the same scale.
    """
    shampoo_beta2 = b2 if shampoo_beta2 is None else shampoo_beta2
    # Pre-compute as a plain Python float so it is a compile-time constant in JIT.
    noise_coeff: float = float(alpha) / math.sqrt(float(learning_rate) * float(batch_size))

    def init_fn(params: PyTree) -> NiSoapState:
        per_param = jtu.tree_map_with_path(
            lambda path, p: _init_per_param(p, use_soap=_should_use_ni_soap(path, p)),
            params,
        )
        if log_skipped:
            skipped: list[str] = []

            def record(path, s):
                if isinstance(s, NiSoapPerParamState) and (
                    not _has_kronecker_precond(s.L, s.R, s.QL, s.QR)
                ):
                    skipped.append(path_to_name(path))
                return s

            jtu.tree_map_with_path(record, per_param)
            print(f"NI-SOAP: {len(skipped)} params routed to AdamW fallback: {skipped}")

        return NiSoapState(
            per_param=per_param,
            rng_key=jax.random.PRNGKey(seed),
        )

    def update_fn(
        grads: PyTree,
        state: NiSoapState,
        params: Optional[PyTree] = None,
    ) -> tuple[PyTree, NiSoapState]:
        flat_grads, treedef = jtu.tree_flatten(grads)
        flat_states, treedef2 = jtu.tree_flatten(
            state.per_param, is_leaf=_is_ni_soap_state,
        )
        if treedef != treedef2:
            raise ValueError("NI-SOAP state and grads PyTrees do not match structure.")

        if params is None:
            flat_params = [None] * len(flat_grads)
        else:
            flat_params, params_treedef = jtu.tree_flatten(params)
            if params_treedef != treedef:
                raise ValueError("NI-SOAP params and grads PyTrees do not match structure.")

        # Advance global key once per step; per-param keys are folded from step_key.
        step_key, new_rng_key = jax.random.split(state.rng_key)

        flat_updates: list[Array] = []
        flat_new_states: list[NiSoapPerParamState] = []

        for i, (g, p, s) in enumerate(zip(flat_grads, flat_params, flat_states)):
            m, v, L, R, QL, QR, eL, eR, step = s
            if g is None:
                flat_updates.append(jnp.zeros_like(m))
                flat_new_states.append(s)
                continue

            g_arr = jnp.asarray(g)
            p_arr = None if p is None else jnp.asarray(p)
            use_weight_decay = (p_arr is not None) and (weight_decay != 0.0)

            if g_arr.shape != m.shape:
                reference = p_arr if p_arr is not None else g_arr
                s = _init_per_param(
                    reference,
                    use_soap=_has_kronecker_precond(L, R, QL, QR),
                )
                m, v, L, R, QL, QR, eL, eR, step = s
                g_arr = jnp.asarray(g)

            if g_arr.ndim == 2 and _has_kronecker_precond(L, R, QL, QR):
                # Sample per-param noise outside cond for consistent operand structure.
                param_key = jax.random.fold_in(step_key, i)
                Z = jax.random.normal(param_key, shape=g_arr.shape, dtype=jnp.float32)

                def init_preconditioner(op):
                    m_i, v_i, L_i, R_i, _QL_i, _QR_i, _eL_i, _eR_i, _step_i, g_i, _Z = op
                    L_upd, R_upd = _kron_second_moments(g_i)
                    L_new = shampoo_beta2 * L_i + (1.0 - shampoo_beta2) * L_upd
                    R_new = shampoo_beta2 * R_i + (1.0 - shampoo_beta2) * R_upd
                    QL_new = _eigh_desc(L_new)
                    QR_new = _eigh_desc(R_new)
                    eL_new = jnp.diag(QL_new.T @ L_new @ QL_new)
                    eR_new = jnp.diag(QR_new.T @ R_new @ QR_new)
                    return (
                        jnp.zeros_like(g_i),
                        NiSoapPerParamState(
                            m=m_i, v=v_i,
                            L=L_new, R=R_new,
                            QL=QL_new, QR=QR_new,
                            eL=eL_new, eR=eR_new,
                            step=jnp.array(0, dtype=jnp.int32),
                        ),
                    )

                def soap_update(op):
                    m_i, v_i, L_i, R_i, QL_i, QR_i, eL_i, eR_i, step_i, g_i, Z_i = op
                    step_new = step_i + jnp.array(1, dtype=jnp.int32)

                    g_rot = _project_2d(g_i, QL_i, QR_i)
                    m_new = (b1 * m_i + (1.0 - b1) * g_rot).astype(m_i.dtype)
                    v_new = (b2 * v_i + (1.0 - b2) * (g_rot * g_rot)).astype(v_i.dtype)

                    if correct_bias:
                        bc1 = 1.0 - jnp.power(b1, step_new.astype(jnp.float32))
                        bc2 = 1.0 - jnp.power(b2, step_new.astype(jnp.float32))
                        m_use = m_new / bc1
                        v_use = v_new / bc2
                    else:
                        m_use = m_new
                        v_use = v_new

                    n_rot = m_use / (jnp.sqrt(v_use) + eps)
                    n = _project_back_2d(n_rot, QL_i, QR_i)

                    # NI: subtract from n so that after -lr scaling the noise ADDS
                    # to the parameter: Δθ_noise = +α √(η/m) · L^{1/2} Z R^{1/2}
                    # Uses cached eL_i/eR_i — updated at same rate as QL_i/QR_i.
                    if alpha != 0.0:
                        kron_noise = _kron_sqrt_noise(Z_i, eL_i, eR_i, QL_i, QR_i)
                        kron_noise = jnp.nan_to_num(kron_noise, nan=0.0, posinf=0.0, neginf=0.0)
                        n = n - noise_coeff * kron_noise

                    if use_weight_decay:
                        n = n + weight_decay * p_arr
                    n = n.astype(g_i.dtype)

                    m_orig = _project_back_2d(m_new, QL_i, QR_i)
                    L_upd, R_upd = _kron_second_moments(g_i)
                    L_new = shampoo_beta2 * L_i + (1.0 - shampoo_beta2) * L_upd
                    R_new = shampoo_beta2 * R_i + (1.0 - shampoo_beta2) * R_upd
                    L_new = jnp.nan_to_num(L_new, nan=0.0, posinf=0.0, neginf=0.0)
                    R_new = jnp.nan_to_num(R_new, nan=0.0, posinf=0.0, neginf=0.0)

                    do_qr = (precondition_frequency > 0) & (
                        (step_new % precondition_frequency) == 0
                    )

                    def refresh(args):
                        L_q, R_q, QL_q, QR_q, eL_q, eR_q, v_q = args
                        QL_r, QR_r, eL_r, eR_r, v_r = _refresh_qr_and_reindex_v(
                            L_q, R_q, QL_q, QR_q, v_q
                        )
                        return QL_r, QR_r, eL_r, eR_r, v_r

                    def keep(args):
                        _L_q, _R_q, QL_q, QR_q, eL_q, eR_q, v_q = args
                        return QL_q, QR_q, eL_q, eR_q, v_q

                    QL_new, QR_new, eL_new, eR_new, v_aligned = jax.lax.cond(
                        do_qr, refresh, keep,
                        operand=(L_new, R_new, QL_i, QR_i, eL_i, eR_i, v_new),
                    )
                    v_aligned = v_aligned.astype(v_i.dtype)
                    m_reproj = _project_2d(m_orig, QL_new, QR_new).astype(m_i.dtype)

                    new_s = NiSoapPerParamState(
                        m=m_reproj, v=v_aligned,
                        L=L_new, R=R_new,
                        QL=QL_new, QR=QR_new,
                        eL=eL_new, eR=eR_new,
                        step=step_new,
                    )
                    return n, new_s

                update, new_s = jax.lax.cond(
                    step < 0,
                    init_preconditioner,
                    soap_update,
                    operand=(m, v, L, R, QL, QR, eL, eR, step, g_arr, Z),
                )
                flat_updates.append(update)
                flat_new_states.append(new_s)
                continue

            # AdamW fallback with isotropic noise injection.
            step_new = step + jnp.array(1, dtype=jnp.int32)
            m_new = (b1 * m + (1.0 - b1) * g_arr).astype(m.dtype)
            v_new = (b2 * v + (1.0 - b2) * (g_arr * g_arr)).astype(v.dtype)
            if correct_bias:
                bc1 = 1.0 - jnp.power(b1, step_new.astype(jnp.float32))
                bc2 = 1.0 - jnp.power(b2, step_new.astype(jnp.float32))
                m_hat = m_new / bc1
                v_hat = v_new / bc2
            else:
                m_hat = m_new
                v_hat = v_new
            n = m_hat / (jnp.sqrt(v_hat) + eps)

            if alpha != 0.0:
                param_key = jax.random.fold_in(step_key, i)
                xi = jax.random.normal(param_key, shape=g_arr.shape, dtype=jnp.float32)
                n = n - noise_coeff * xi.astype(n.dtype)

            if use_weight_decay:
                n = n + weight_decay * p_arr
            n = n.astype(g_arr.dtype)

            new_s = NiSoapPerParamState(
                m=m_new, v=v_new,
                L=L, R=R, QL=QL, QR=QR,
                eL=s.eL, eR=s.eR,
                step=step_new,
            )
            flat_updates.append(n)
            flat_new_states.append(new_s)

        updates = jtu.tree_unflatten(treedef, flat_updates)
        new_per_param = jtu.tree_unflatten(treedef, flat_new_states)
        return updates, NiSoapState(per_param=new_per_param, rng_key=new_rng_key)

    return optax.GradientTransformation(init_fn, update_fn)


def ni_soap(
    learning_rate: float,
    b1: float = 0.95,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    precondition_frequency: int = 10,
    shampoo_beta2: Optional[float] = None,
    log_skipped: bool = False,
    correct_bias: bool = True,
    batch_size: int = 128,
    alpha: float = 1.0,
    seed: int = 0,
) -> optax.GradientTransformation:
    """NI-SOAP: SOAP with energy-matched Kronecker noise injection.

    Injects noise α √(η/m) · L^{1/2} Z R^{1/2} after each SOAP step to
    restore the SGD noise energy lost through Kronecker preconditioning
    (Zhu et al. trace constraint). alpha=1 is the energy-matched baseline;
    sweep alpha ∈ {0.1, 0.3, 1.0, 3.0} to tune strength.

    batch_size (m): effective tokens or sequences per gradient step.
    alpha: noise strength relative to energy-matched SGD baseline.
    seed: initial PRNG seed for reproducibility.
    """
    return optax.chain(
        scale_by_ni_soap(
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            shampoo_beta2=shampoo_beta2,
            log_skipped=log_skipped,
            correct_bias=correct_bias,
            learning_rate=learning_rate,
            batch_size=batch_size,
            alpha=alpha,
            seed=seed,
        ),
        optax.scale_by_learning_rate(learning_rate),
    )
