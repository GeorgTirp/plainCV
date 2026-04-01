from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax

from .matrix_routing import path_to_name, should_use_matrix_preconditioner

Array = jax.Array
PyTree = Any

class SoapPerParamState(NamedTuple):
    m: Array
    v: Array
    L: Array
    R: Array
    QL: Array
    QR: Array
    # For SOAP params:
    #   step = -1 means "preconditioner not initialized yet" (first-step skip).
    #   step >= 0 means normal Adam-step count for bias correction.
    # For AdamW fallback params:
    #   step >= 0 always.
    step: Array


class SoapState(NamedTuple):
    per_param: PyTree  # PyTree[SoapPerParamState]


def _is_soap_state(x: Any) -> bool:
    return isinstance(x, SoapPerParamState)


def _is_soap_matrix(p: Array) -> bool:
    return (
        p.ndim == 2
        and p.shape[0] > 1
        and p.shape[1] > 1
    )


def _should_use_soap(path, p: Array) -> bool:
    return should_use_matrix_preconditioner(path, p)


def _has_soap_preconditioner_state(L: Array, R: Array, QL: Array, QR: Array) -> bool:
    return (L.shape[0] > 1) and (R.shape[0] > 1) and (QL.shape[0] > 1) and (QR.shape[0] > 1)


def _init_per_param(p: Array, *, use_soap: bool) -> SoapPerParamState:
    p_arr = jnp.asarray(p)
    m0 = jnp.zeros_like(p_arr)
    v0 = jnp.zeros_like(p_arr)

    if _is_soap_matrix(p_arr) and use_soap:
        rows, cols = p_arr.shape
        eye_rows = jnp.eye(rows, dtype=jnp.float32)
        eye_cols = jnp.eye(cols, dtype=jnp.float32)
        zero_rows = jnp.zeros((rows, rows), dtype=jnp.float32)
        zero_cols = jnp.zeros((cols, cols), dtype=jnp.float32)
        return SoapPerParamState(
            m=m0,
            v=v0,
            L=zero_rows,
            R=zero_cols,
            QL=eye_rows,
            QR=eye_cols,
            step=jnp.array(-1, dtype=jnp.int32),
        )

    one = jnp.eye(1, dtype=jnp.float32)
    return SoapPerParamState(
        m=m0,
        v=v0,
        L=one,
        R=one,
        QL=one,
        QR=one,
        step=jnp.array(0, dtype=jnp.int32),
    )


def _kronecker_second_moments(g2d: Array) -> tuple[Array, Array]:
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
    L: Array,
    R: Array,
    QL: Array,
    QR: Array,
    v: Array,
) -> tuple[Array, Array, Array]:
    """SOAP-style QR refresh with exp_avg_sq axis reordering."""
    L32 = jnp.asarray(L, dtype=jnp.float32)
    R32 = jnp.asarray(R, dtype=jnp.float32)
    QL32 = jnp.asarray(QL, dtype=jnp.float32)
    QR32 = jnp.asarray(QR, dtype=jnp.float32)

    est_eig_l = jnp.diag(QL32.T @ L32 @ QL32)
    sort_idx_l = jnp.argsort(-est_eig_l)
    v_new = jnp.take(v, sort_idx_l, axis=0)
    QL_sorted = jnp.take(QL32, sort_idx_l, axis=1)
    QL_new, _ = jnp.linalg.qr(L32 @ QL_sorted, mode="reduced")

    est_eig_r = jnp.diag(QR32.T @ R32 @ QR32)
    sort_idx_r = jnp.argsort(-est_eig_r)
    v_new = jnp.take(v_new, sort_idx_r, axis=1)
    QR_sorted = jnp.take(QR32, sort_idx_r, axis=1)
    QR_new, _ = jnp.linalg.qr(R32 @ QR_sorted, mode="reduced")

    return QL_new, QR_new, v_new


def scale_by_soap(
    b1: float = 0.95,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    precondition_frequency: int = 10,
    shampoo_beta2: Optional[float] = None,
    log_skipped: bool = False,
    correct_bias: bool = True,
) -> optax.GradientTransformation:
    """SOAP on selected 2D non-embedding kernels with AdamW fallback elsewhere."""
    shampoo_beta2 = b2 if shampoo_beta2 is None else shampoo_beta2

    def init_fn(params: PyTree) -> SoapState:
        per_param = jtu.tree_map_with_path(
            lambda path, p: _init_per_param(
                p,
                use_soap=_should_use_soap(path, p),
            ),
            params,
        )
        if log_skipped:
            skipped = []

            def record(path, s):
                if isinstance(s, SoapPerParamState) and (
                    not _has_soap_preconditioner_state(s.L, s.R, s.QL, s.QR)
                ):
                    name = path_to_name(path)
                    skipped.append(name)
                return s

            jtu.tree_map_with_path(record, per_param)
            print(f"SOAP: routed {len(skipped)} params to AdamW fallback: {skipped}")

        return SoapState(per_param=per_param)

    def update_fn(
        grads: PyTree,
        state: SoapState,
        params: Optional[PyTree] = None,
    ) -> tuple[PyTree, SoapState]:
        flat_grads, treedef = jtu.tree_flatten(grads)
        flat_states, treedef2 = jtu.tree_flatten(
            state.per_param,
            is_leaf=_is_soap_state,
        )
        if treedef != treedef2:
            raise ValueError("SOAP state and grads PyTrees do not match structure.")

        if params is None:
            flat_params = [None] * len(flat_grads)
        else:
            flat_params, params_treedef = jtu.tree_flatten(params)
            if params_treedef != treedef:
                raise ValueError("SOAP params and grads PyTrees do not match structure.")

        flat_updates: list[Array] = []
        flat_new_states: list[SoapPerParamState] = []

        for g, p, s in zip(flat_grads, flat_params, flat_states):
            m, v, L, R, QL, QR, step = s
            if g is None:
                flat_updates.append(jnp.zeros_like(m))
                flat_new_states.append(s)
                continue

            g_arr = jnp.asarray(g)
            p_arr = None if p is None else jnp.asarray(p)
            use_weight_decay = (p_arr is not None) and (weight_decay != 0.0)

            # Reinitialize if shape changed.
            if g_arr.shape != m.shape:
                reference = p_arr if p_arr is not None else g_arr
                s = _init_per_param(
                    reference,
                    use_soap=_has_soap_preconditioner_state(L, R, QL, QR),
                )
                m, v, L, R, QL, QR, step = s
                g_arr = jnp.asarray(g)

            if g_arr.ndim == 2 and _has_soap_preconditioner_state(L, R, QL, QR):
                def init_preconditioner(op):
                    m_i, v_i, L_i, R_i, _QL_i, _QR_i, _step_i, g_i = op
                    L_update, R_update = _kronecker_second_moments(g_i)
                    L_new = shampoo_beta2 * L_i + (1.0 - shampoo_beta2) * L_update
                    R_new = shampoo_beta2 * R_i + (1.0 - shampoo_beta2) * R_update
                    QL_new = _eigh_desc(L_new)
                    QR_new = _eigh_desc(R_new)
                    return (
                        jnp.zeros_like(g_i),
                        SoapPerParamState(
                            m=m_i,
                            v=v_i,
                            L=L_new,
                            R=R_new,
                            QL=QL_new,
                            QR=QR_new,
                            step=jnp.array(0, dtype=jnp.int32),
                        ),
                    )

                def soap_update(op):
                    m_i, v_i, L_i, R_i, QL_i, QR_i, step_i, g_i = op

                    step_new = step_i + jnp.array(1, dtype=jnp.int32)
                    g_rot = _project_2d(g_i, QL_i, QR_i)
                    m_new = (b1 * m_i + (1.0 - b1) * g_rot).astype(m_i.dtype)
                    v_new = (b2 * v_i + (1.0 - b2) * (g_rot * g_rot)).astype(v_i.dtype)

                    if correct_bias:
                        bias_correction1 = 1.0 - jnp.power(b1, step_new.astype(jnp.float32))
                        bias_correction2 = 1.0 - jnp.power(b2, step_new.astype(jnp.float32))
                        m_use = m_new / bias_correction1
                        v_use = v_new / bias_correction2
                    else:
                        m_use = m_new
                        v_use = v_new

                    n_rot = m_use / (jnp.sqrt(v_use) + eps)
                    n = _project_back_2d(n_rot, QL_i, QR_i)
                    if use_weight_decay:
                        n = n + weight_decay * p_arr
                    n = n.astype(g_i.dtype)

                    # Preconditioner update happens after the gradient step.
                    m_orig = _project_back_2d(m_new, QL_i, QR_i)
                    L_update, R_update = _kronecker_second_moments(g_i)
                    L_new = shampoo_beta2 * L_i + (1.0 - shampoo_beta2) * L_update
                    R_new = shampoo_beta2 * R_i + (1.0 - shampoo_beta2) * R_update

                    do_qr = (precondition_frequency > 0) & (
                        (step_new % precondition_frequency) == 0
                    )

                    def refresh(args):
                        L_q, R_q, QL_q, QR_q, v_q = args
                        return _refresh_qr_and_reindex_v(L_q, R_q, QL_q, QR_q, v_q)

                    def keep(args):
                        _L_q, _R_q, QL_q, QR_q, v_q = args
                        return QL_q, QR_q, v_q

                    QL_new, QR_new, v_aligned = jax.lax.cond(
                        do_qr,
                        refresh,
                        keep,
                        operand=(L_new, R_new, QL_i, QR_i, v_new),
                    )
                    v_aligned = v_aligned.astype(v_i.dtype)

                    m_reprojected = _project_2d(m_orig, QL_new, QR_new).astype(m_i.dtype)
                    new_s = SoapPerParamState(
                        m=m_reprojected,
                        v=v_aligned,
                        L=L_new,
                        R=R_new,
                        QL=QL_new,
                        QR=QR_new,
                        step=step_new,
                    )
                    return n, new_s

                update, new_s = jax.lax.cond(
                    step < 0,
                    init_preconditioner,
                    soap_update,
                    operand=(m, v, L, R, QL, QR, step, g_arr),
                )
                flat_updates.append(update)
                flat_new_states.append(new_s)
                continue

            # AdamW fallback for all non-SOAP-routed parameters.
            step_new = step + jnp.array(1, dtype=jnp.int32)
            m_new = (b1 * m + (1.0 - b1) * g_arr).astype(m.dtype)
            v_new = (b2 * v + (1.0 - b2) * (g_arr * g_arr)).astype(v.dtype)
            if correct_bias:
                bias_correction1 = 1.0 - jnp.power(b1, step_new.astype(jnp.float32))
                bias_correction2 = 1.0 - jnp.power(b2, step_new.astype(jnp.float32))
                m_hat = m_new / bias_correction1
                v_hat = v_new / bias_correction2
            else:
                m_hat = m_new
                v_hat = v_new
            n = m_hat / (jnp.sqrt(v_hat) + eps)
            if use_weight_decay:
                n = n + weight_decay * p_arr
            n = n.astype(g_arr.dtype)

            new_s = SoapPerParamState(
                m=m_new,
                v=v_new,
                L=L,
                R=R,
                QL=QL,
                QR=QR,
                step=step_new,
            )
            flat_updates.append(n)
            flat_new_states.append(new_s)

        updates = jtu.tree_unflatten(treedef, flat_updates)
        new_per_param = jtu.tree_unflatten(treedef, flat_new_states)
        return updates, SoapState(per_param=new_per_param)

    return optax.GradientTransformation(init_fn, update_fn)


def soap(
    learning_rate: float,
    b1: float = 0.95,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    precondition_frequency: int = 10,
    shampoo_beta2: Optional[float] = None,
    log_skipped: bool = False,
    correct_bias: bool = True,
) -> optax.GradientTransformation:
    return optax.chain(
        scale_by_soap(
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            shampoo_beta2=shampoo_beta2,
            log_skipped=log_skipped,
            correct_bias=correct_bias,
        ),
        optax.scale_by_learning_rate(learning_rate),
    )
