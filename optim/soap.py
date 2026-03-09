from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax

Array = jax.Array
PyTree = Any

# Only run full SOAP on reasonably-sized 2D matrices.
MAX_DIM = 2048


class SoapPerParamState(NamedTuple):
    """Per-parameter state.

    For 2D matrix params (`use_soap=True`), SOAP uses (m, v, L, R, QL, QR).
    For non-2D params (`use_soap=False`), we use AdamW moments (m, v) and keep
    (L, R, QL, QR) as 1x1 placeholders.
    """

    m: Array
    v: Array
    L: Array
    R: Array
    QL: Array
    QR: Array
    use_soap: bool


class SoapState(NamedTuple):
    count: Array
    per_param: PyTree  # PyTree[SoapPerParamState]


def _is_soap_state(x: Any) -> bool:
    return isinstance(x, SoapPerParamState)


def _is_soap_matrix(p: Array) -> bool:
    return (
        p.ndim == 2
        and p.shape[0] > 1
        and p.shape[1] > 1
        and p.shape[0] <= MAX_DIM
        and p.shape[1] <= MAX_DIM
    )


def _init_per_param(p: Array) -> SoapPerParamState:
    p_arr = jnp.asarray(p)
    m0 = jnp.zeros_like(p_arr)
    v0 = jnp.zeros_like(p_arr)

    if _is_soap_matrix(p_arr):
        rows, cols = p_arr.shape
        eye_rows = jnp.eye(rows, dtype=p_arr.dtype)
        eye_cols = jnp.eye(cols, dtype=p_arr.dtype)
        return SoapPerParamState(
            m=m0,
            v=v0,
            L=eye_rows,
            R=eye_cols,
            QL=eye_rows,
            QR=eye_cols,
            use_soap=True,
        )

    one = jnp.eye(1, dtype=p_arr.dtype)
    return SoapPerParamState(
        m=m0,
        v=v0,
        L=one,
        R=one,
        QL=one,
        QR=one,
        use_soap=False,
    )


def _kronecker_second_moments(g2d: Array) -> tuple[Array, Array]:
    return g2d @ g2d.T, g2d.T @ g2d


def scale_by_soap(
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    precondition_frequency: int = 10,
    shampoo_beta2: Optional[float] = None,
    log_skipped: bool = False,
) -> optax.GradientTransformation:
    """Muon-style routing:
    - 2D matrices: SOAP
    - everything else: AdamW
    """
    shampoo_beta2 = b2 if shampoo_beta2 is None else shampoo_beta2

    def init_fn(params: PyTree) -> SoapState:
        per_param = jtu.tree_map(_init_per_param, params)
        if log_skipped:
            skipped = []

            def record(path, s):
                if isinstance(s, SoapPerParamState) and (not bool(s.use_soap)):
                    name = "/".join(str(k) for k in path)
                    skipped.append(name)
                return s

            jtu.tree_map_with_path(record, per_param)
            print(f"SOAP: routed {len(skipped)} params to AdamW fallback: {skipped}")

        return SoapState(
            count=jnp.zeros([], dtype=jnp.int32),
            per_param=per_param,
        )

    def update_fn(
        grads: PyTree,
        state: SoapState,
        params: Optional[PyTree] = None,
    ) -> tuple[PyTree, SoapState]:
        count = state.count + jnp.array(1, dtype=jnp.int32)

        b1_t = jnp.power(b1, count.astype(jnp.float32))
        b2_t = jnp.power(b2, count.astype(jnp.float32))
        m_bc_den = 1.0 - b1_t
        v_bc_den = 1.0 - b2_t

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
            m, v, L, R, QL, QR, use_soap = s
            g_arr = jnp.zeros_like(m) if g is None else jnp.asarray(g)
            p_arr = None if p is None else jnp.asarray(p)

            # Reinitialize if shape changed.
            if g_arr.shape != m.shape:
                reference = p_arr if p_arr is not None else g_arr
                s = _init_per_param(reference)
                m, v, L, R, QL, QR, use_soap = s
                g_arr = jnp.zeros_like(m) if g is None else jnp.asarray(g)

            if bool(use_soap):
                # If a matrix param no longer looks valid for SOAP, fall back to AdamW.
                if (g_arr.ndim != 2) or (
                    QL.shape != (g_arr.shape[0], g_arr.shape[0])
                ) or (QR.shape != (g_arr.shape[1], g_arr.shape[1])):
                    use_soap = False

            if bool(use_soap):
                g2d = g_arr
                rows, cols = g2d.shape

                m_new = (1.0 - b1) * g2d + b1 * m
                g_rot = QL.T @ g2d @ QR
                m_rot = QL.T @ m_new @ QR
                v_new = (1.0 - b2) * (g_rot * g_rot) + b2 * v

                m_hat = m_rot / m_bc_den
                v_hat = v_new / v_bc_den
                n_rot = m_hat / (jnp.sqrt(v_hat) + eps)
                n = QL @ n_rot @ QR.T

                L_update, R_update = _kronecker_second_moments(g2d)
                L_new = shampoo_beta2 * L + (1.0 - shampoo_beta2) * L_update
                R_new = shampoo_beta2 * R + (1.0 - shampoo_beta2) * R_update

                def recompute_eig(LRQLQR):
                    L_in, R_in, _, _ = LRQLQR
                    L_sym = 0.5 * (L_in + L_in.T)
                    R_sym = 0.5 * (R_in + R_in.T)
                    _, QL_new = jnp.linalg.eigh(L_sym)
                    _, QR_new = jnp.linalg.eigh(R_sym)
                    return L_in, R_in, QL_new, QR_new

                def keep_eig(LRQLQR):
                    return LRQLQR

                do_eig = (precondition_frequency > 0) & (
                    (count % precondition_frequency) == 0
                )
                L_fin, R_fin, QL_fin, QR_fin = jax.lax.cond(
                    do_eig,
                    recompute_eig,
                    keep_eig,
                    operand=(L_new, R_new, QL, QR),
                )

                if (p_arr is not None) and (weight_decay != 0.0):
                    n = n + weight_decay * p_arr

                new_s = SoapPerParamState(
                    m=m_new,
                    v=v_new,
                    L=L_fin,
                    R=R_fin,
                    QL=QL_fin,
                    QR=QR_fin,
                    use_soap=True,
                )
                flat_updates.append(n)
                flat_new_states.append(new_s)
                continue

            # AdamW fallback for all non-2D (and invalid) parameters.
            m_new = (1.0 - b1) * g_arr + b1 * m
            v_new = (1.0 - b2) * (g_arr * g_arr) + b2 * v
            m_hat = m_new / m_bc_den
            v_hat = v_new / v_bc_den
            n = m_hat / (jnp.sqrt(v_hat) + eps)
            if (p_arr is not None) and (weight_decay != 0.0):
                n = n + weight_decay * p_arr

            new_s = SoapPerParamState(
                m=m_new,
                v=v_new,
                L=L,
                R=R,
                QL=QL,
                QR=QR,
                use_soap=False,
            )
            flat_updates.append(n)
            flat_new_states.append(new_s)

        updates = jtu.tree_unflatten(treedef, flat_updates)
        new_per_param = jtu.tree_unflatten(treedef, flat_new_states)
        return updates, SoapState(count=count, per_param=new_per_param)

    return optax.GradientTransformation(init_fn, update_fn)


def soap(
    learning_rate: float,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    precondition_frequency: int = 10,
    shampoo_beta2: Optional[float] = None,
    log_skipped: bool = False,
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
        ),
        optax.scale_by_learning_rate(learning_rate),
    )
