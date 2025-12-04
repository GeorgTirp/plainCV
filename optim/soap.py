from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax

Array = jax.Array
PyTree = Any

# Only run full SOAP on reasonably-sized 2D views
MAX_DIM = 2048


class SoapPerParamState(NamedTuple):
    """State for a single parameter tensor (stored on a 2D view)."""
    m: Array      # first moment, (rows, cols)
    v: Array      # second moment, (rows, cols) (in eigenbasis)
    L: Array      # Kronecker factor, (rows, rows)
    R: Array      # Kronecker factor, (cols, cols)
    QL: Array     # eigenvectors of L, (rows, rows)
    QR: Array     # eigenvectors of R, (cols, cols)
    use_soap: bool


class SoapState(NamedTuple):
    count: Array    # scalar int32
    per_param: PyTree  # PyTree[SoapPerParamState]


def _is_soap_state(x: Any) -> bool:
    return isinstance(x, SoapPerParamState)


def _reshape_to_2d(x: Array) -> Array:
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        # Bias / BN scale/bias -> treat as vector; SOAP will mark as too_big/use_soap=False
        return x.reshape(1, -1)
    if x.ndim == 2:
        # Typical dense weights: (in_features, out_features)
        return x
    if x.ndim == 4:
        # Conv kernels (Kh, Kw, Cin, Cout) -> rows=Cout, cols=Kh*Kw*Cin
        kh, kw, cin, cout = x.shape
        return jnp.reshape(x, (cout, kh * kw * cin))
    # Fallback: flatten leading dim, keep rest in columns
    return x.reshape(x.shape[0], -1)


def _init_per_param(p: Array) -> SoapPerParamState:
    p2d = _reshape_to_2d(p)
    rows, cols = p2d.shape
    zeros = jnp.zeros_like(p2d)

    too_big = (rows > MAX_DIM) or (cols > MAX_DIM)
    if too_big:
        one = jnp.eye(1, dtype=p.dtype)
        return SoapPerParamState(
            m=zeros,
            v=zeros,
            L=one,
            R=one,
            QL=one,
            QR=one,
            use_soap=False,
        )
    else:
        eye_rows = jnp.eye(rows, dtype=p.dtype)
        eye_cols = jnp.eye(cols, dtype=p.dtype)
        return SoapPerParamState(
            m=zeros,
            v=zeros,
            L=eye_rows,
            R=eye_cols,
            QL=eye_rows,
            QR=eye_cols,
            use_soap=True,
        )


def _kronecker_second_moments(g2d: Array) -> tuple[Array, Array]:
    return g2d @ g2d.T, g2d.T @ g2d


def scale_by_soap(
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    precondition_frequency: int = 10,
    shampoo_beta2: Optional[float] = None,
    log_skipped: bool = False,
) -> optax.GradientTransformation:
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
            print(
                f"SOAP: skipped {len(skipped)} params "
                f"(too large/degenerate): {skipped}"
            )
        return SoapState(
            count=jnp.zeros([], dtype=jnp.int32),
            per_param=per_param,
        )

    def update_fn(
        grads: PyTree,
        state: SoapState,
        params: Optional[PyTree] = None,
    ) -> tuple[PyTree, SoapState]:
        del params
        count = state.count + jnp.array(1, dtype=jnp.int32)

        b1_t = jnp.power(b1, count.astype(jnp.float32))
        b2_t = jnp.power(b2, count.astype(jnp.float32))
        m_bc_den = 1.0 - b1_t
        v_bc_den = 1.0 - b2_t

        flat_grads, treedef = jtu.tree_flatten(grads)
        flat_states, treedef2 = jtu.tree_flatten(
            state.per_param,
            is_leaf=_is_soap_state,  # keep SoapPerParamState as a leaf
        )
        if treedef != treedef2:
            raise ValueError("SOAP state and grads PyTrees do not match structure.")

        flat_updates: list[Array] = []
        flat_new_states: list[SoapPerParamState] = []

        for g, s in zip(flat_grads, flat_states):
            # For robustness: treat non-array or None as zero update.
            if g is None or not isinstance(g, jax.Array):
                # Keeps structure but makes sure update is an array
                g2d = s.m  # same 2D shape as this param’s state
                zero = jnp.zeros_like(g2d)
                flat_updates.append(zero.reshape(g2d.shape))
                flat_new_states.append(s)
                continue

            g2d = _reshape_to_2d(g)
            m, v, L, R, QL, QR, use_soap = s

            # If shapes drift (e.g., parameter resized), reinitialize per-param
            # state to match the current gradient shape to avoid matmul errors.
            shape_ok = (
                g2d.shape == m.shape
                and QL.shape[0] == g2d.shape[0]
                and QR.shape[0] == g2d.shape[1]
            )
            if not shape_ok:
                s = _init_per_param(g)
                m, v, L, R, QL, QR, use_soap = s
                g2d = _reshape_to_2d(g)

            rows, cols = g2d.shape

            def adam_branch(_):
                # -------- plain Adam fallback --------
                m_new = (1.0 - b1) * g2d + b1 * m
                v_new = (1.0 - b2) * (g2d * g2d) + b2 * v

                m_hat = m_new / m_bc_den
                v_hat = v_new / v_bc_den

                n2d = m_hat / (jnp.sqrt(v_hat) + eps)
                n = n2d.reshape(g.shape)

                new_s = SoapPerParamState(
                    m=m_new,
                    v=v_new,
                    L=L,
                    R=R,
                    QL=QL,
                    QR=QR,
                    use_soap=use_soap,
                )

                return n, new_s

            # Hard skip SOAP for vectors or degenerate shapes; reshape won't fix those.
            if (rows <= 1) or (cols <= 1):
                n, new_s = adam_branch(None)
                flat_updates.append(n)
                flat_new_states.append(new_s)
                continue

            # Final guard: eigen factors must exactly match current g2d shape.
            eig_shape_ok = (
                QL.shape[0] == rows
                and QL.shape[1] == rows
                and QR.shape[0] == cols
                and QR.shape[1] == cols
            )
            if not eig_shape_ok:
                n, new_s = adam_branch(None)
                flat_updates.append(n)
                flat_new_states.append(new_s)
                continue

            use_soap_pred = jnp.asarray(use_soap, dtype=bool)

            def soap_branch(_):
                # -------- SOAP branch --------
                m_new = (1.0 - b1) * g2d + b1 * m

                g_rot = QL.T @ g2d @ QR
                m_rot = QL.T @ m_new @ QR

                v_new = (1.0 - b2) * (g_rot * g_rot) + b2 * v

                m_hat = m_rot / m_bc_den
                v_hat = v_new / v_bc_den
                n_rot = m_hat / (jnp.sqrt(v_hat) + eps)

                n2d = QL @ n_rot @ QR.T

                L_update, R_update = _kronecker_second_moments(g2d)
                L_new = shampoo_beta2 * L + (1.0 - shampoo_beta2) * L_update
                R_new = shampoo_beta2 * R + (1.0 - shampoo_beta2) * R_update

                def recompute_eig(LRQLQR):
                    L_in, R_in, QL_in, QR_in = LRQLQR
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

                new_s = SoapPerParamState(
                    m=m_new,
                    v=v_new,
                    L=L_fin,
                    R=R_fin,
                    QL=QL_fin,
                    QR=QR_fin,
                    use_soap=use_soap,
                )

                n = n2d.reshape(g.shape)
                return n, new_s

            n, new_s = jax.lax.cond(
                use_soap_pred,
                soap_branch,
                adam_branch,
                operand=None,
            )
            flat_updates.append(n)
            flat_new_states.append(new_s)

        updates = jtu.tree_unflatten(treedef, flat_updates)
        new_per_param = jtu.tree_unflatten(treedef, flat_new_states)

        new_state = SoapState(
            count=count,
            per_param=new_per_param,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def soap(
    learning_rate: float,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,  # unused in this minimal version
    precondition_frequency: int = 10,
    shampoo_beta2: Optional[float] = None,
    log_skipped: bool = False,
) -> optax.GradientTransformation:
    """SOAP optimizer as an Optax alias (without decoupled weight decay)."""
    del weight_decay  # simplify for now; add back once Optax is patched

    return optax.chain(
        scale_by_soap(
            b1=b1,
            b2=b2,
            eps=eps,
            precondition_frequency=precondition_frequency,
            shampoo_beta2=shampoo_beta2,
            log_skipped=log_skipped,
        ),
        optax.scale_by_learning_rate(learning_rate),
    )
