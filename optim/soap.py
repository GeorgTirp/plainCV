# optim/soap.py
from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax

# Optional: if you have a Kronecker helper already, we reuse it.
# Otherwise we fall back to a local implementation.
try:
    from .kronecker import compute_kronecker_factors  # (G) -> (GG^T, G^T G)
except Exception:  # pragma: no cover
    compute_kronecker_factors = None  # type: ignore[assignment]

Array = jax.Array
PyTree = Any


class SoapPerParamState(NamedTuple):
    """State for a single parameter tensor, stored on a 2D view (rows, cols)."""

    # Adam first moment in original basis (same shape as reshaped param)
    m: Array              # (rows, cols)
    # Adam second moment in the eigenbasis of (L, R)
    v: Array              # (rows, cols)
    # Kronecker second-moment factors (Shampoo-style)
    L: Array              # (rows, rows)
    R: Array              # (cols, cols)
    # Eigenvectors of L and R (columns are eigenvectors)
    QL: Array             # (rows, rows)
    QR: Array             # (cols, cols)


class SoapState(NamedTuple):
    """Global SOAP optimizer state."""
    count: Array          # scalar step counter (int32)
    per_param: PyTree     # PyTree of SoapPerParamState with same structure as params


def _reshape_to_2d(x: Array) -> Array:
    """Reshape an arbitrary tensor to 2D (rows, cols).

    - Scalars -> (1, 1)
    - Vectors -> (N, 1)
    - Matrices -> (M, N)
    - Higher rank -> (shape[0], prod(shape[1:]))
    """
    if x.ndim == 0:
        return x.reshape(1, 1)
    else:
        return x.reshape(x.shape[0], -1)


def _init_per_param(p: Array) -> SoapPerParamState:
    """Initialize SOAP state for a single parameter tensor."""
    p2d = _reshape_to_2d(p)
    rows, cols = p2d.shape
    zeros = jnp.zeros_like(p2d)
    eye_rows = jnp.eye(rows, dtype=p.dtype)
    eye_cols = jnp.eye(cols, dtype=p.dtype)
    return SoapPerParamState(
        m=zeros,
        v=zeros,
        L=eye_rows,
        R=eye_cols,
        QL=eye_rows,
        QR=eye_cols,
    )


def _kronecker_second_moments(g2d: Array) -> tuple[Array, Array]:
    """Compute per-layer Kronecker factors GG^T and G^T G.

    If you already have a helper in kronecker.py called `compute_kronecker_factors`,
    this will call that. Otherwise we just do GG^T and G^T G here.
    """
    if compute_kronecker_factors is not None:
        return compute_kronecker_factors(g2d)
    return g2d @ g2d.T, g2d.T @ g2d


def scale_by_soap(
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    precondition_frequency: int = 10,
    shampoo_beta2: Optional[float] = None,
) -> optax.GradientTransformation:
    """Gradient transformation for SOAP (Adam in Shampoo eigenbasis).

    Roughly matches Algorithm 3 of the SOAP paper, but implemented per-parameter
    and in a simple Optax-friendly way. :contentReference[oaicite:0]{index=0}

    Args:
      b1: Adam first-moment decay (β₁).
      b2: Adam second-moment decay for v in the rotated basis (β₂).
      eps: Small epsilon in the Adam denominator.
      precondition_frequency: How often to recompute eigenvectors of (L, R).
        If <= 0, eigenvectors are never updated after initialization.
      shampoo_beta2: Decay rate for Kronecker second moments (L, R).
        If None, we reuse `b2`.

    Returns:
      An optax.GradientTransformation that rescales gradients à la SOAP.
    """
    shampoo_beta2 = b2 if shampoo_beta2 is None else shampoo_beta2

    def init_fn(params: PyTree) -> SoapState:
        per_param = jax.tree_map(_init_per_param, params)
        return SoapState(
            count=jnp.zeros([], dtype=jnp.int32),
            per_param=per_param,
        )

    def update_fn(
        grads: PyTree,
        state: SoapState,
        params: Optional[PyTree] = None,
    ) -> tuple[PyTree, SoapState]:
        del params  # unused
        count = state.count + 1

        # Bias-correction factors (global scalar, shared across all params).
        b1_t = jnp.power(b1, count.astype(jnp.float32))
        b2_t = jnp.power(b2, count.astype(jnp.float32))

        def update_leaf(g: Array, p_state: SoapPerParamState):
            # Some leaves might have no gradient (e.g. frozen params).
            if g is None:
                return g, p_state

            g2d = _reshape_to_2d(g)  # (rows, cols)
            m, v, L, R, QL, QR = p_state

            # 1. Adam momentum in original basis.
            m = (1.0 - b1) * g2d + b1 * m

            # 2. Rotate gradient + momentum into eigenbasis of (L,R).
            g_rot = QL.T @ g2d @ QR
            m_rot = QL.T @ m @ QR

            # 3. Adam-style second moment in eigenbasis.
            v = (1.0 - b2) * (g_rot * g_rot) + b2 * v

            # 4. Bias-correct (standard Adam).
            m_hat = m_rot / (1.0 - b1_t)
            v_hat = v / (1.0 - b2_t)

            # 5. Adam step in eigenbasis.
            n_rot = m_hat / (jnp.sqrt(v_hat) + eps)

            # 6. Rotate back to original basis.
            n2d = QL @ n_rot @ QR.T

            # 7. Update Kronecker second moments in original basis.
            L_update, R_update = _kronecker_second_moments(g2d)
            L = shampoo_beta2 * L + (1.0 - shampoo_beta2) * L_update
            R = shampoo_beta2 * R + (1.0 - shampoo_beta2) * R_update

            def recompute_eig(LRQLQR):
                L_, R_, QL_, QR_ = LRQLQR
                # Symmetric eigendecompositions; we only care about eigenvectors.
                _, QL_new = jnp.linalg.eigh(L_)
                _, QR_new = jnp.linalg.eigh(R_)
                return SoapPerParamState(
                    m=m,
                    v=v,
                    L=L_,
                    R=R_,
                    QL=QL_new,
                    QR=QR_new,
                ), n2d

            def keep_eig(LRQLQR):
                L_, R_, QL_, QR_ = LRQLQR
                return SoapPerParamState(
                    m=m,
                    v=v,
                    L=L_,
                    R=R_,
                    QL=QL_,
                    QR=QR_,
                ), n2d

            # Only recompute eigen-decomposition every `precondition_frequency` steps.
            do_eig = (precondition_frequency > 0) & (
                (count % precondition_frequency) == 0
            )

            new_p_state, n2d = jax.lax.cond(
                do_eig,
                recompute_eig,
                keep_eig,
                (L, R, QL, QR),
            )

            # Reshape back to original parameter shape.
            n = n2d.reshape(g.shape)
            return n, new_p_state

        updates, new_per_param = jax.tree_map(
            update_leaf, grads, state.per_param
        )

        new_state = SoapState(count=count, per_param=new_per_param)
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def soap(
    learning_rate: float,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    precondition_frequency: int = 10,
    shampoo_beta2: Optional[float] = None,
) -> optax.GradientTransformation:
    """Full SOAP optimizer (AdamW-style) as an Optax alias.

    This wraps `scale_by_soap` with decoupled weight decay and
    `scale_by_learning_rate`, mirroring AdamW’s structure in Optax. :contentReference[oaicite:1]{index=1}
    """
    return optax.chain(
        scale_by_soap(
            b1=b1,
            b2=b2,
            eps=eps,
            precondition_frequency=precondition_frequency,
            shampoo_beta2=shampoo_beta2,
        ),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_learning_rate(learning_rate),
    )
