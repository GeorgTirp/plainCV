from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax

Array = jax.Array
PyTree = Any

# We reuse the same idea as SOAP: only do full Shampoo on reasonably-sized 2D views
MAX_DIM = 2048


class ShampooPerParamState(NamedTuple):
    """State for a single parameter tensor (stored on a 2D view)."""
    L: Array          # left Kronecker factor (rows x rows)
    R: Array          # right Kronecker factor (cols x cols)
    use_shampoo: bool


class ShampooState(NamedTuple):
    count: Array      # scalar int32 step counter
    per_param: PyTree # PyTree[ShampooPerParamState]


def _is_shampoo_state(x: Any) -> bool:
    return isinstance(x, ShampooPerParamState)


def _reshape_to_2d(x: Array) -> Array:
    """Same convention as SOAP: map arbitrary param shapes to a 2D view.

    - 0D: scalar -> 1x1
    - 1D: vector -> 1 x N
    - 2D: matrix -> as is
    - 4D: conv kernel (Kh, Kw, Cin, Cout) -> Cout x (Kh*Kw*Cin)
    - Fallback: flatten leading dim, pack rest in columns.
    """
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim == 2:
        return x
    if x.ndim == 4:
        kh, kw, cin, cout = x.shape
        return jnp.reshape(x, (cout, kh * kw * cin))
    return x.reshape(x.shape[0], -1)


def scale_by_shampoo(
    eps: float = 1e-4,
    max_dim: int = MAX_DIM,
    exponent: float = 0.25,  # original Shampoo uses 1/4
) -> optax.GradientTransformation:
    """Shampoo preconditioner (matrix case) as an Optax gradient transformation.

    This implements Algorithm 1 (matrix case) from:
      *Shampoo: Preconditioned Stochastic Tensor Optimization*,
      Gupta et al., ICML 2018.

    For a 2D weight matrix W (rows x cols) and gradient G:
      - Maintain:
          L_t = sum_{s<=t} G_s G_s^T
          R_t = sum_{s<=t} G_s^T G_s
      - Update direction:
          ΔW ∝ L_t^{-1/4} G_t R_t^{-1/4}

    This transformation only computes the preconditioned gradient; the actual
    step (minus sign and learning rate) is applied by `optax.scale_by_learning_rate`.

    Args:
      eps: small diagonal jitter added inside the matrix power to keep it PD.
      max_dim: maximum rows/cols to run full Shampoo on; larger matrices
        fall back to un-preconditioned gradients for safety.
      exponent: matrix power exponent for preconditioning; 0.25 is the
        original choice (L^{-1/4}, R^{-1/4}).

    Returns:
      optax.GradientTransformation that maps raw grads -> preconditioned grads.
    """

    def init_per_param(p: Array) -> ShampooPerParamState:
        p2d = _reshape_to_2d(p)
        rows, cols = p2d.shape

        # Shapes of Kronecker factors always match gradient's 2D view
        L0 = eps * jnp.eye(rows, dtype=p.dtype)
        R0 = eps * jnp.eye(cols, dtype=p.dtype)

        # We may choose not to actually *use* Shampoo on some shapes (too big or degenerate),
        # but we still keep L,R with consistent shapes so lax.cond tracing is happy.
        too_big_or_degenerate = (
            (rows > max_dim)
            or (cols > max_dim)
            or (rows <= 1)
            or (cols <= 1)
        )
        use_shampoo = not too_big_or_degenerate

        return ShampooPerParamState(
            L=L0,
            R=R0,
            use_shampoo=use_shampoo,
        )

    def init_fn(params: PyTree) -> ShampooState:
        per_param = jtu.tree_map(init_per_param, params)
        return ShampooState(
            count=jnp.zeros([], dtype=jnp.int32),
            per_param=per_param,
        )

    def update_fn(
        grads: PyTree,
        state: ShampooState,
        params: Optional[PyTree] = None,
    ) -> tuple[PyTree, ShampooState]:
        del params  # Shampoo uses only gradients

        count = state.count + jnp.array(1, dtype=jnp.int32)

        flat_grads, treedef = jtu.tree_flatten(grads)
        flat_states, treedef2 = jtu.tree_flatten(
            state.per_param,
            is_leaf=_is_shampoo_state,
        )
        if treedef != treedef2:
            raise ValueError("Shampoo state and grads PyTrees do not match structure.")

        flat_updates: list[Array] = []
        flat_new_states: list[ShampooPerParamState] = []

        for g, s in zip(flat_grads, flat_states):
            # Convert any non-array leaf into an array; this keeps things robust
            g_arr = jnp.asarray(g)
            g2d = _reshape_to_2d(g_arr)
            rows, cols = g2d.shape

            L, R, use_shampoo = s
            use_shampoo_pred = jnp.asarray(use_shampoo, dtype=bool)

            def shampoo_branch(_):
                # -------- Shampoo update (Algorithm 1, matrix case) --------
                # L_t = L_{t-1} + G_t G_t^T
                # R_t = R_{t-1} + G_t^T G_t
                L_new = L + g2d @ g2d.T
                R_new = R + g2d.T @ g2d

                # Regularize & compute matrix powers L_new^{-exponent}, R_new^{-exponent}
                L_reg = L_new + eps * jnp.eye(rows, dtype=L_new.dtype)
                R_reg = R_new + eps * jnp.eye(cols, dtype=R_new.dtype)

                eig_L, U_L = jnp.linalg.eigh(L_reg)
                eig_R, U_R = jnp.linalg.eigh(R_reg)

                eig_L_clamped = jnp.maximum(eig_L, eps)
                eig_R_clamped = jnp.maximum(eig_R, eps)

                pow_L = eig_L_clamped ** (-exponent)
                pow_R = eig_R_clamped ** (-exponent)

                # U diag(pow) U^T, done without constructing diag explicitly
                P_L = (U_L * pow_L) @ U_L.T
                P_R = (U_R * pow_R) @ U_R.T

                # Preconditioned gradient: L^{-exp} G R^{-exp}
                g_pre_2d = P_L @ g2d @ P_R
                g_pre = g_pre_2d.reshape(g_arr.shape)

                new_s = ShampooPerParamState(
                    L=L_new,
                    R=R_new,
                    use_shampoo=use_shampoo,
                )
                return g_pre, new_s

            def identity_branch(_):
                # Fallback: just pass gradient through unchanged; still keep L,R shapes.
                new_s = ShampooPerParamState(
                    L=L,
                    R=R,
                    use_shampoo=use_shampoo,
                )
                return g_arr, new_s

            g_pre, new_s = jax.lax.cond(
                use_shampoo_pred,
                shampoo_branch,
                identity_branch,
                operand=None,
            )

            flat_updates.append(g_pre)
            flat_new_states.append(new_s)

        updates = jtu.tree_unflatten(treedef, flat_updates)
        new_per_param = jtu.tree_unflatten(treedef, flat_new_states)

        new_state = ShampooState(
            count=count,
            per_param=new_per_param,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def shampoo(
    learning_rate: float,
    eps: float = 1e-4,
    max_dim: int = MAX_DIM,
    exponent: float = 0.25,
    weight_decay: float = 0.0,  # if you want, you can wrap with add_decayed_weights externally
) -> optax.GradientTransformation:
    """Shampoo optimizer (matrix case) as an Optax alias.

    This is the “pure” Shampoo update (no Adam part):

        W_{t+1} = W_t - η * L_t^{-1/4} G_t R_t^{-1/4}

    with:
        L_t = ε I + ∑_{s<=t} G_s G_s^T
        R_t = ε I + ∑_{s<=t} G_s^T G_s

    Args:
      learning_rate: global step size η.
      eps: diagonal jitter in the preconditioners.
      max_dim: max rows/cols for a tensor to be Shampoo-preconditioned.
      exponent: matrix power exponent (original paper uses 1/4).
      weight_decay: currently unused here; you can add decoupled weight decay
        around this using `optax.add_decayed_weights` if you like.

    Returns:
      An Optax GradientTransformation implementing Shampoo.
    """
    del weight_decay  # keeping interface parallel to other opts; can wire in later

    return optax.chain(
        scale_by_shampoo(
            eps=eps,
            max_dim=max_dim,
            exponent=exponent,
        ),
        optax.scale_by_learning_rate(learning_rate),
    )
