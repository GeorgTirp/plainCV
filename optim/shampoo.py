from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax

Array = jax.Array
PyTree = Any

# Only run full Shampoo on non-degenerate 2D matrices.


class ShampooPerParamState(NamedTuple):
    """Per-parameter state.

    For 2D matrix params (`use_shampoo=True`), we use Shampoo's Kronecker state.
    For non-2D params (`use_shampoo=False`), we use AdamW moments (m, v).
    """

    m: Array
    v: Array
    L: Array
    R: Array
    use_shampoo: bool


class ShampooState(NamedTuple):
    count: Array
    per_param: PyTree  # PyTree[ShampooPerParamState]


def _is_shampoo_state(x: Any) -> bool:
    return isinstance(x, ShampooPerParamState)


def _is_shampoo_matrix(p: Array) -> bool:
    return (
        p.ndim == 2
        and p.shape[0] > 1
        and p.shape[1] > 1
    )


def scale_by_shampoo(
    shampoo_eps: float = 1e-4,
    weight_decay: float = 0.0,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    fallback_to_adamw: bool = True,
    exponent: float = 0.25,
) -> optax.GradientTransformation:
    """Muon-style routing:
    - 2D matrices: Shampoo preconditioning
    - everything else: AdamW (or identity if `fallback_to_adamw=False`)
    """

    def init_per_param(p: Array) -> ShampooPerParamState:
        p_arr = jnp.asarray(p)
        m0 = jnp.zeros_like(p_arr)
        v0 = jnp.zeros_like(p_arr)

        if p_arr.ndim == 2 and not _is_shampoo_matrix(p_arr):
            rows, cols = p_arr.shape
            if rows <= 1 or cols <= 1:
                raise ValueError(
                    f"Shampoo requires non-degenerate 2D matrices, got shape={p_arr.shape}."
                )

        if _is_shampoo_matrix(p_arr):
            rows, cols = p_arr.shape
            L0 = shampoo_eps * jnp.eye(rows, dtype=p_arr.dtype)
            R0 = shampoo_eps * jnp.eye(cols, dtype=p_arr.dtype)
            return ShampooPerParamState(
                m=m0,
                v=v0,
                L=L0,
                R=R0,
                use_shampoo=True,
            )

        one = jnp.eye(1, dtype=p_arr.dtype)
        return ShampooPerParamState(
            m=m0,
            v=v0,
            L=one,
            R=one,
            use_shampoo=False,
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
        count = state.count + jnp.array(1, dtype=jnp.int32)
        b1_t = jnp.power(adam_b1, count.astype(jnp.float32))
        b2_t = jnp.power(adam_b2, count.astype(jnp.float32))
        m_bc_den = 1.0 - b1_t
        v_bc_den = 1.0 - b2_t

        flat_grads, treedef = jtu.tree_flatten(grads)
        flat_states, treedef2 = jtu.tree_flatten(
            state.per_param,
            is_leaf=_is_shampoo_state,
        )
        if treedef != treedef2:
            raise ValueError("Shampoo state and grads PyTrees do not match structure.")

        if params is None:
            flat_params = [None] * len(flat_grads)
        else:
            flat_params, params_treedef = jtu.tree_flatten(params)
            if params_treedef != treedef:
                raise ValueError("Shampoo params and grads PyTrees do not match structure.")

        flat_updates: list[Array] = []
        flat_new_states: list[ShampooPerParamState] = []

        for g, p, s in zip(flat_grads, flat_params, flat_states):
            m, v, L, R, _ = s
            g_arr = jnp.zeros_like(m) if g is None else jnp.asarray(g)
            p_arr = None if p is None else jnp.asarray(p)

            # Reinitialize if shape changed.
            if g_arr.shape != m.shape:
                reference = p_arr if p_arr is not None else g_arr
                s = init_per_param(reference)
                m, v, L, R, _ = s
                g_arr = jnp.zeros_like(m) if g is None else jnp.asarray(g)

            # Use static rank routing to avoid Python bool conversion on traced
            # optimizer-state flags inside jit.
            if g_arr.ndim == 2:
                if (L.shape != (g_arr.shape[0], g_arr.shape[0])) or (
                    R.shape != (g_arr.shape[1], g_arr.shape[1])
                ):
                    raise ValueError(
                        "Shampoo state mismatch for a 2D matrix leaf: "
                        f"grad_shape={g_arr.shape}, L_shape={L.shape}, R_shape={R.shape}."
                    )

                g2d = g_arr
                rows, cols = g2d.shape

                L_new = L + g2d @ g2d.T
                R_new = R + g2d.T @ g2d

                L_reg = L_new + shampoo_eps * jnp.eye(rows, dtype=L_new.dtype)
                R_reg = R_new + shampoo_eps * jnp.eye(cols, dtype=R_new.dtype)

                eig_L, U_L = jnp.linalg.eigh(L_reg)
                eig_R, U_R = jnp.linalg.eigh(R_reg)
                eig_L_clamped = jnp.maximum(eig_L, shampoo_eps)
                eig_R_clamped = jnp.maximum(eig_R, shampoo_eps)
                pow_L = eig_L_clamped ** (-exponent)
                pow_R = eig_R_clamped ** (-exponent)

                P_L = (U_L * pow_L) @ U_L.T
                P_R = (U_R * pow_R) @ U_R.T
                g_pre = P_L @ g2d @ P_R

                if (p_arr is not None) and (weight_decay != 0.0):
                    g_pre = g_pre + weight_decay * p_arr

                flat_updates.append(g_pre)
                flat_new_states.append(
                    ShampooPerParamState(
                        m=m,
                        v=v,
                        L=L_new,
                        R=R_new,
                        use_shampoo=True,
                    )
                )
                continue

            if fallback_to_adamw:
                # AdamW fallback for non-2D parameters.
                m_new = (1.0 - adam_b1) * g_arr + adam_b1 * m
                v_new = (1.0 - adam_b2) * (g_arr * g_arr) + adam_b2 * v
                m_hat = m_new / m_bc_den
                v_hat = v_new / v_bc_den
                upd = m_hat / (jnp.sqrt(v_hat) + adam_eps)
                if (p_arr is not None) and (weight_decay != 0.0):
                    upd = upd + weight_decay * p_arr
                new_s = ShampooPerParamState(
                    m=m_new,
                    v=v_new,
                    L=L,
                    R=R,
                    use_shampoo=False,
                )
            else:
                # Identity fallback to preserve callers that only want Shampoo
                # preconditioning on matrix params (e.g., Sophia+Shampoo).
                upd = g_arr
                new_s = ShampooPerParamState(
                    m=m,
                    v=v,
                    L=L,
                    R=R,
                    use_shampoo=False,
                )
            flat_updates.append(upd)
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
    exponent: float = 0.25,
    weight_decay: float = 0.0,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
) -> optax.GradientTransformation:
    """Muon-style Shampoo optimizer:
    - 2D params: Shampoo
    - non-2D params: AdamW
    """
    return optax.chain(
        scale_by_shampoo(
            shampoo_eps=eps,
            weight_decay=weight_decay,
            adam_b1=adam_b1,
            adam_b2=adam_b2,
            adam_eps=adam_eps,
            exponent=exponent,
        ),
        optax.scale_by_learning_rate(learning_rate),
    )
