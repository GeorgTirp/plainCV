# optim/sophia.py
from __future__ import annotations

from typing import Any, NamedTuple, Callable, Optional

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax.flatten_util import ravel_pytree
import optax

from .ggn_utils import HessianMatvecFn
from .shampoo import scale_by_shampoo, ShampooState  # reuse your Shampoo code

Array = jax.Array
PyTree = Any
Params = Any


# ---------------------------------------------------------------------------
# Small PyTree helpers
# ---------------------------------------------------------------------------

def _tree_zeros_like(x: PyTree) -> PyTree:
    return jtu.tree_map(jnp.zeros_like, x)


def _tree_scale(x: PyTree, alpha: Array) -> PyTree:
    return jtu.tree_map(lambda a: alpha * a, x)


# ---------------------------------------------------------------------------
# Sophia (Hutchinson Hessian-diagonal + momentum)
# ---------------------------------------------------------------------------

class SophiaState(NamedTuple):
    count: Array          # scalar int32 step
    m: PyTree             # first moment (momentum)
    h: PyTree             # EMA of Hessian diagonal estimate
    rng_key: Array        # PRNG key for Hutchinson & dropout


def sophia(
    learning_rate: float,
    hessian_matvec_fn: HessianMatvecFn,
    beta1: float = 0.9,
    beta2: float = 0.99,
    rho: float = 0.01,        # damping / numerical stabilizer
    h_max: float = 1e6,       # clip for Hessian diagonal
    eps: float = 1e-8,
    hessian_update_every: int = 10,
) -> optax.GradientTransformation:
    """
    Plain Sophia-style optimizer (no Shampoo) following the core idea of:

      Liu et al., "Sophia: A Scalable Stochastic Second-order Optimizer
      for Language Model Pre-training" (arXiv:2305.14342)

    We maintain:
      m_t = β1 m_{t-1} + (1-β1) g_t
      h_t = β2 h_{t-1} + (1-β2) clip( diag(H_t), 0, h_max )

    Update:
      θ_{t+1} = θ_t - η * m_t / (h_t + ρ + ε)

    where diag(H_t) is a Hutchinson-style diagonal estimate using
    a Hessian-vector product from `hessian_matvec_fn`.
    """
    lr = float(learning_rate)

    def init_fn(params: Params) -> SophiaState:
        m0 = _tree_zeros_like(params)
        h0 = _tree_zeros_like(params)
        rng_key = jax.random.PRNGKey(0)
        count = jnp.zeros([], dtype=jnp.int32)
        return SophiaState(
            count=count,
            m=m0,
            h=h0,
            rng_key=rng_key,
        )

    def update_fn(
        grads: PyTree,
        state: SophiaState,
        params: Optional[Params] = None,
    ) -> tuple[PyTree, SophiaState]:
        assert params is not None, "Sophia requires `params` for Hessian-vector products."

        count = state.count + jnp.array(1, dtype=jnp.int32)
        m_old = state.m
        h_old = state.h
        rng_key = state.rng_key

        # --------------------------------------------------------------
        # 1) Hessian diagonal update via Hutchinson every K steps
        # --------------------------------------------------------------
        def do_hessian_update(carry):
            h_prev, rng = carry

            flat_params, unravel = ravel_pytree(params)
            dim = flat_params.shape[0]

            # Sample Rademacher vector ξ ∈ {−1, +1}^d
            rng, key_xi = jax.random.split(rng)
            xi_flat = jax.random.choice(
                key_xi, jnp.array([-1.0, 1.0]), shape=(dim,)
            )
            xi_tree = unravel(xi_flat)

            # Hessian-vector product Hv = H ξ (on the curvature batch)
            rng, key_hess = jax.random.split(rng)
            hv_tree = hessian_matvec_fn(params, xi_tree, key_hess)
            hv_flat, _ = ravel_pytree(hv_tree)

            # Hutchinson diag estimate: diag(H) ≈ (H ξ) ⊙ ξ
            diag_est_flat = hv_flat * xi_flat
            diag_est_tree = unravel(diag_est_flat)

            def _update_h(h_prev_leaf, est_leaf):
                est_clipped = jnp.clip(est_leaf, 0.0, h_max)
                return beta2 * h_prev_leaf + (1.0 - beta2) * est_clipped

            h_new = jtu.tree_map(_update_h, h_prev, diag_est_tree)
            return (h_new, rng)

        def skip_hessian_update(carry):
            return carry

        do_update = (hessian_update_every > 0) & (
            (count % hessian_update_every) == 0
        )

        (h_new, rng_key) = jax.lax.cond(
            do_update,
            do_hessian_update,
            skip_hessian_update,
            operand=(h_old, rng_key),
        )

        # --------------------------------------------------------------
        # 2) Momentum on the (raw) gradients
        # --------------------------------------------------------------
        def _update_m(m_leaf, g_leaf):
            return beta1 * m_leaf + (1.0 - beta1) * g_leaf

        m_new = jtu.tree_map(_update_m, m_old, grads)

        # --------------------------------------------------------------
        # 3) Diagonal second-order scaling: m / (h + ρ + ε)
        # --------------------------------------------------------------
        def _scale_step(m_leaf, h_leaf):
            denom = h_leaf + rho + eps
            return m_leaf / denom

        step = jtu.tree_map(_scale_step, m_new, h_new)

        # --------------------------------------------------------------
        # 4) Form updates: Δθ = -η * step
        # --------------------------------------------------------------
        updates = _tree_scale(step, -lr)

        new_state = SophiaState(
            count=count,
            m=m_new,
            h=h_new,
            rng_key=rng_key,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Sophia + Shampoo
# ---------------------------------------------------------------------------

class SophiaShampooState(NamedTuple):
    count: Array
    m: PyTree
    h: PyTree
    shampoo_state: ShampooState
    rng_key: Array


def sophia_shampoo(
    learning_rate: float,
    hessian_matvec_fn: HessianMatvecFn,
    beta1: float = 0.9,
    beta2: float = 0.99,
    rho: float = 0.01,
    h_max: float = 1e6,
    eps: float = 1e-8,
    hessian_update_every: int = 10,
    # Shampoo-specific knobs
    shampoo_eps: float = 1e-4,
    shampoo_max_dim: int = 2048,
    shampoo_exponent: float = 0.25,
) -> optax.GradientTransformation:
    """
    Sophia + Shampoo:

      1. Precondition gradients with Shampoo: g̃_t = L^{-1/4} g_t R^{-1/4}.
      2. Maintain momentum on g̃_t:
           m_t = β1 m_{t-1} + (1-β1) g̃_t
      3. Maintain Hessian-diagonal EMA h_t from Hutchinson HVPs (as in Sophia).
      4. Update:
           θ_{t+1} = θ_t - η * m_t / (h_t + ρ + ε)

    So Shampoo provides Kronecker preconditioning for matrix-shaped params,
    and Sophia’s diagonal Hessian EMA provides additional curvature-aware
    coordinate scaling.
    """

    lr = float(learning_rate)
    # Build an inner Shampoo transform to reuse its state and update logic
    shampoo_tx = scale_by_shampoo(
        eps=shampoo_eps,
        max_dim=shampoo_max_dim,
        exponent=shampoo_exponent,
    )

    def init_fn(params: Params) -> SophiaShampooState:
        m0 = _tree_zeros_like(params)
        h0 = _tree_zeros_like(params)
        shampoo_state = shampoo_tx.init(params)
        rng_key = jax.random.PRNGKey(0)
        count = jnp.zeros([], dtype=jnp.int32)
        return SophiaShampooState(
            count=count,
            m=m0,
            h=h0,
            shampoo_state=shampoo_state,
            rng_key=rng_key,
        )

    def update_fn(
        grads: PyTree,
        state: SophiaShampooState,
        params: Optional[Params] = None,
    ) -> tuple[PyTree, SophiaShampooState]:
        assert params is not None, "Sophia+Shampoo requires `params` for HVPs."

        count = state.count + jnp.array(1, dtype=jnp.int32)
        m_old = state.m
        h_old = state.h
        shampoo_state = state.shampoo_state
        rng_key = state.rng_key

        # --------------------------------------------------------------
        # 1) Hessian diagonal update (same as Sophia)
        # --------------------------------------------------------------
        def do_hessian_update(carry):
            h_prev, rng = carry

            flat_params, unravel = ravel_pytree(params)
            dim = flat_params.shape[0]

            rng, key_xi = jax.random.split(rng)
            xi_flat = jax.random.choice(
                key_xi, jnp.array([-1.0, 1.0]), shape=(dim,)
            )
            xi_tree = unravel(xi_flat)

            rng, key_hess = jax.random.split(rng)
            hv_tree = hessian_matvec_fn(params, xi_tree, key_hess)
            hv_flat, _ = ravel_pytree(hv_tree)

            diag_est_flat = hv_flat * xi_flat
            diag_est_tree = unravel(diag_est_flat)

            def _update_h(h_prev_leaf, est_leaf):
                est_clipped = jnp.clip(est_leaf, 0.0, h_max)
                return beta2 * h_prev_leaf + (1.0 - beta2) * est_clipped

            h_new = jtu.tree_map(_update_h, h_prev, diag_est_tree)
            return (h_new, rng)

        def skip_hessian_update(carry):
            return carry

        do_update = (hessian_update_every > 0) & (
            (count % hessian_update_every) == 0
        )

        (h_new, rng_key) = jax.lax.cond(
            do_update,
            do_hessian_update,
            skip_hessian_update,
            operand=(h_old, rng_key),
        )

        # --------------------------------------------------------------
        # 2) Shampoo preconditioning on the raw gradients
        # --------------------------------------------------------------
        #   g̃_t, shampoo_state_t = Shampoo(grads)
        g_pre, new_shampoo_state = shampoo_tx.update(
            grads,
            shampoo_state,
            params=params,
        )

        # --------------------------------------------------------------
        # 3) Momentum on the Shampoo-preconditioned gradients
        # --------------------------------------------------------------
        def _update_m(m_leaf, g_leaf):
            return beta1 * m_leaf + (1.0 - beta1) * g_leaf

        m_new = jtu.tree_map(_update_m, m_old, g_pre)

        # --------------------------------------------------------------
        # 4) Diagonal second-order scaling: m / (h + ρ + ε)
        # --------------------------------------------------------------
        def _scale_step(m_leaf, h_leaf):
            denom = h_leaf + rho + eps
            return m_leaf / denom

        step = jtu.tree_map(_scale_step, m_new, h_new)

        # --------------------------------------------------------------
        # 5) Form updates: Δθ = -η * step
        # --------------------------------------------------------------
        updates = _tree_scale(step, -lr)

        new_state = SophiaShampooState(
            count=count,
            m=m_new,
            h=h_new,
            shampoo_state=new_shampoo_state,
            rng_key=rng_key,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
