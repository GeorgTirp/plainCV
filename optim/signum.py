"""Signum optimizer (signSGD with momentum)."""

from typing import NamedTuple, Optional

import jax.numpy as jnp
from jax import tree_util as jtu
import optax


class SignumState(NamedTuple):
    momentum_buffer: optax.Updates


def signum(
    learning_rate: float,
    momentum: float = 0.9,
    nesterov: bool = False,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Build Signum with optional Nesterov momentum and decoupled weight decay."""
    if learning_rate < 0.0:
        raise ValueError(f"learning_rate must be >= 0, got {learning_rate}.")
    if momentum < 0.0 or momentum >= 1.0:
        raise ValueError(f"momentum must be in [0, 1), got {momentum}.")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}.")

    one_minus_momentum = 1.0 - momentum

    def init_fn(params: optax.Params) -> SignumState:
        momentum_buffer = jtu.tree_map(jnp.zeros_like, params)
        return SignumState(momentum_buffer=momentum_buffer)

    def update_fn(
        updates: optax.Updates,
        state: SignumState,
        params: Optional[optax.Params] = None,
    ) -> tuple[optax.Updates, SignumState]:
        momentum_buffer = jtu.tree_map(
            lambda m, g: momentum * m + one_minus_momentum * g,
            state.momentum_buffer,
            updates,
        )

        if nesterov:
            direction = jtu.tree_map(
                lambda g, m: one_minus_momentum * g + momentum * m,
                updates,
                momentum_buffer,
            )
        else:
            direction = momentum_buffer

        signed_updates = jtu.tree_map(jnp.sign, direction)

        if weight_decay > 0.0:
            if params is None:
                raise ValueError("Signum with weight_decay requires current params.")
            signed_updates = jtu.tree_map(
                lambda u, p: u + weight_decay * p, signed_updates, params
            )

        scaled_updates = jtu.tree_map(lambda u: -learning_rate * u, signed_updates)
        return scaled_updates, SignumState(momentum_buffer=momentum_buffer)

    return optax.GradientTransformation(init_fn, update_fn)
