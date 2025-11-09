# optim/pns_eigenadam.py
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax


class PnsEigenAdamState(NamedTuple):
    """State for PN-S EigenAdam.

    Fields here are just a starting point; you’ll extend them with:
      - curvature / eigen info
      - RNG key for stochastic stuff
      - etc.
    """
    # Underlying Adam moments (m, v, etc.)
    adam_state: optax.OptState

    # Step counter
    step: int

    # Placeholder for curvature-related state (GGN eigenvectors, eigenvalues, etc.)
    # Could be PyTree matching params, or a global structure.
    curvature_state: Any

    # Optional RNG key for sampling directions, stochastic Lanczos, etc.
    rng_key: Optional[jax.Array] = None


def pns_eigenadam(
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    curvature_update_every: int = 100,
    max_eigenvectors: int = 16,
) -> optax.GradientTransformation:
    """PN-S EigenAdam as an Optax gradient transformation (skeleton).

    Args:
      learning_rate: base learning rate.
      beta1, beta2, eps: Adam hyperparameters.
      weight_decay: optional weight decay.
      curvature_update_every: how often (in steps) to recompute curvature info.
      max_eigenvectors: target number of eigen-directions to track.

    Returns:
      An optax.GradientTransformation that can be passed to TrainState.
    """
    # Start from a standard AdamW-like transform for the "inner" update.
    base_adam = optax.adamw(
        learning_rate=learning_rate,
        b1=beta1,
        b2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )

    def init_fn(params):
        """Initialize optimizer state given initial params."""
        adam_state = base_adam.init(params)

        # Placeholder curvature state: you will later store
        # eigenvalues/eigenvectors per layer, etc.
        curvature_state = None

        # Initialize RNG for curvature sampling / Lanczos if you want
        rng_key = jax.random.PRNGKey(0)

        return PnsEigenAdamState(
            adam_state=adam_state,
            step=0,
            curvature_state=curvature_state,
            rng_key=rng_key,
        )

    def update_fn(grads, state: PnsEigenAdamState, params=None):
        """Apply one optimization step.

        This is where the PN-S magic will eventually go.
        """
        step = state.step + 1
        rng_key = state.rng_key
        curvature_state = state.curvature_state

        # 1. OPTIONALLY: update / recompute curvature info every N steps
        #    e.g., run Lanczos with GGN HVPs to update eigenbasis.
        if (step % curvature_update_every == 0) and (params is not None):
            # rng_key, curvature_state = update_curvature(
            #     rng_key, params, grads, curvature_state, max_eigenvectors
            # )
            pass

        # 2. OPTIONALLY: precondition the gradients using curvature info.
        #    For now, we leave them unchanged as a placeholder.
        #    Later you'll implement:
        #      g_pre = M_partial @ grads   (Eigen-space + identity elsewhere)
        preconditioned_grads = grads

        # 3. Forward to the inner AdamW transform with preconditioned gradients.
        updates, new_adam_state = base_adam.update(
            preconditioned_grads,
            state.adam_state,
            params=params,
        )

        new_state = PnsEigenAdamState(
            adam_state=new_adam_state,
            step=step,
            curvature_state=curvature_state,
            rng_key=rng_key,
        )

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
