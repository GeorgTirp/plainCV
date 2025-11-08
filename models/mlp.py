# models/mlp.py
from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    """Simple MLP classifier for (Fashion-)MNIST."""
    hidden_sizes: Sequence[int] = (512, 512)
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (batch, H, W, C)
        x = x.astype(jnp.float32) / 255.0
        x = x.reshape((x.shape[0], -1))  # flatten

        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x
