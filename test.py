import jax, jax.numpy as jnp
from flax import linen as nn

print("jax:", jax.__version__)
print("devices:", jax.devices())
print("backend:", jax.default_backend())

class TestDense(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)

x = jnp.ones((2, 4, 8), dtype=jnp.float32)
model = TestDense(features=24)

params = model.init(jax.random.PRNGKey(0), x)
y = model.apply(params, x)
print("output shape:", y.shape)
