# models/resnet_small.py
from typing import Callable, Any

import jax.numpy as jnp
from flax import linen as nn


ModuleDef = Any


class ResidualBlock(nn.Module):
    filters: int
    stride: int = 1
    use_bn: bool = True

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x

        y = nn.Conv(self.filters, (3, 3), self.stride, padding="SAME", use_bias=not self.use_bn)(x)
        if self.use_bn:
            y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)

        y = nn.Conv(self.filters, (3, 3), strides=1, padding="SAME", use_bias=not self.use_bn)(y)
        if self.use_bn:
            y = nn.BatchNorm(use_running_average=not train)(y)

        if residual.shape != y.shape:
            residual = nn.Conv(self.filters, (1, 1), self.stride, padding="SAME", use_bias=False)(residual)
            if self.use_bn:
                residual = nn.BatchNorm(use_running_average=not train)(residual)

        return nn.relu(residual + y)


class SmallResNet(nn.Module):
    """Tiny ResNet for (Fashion-)MNIST / CIFAR size images."""
    num_classes: int = 10
    use_bn: bool = True

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Expect x: (batch, H, W, C), Fashion-MNIST: 28x28x1
        x = x.astype(jnp.float32) / 255.0

        # Stem
        x = nn.Conv(64, (3, 3), strides=1, padding="SAME", use_bias=not self.use_bn)(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # ResNet-like stages, very small
        x = ResidualBlock(64, stride=1, use_bn=self.use_bn)(x, train=train)
        x = ResidualBlock(64, stride=1, use_bn=self.use_bn)(x, train=train)

        x = ResidualBlock(128, stride=2, use_bn=self.use_bn)(x, train=train)
        x = ResidualBlock(128, stride=1, use_bn=self.use_bn)(x, train=train)

        x = ResidualBlock(256, stride=2, use_bn=self.use_bn)(x, train=train)
        x = ResidualBlock(256, stride=1, use_bn=self.use_bn)(x, train=train)

        x = jnp.mean(x, axis=(1, 2))  # global average pool
        x = nn.Dense(self.num_classes)(x)
        return x
