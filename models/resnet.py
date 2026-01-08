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



class ResNet30(nn.Module):
    """
    ResNet-30 (BasicBlock) for small images (MNIST/CIFAR-ish).
    Layer count: stem conv (1) + 2*sum(blocks)=28 + classifier dense (1) => 30
    Uses block configuration [3, 4, 4, 3].
    """
    num_classes: int = 10
    use_bn: bool = True

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Input: (B, H, W, C)
        x = x.astype(jnp.float32) / 255.0

        def make_stage(x, filters: int, blocks: int, first_stride: int):
            # First block may downsample via stride
            x = ResidualBlock(filters, stride=first_stride, use_bn=self.use_bn)(x, train=train)
            for _ in range(blocks - 1):
                x = ResidualBlock(filters, stride=1, use_bn=self.use_bn)(x, train=train)
            return x

        # Stem (small-image friendly: 3x3, stride 1, no maxpool)
        x = nn.Conv(64, (3, 3), strides=1, padding="SAME", use_bias=not self.use_bn)(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Stages: [3, 4, 4, 3] blocks with channel sizes [64, 128, 256, 512]
        x = make_stage(x, 64,  blocks=3, first_stride=1)
        x = make_stage(x, 128, blocks=4, first_stride=2)
        x = make_stage(x, 256, blocks=4, first_stride=2)
        x = make_stage(x, 512, blocks=3, first_stride=2)

        # Head
        x = jnp.mean(x, axis=(1, 2))  # global average pool
        x = nn.Dense(self.num_classes)(x)
        return x


class ResNet18(nn.Module):
    """
    ResNet-18 (BasicBlock) for small images (MNIST/CIFAR-ish).
    Uses block configuration [2, 2, 2, 2] with channel sizes [64, 128, 256, 512].

    Note: This is "small-image" ResNet-18 (no 7x7 conv / no maxpool).
    If you want ImageNet-style ResNet-18 stem, say so and I’ll provide that variant too.
    """
    num_classes: int = 10
    use_bn: bool = True

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.astype(jnp.float32) / 255.0

        def make_stage(x, filters: int, blocks: int, first_stride: int):
            x = ResidualBlock(filters, stride=first_stride, use_bn=self.use_bn)(x, train=train)
            for _ in range(blocks - 1):
                x = ResidualBlock(filters, stride=1, use_bn=self.use_bn)(x, train=train)
            return x

        # Stem (small-image friendly)
        x = nn.Conv(64, (3, 3), strides=1, padding="SAME", use_bias=not self.use_bn)(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        # Stages: [2, 2, 2, 2]
        x = make_stage(x, 64,  blocks=2, first_stride=1)
        x = make_stage(x, 128, blocks=2, first_stride=2)
        x = make_stage(x, 256, blocks=2, first_stride=2)
        x = make_stage(x, 512, blocks=2, first_stride=2)

        # Head
        x = jnp.mean(x, axis=(1, 2))  # global average pool
        x = nn.Dense(self.num_classes)(x)
        return x
