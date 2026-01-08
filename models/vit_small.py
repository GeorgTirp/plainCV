# models/vit_small.py
import jax.numpy as jnp
from flax import linen as nn


class MlpBlock(nn.Module):
    """Transformer MLP block."""
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool):
        hidden = nn.Dense(self.mlp_dim)(x)
        hidden = nn.gelu(hidden)
        hidden = nn.Dropout(rate=self.dropout_rate)(hidden, deterministic=not train)
        hidden = nn.Dense(x.shape[-1])(hidden)
        hidden = nn.Dropout(rate=self.dropout_rate)(hidden, deterministic=not train)
        return hidden


class EncoderBlock(nn.Module):
    """Single Transformer encoder block with pre-norm."""
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1
    use_layernorm: bool = True
    use_batchnorm: bool = False

    @nn.compact
    def __call__(self, x, train: bool):
        if self.use_batchnorm and self.use_layernorm:
            raise ValueError("use_batchnorm and use_layernorm cannot both be True.")

        # Multi-head self-attention
        if self.use_batchnorm:
            y = nn.BatchNorm(use_running_average=not train)(x)
        elif self.use_layernorm:
            y = nn.LayerNorm()(x)
        else:
            y = x
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            deterministic=not train,
            dropout_rate=self.dropout_rate,
        )(y)
        x = x + y

        # MLP
        if self.use_batchnorm:
            y = nn.BatchNorm(use_running_average=not train)(x)
        elif self.use_layernorm:
            y = nn.LayerNorm()(x)
        else:
            y = x
        y = MlpBlock(mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate)(y, train=train)
        return x + y


class VisionTransformer(nn.Module):
    """Tiny ViT for small grayscale images like Fashion-MNIST."""
    num_classes: int = 10
    patch_size: int = 4
    hidden_size: int = 128
    mlp_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout_rate: float = 0.1
    use_layernorm: bool = True
    use_batchnorm: bool = False

    def setup(self):
        self.cls_token = self.param(
            "cls_token",
            nn.initializers.zeros,
            (1, 1, self.hidden_size),
        )

    def _patchify(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert images to a sequence of patch embeddings."""
        # x: (batch, H, W, C)
        conv = nn.Conv(
            features=self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=True,
        )
        x = conv(x)  # (batch, H/ps, W/ps, hidden)
        b, h, w, c = x.shape
        x = x.reshape((b, h * w, c))
        return x, (h, w)

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x.astype(jnp.float32) / 255.0

        # Patch embedding + positional encodings
        x, (grid_h, grid_w) = self._patchify(x)
        num_tokens = grid_h * grid_w + 1  # +1 for cls token
        pos_embed = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, num_tokens, self.hidden_size),
        )

        cls_tokens = jnp.tile(self.cls_token, (x.shape[0], 1, 1))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = x + pos_embed
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # Transformer encoder
        for _ in range(self.num_layers):
            x = EncoderBlock(
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            use_layernorm=self.use_layernorm,
            use_batchnorm=self.use_batchnorm,
        )(x, train=train)

        if self.use_batchnorm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        elif self.use_layernorm:
            x = nn.LayerNorm()(x)
        cls_repr = x[:, 0]
        logits = nn.Dense(self.num_classes)(cls_repr)
        return logits
