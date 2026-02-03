import math
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from .embedding import precompute_freqs_cis, apply_rotary_emb_complex_like

# -------------------------
# Config
# -------------------------
@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    dim: int
    expand: float
    n_layers: int
    n_heads: int
    mlp: Literal["mlp", "glu", "mlp_relu_sq"] = "mlp"
    rmsnorm_eps: float = 1e-6
    tie_embeddings: bool = False
    rope_theta: float = 500000.0  # your PyTorch uses 500000


# -------------------------
# Norm selector (optional)
# -------------------------
class Norm(nn.Module):
    dim: int
    normtype: Literal["rmsnorm", "layernorm", "batchnorm"] = "rmsnorm"
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        if self.normtype == "rmsnorm":
            # flax.linen.RMSNorm exists in Linen.
            return nn.RMSNorm(epsilon=self.eps)(x)
        if self.normtype == "layernorm":
            return nn.LayerNorm(epsilon=self.eps)(x)
        if self.normtype == "batchnorm":
            # BatchNorm needs mutable batch_stats when training.
            return nn.BatchNorm(use_running_average=deterministic, momentum=0.9, epsilon=self.eps)(x)
        raise ValueError("normtype must be one of: rmsnorm, layernorm, batchnorm")



# -------------------------
# MLP variants
# -------------------------
class MLP(nn.Module):
    dim: int
    hidden_dim: int
    base_init: nn.initializers.Initializer
    out_init: nn.initializers.Initializer

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=self.base_init, name="fc1")(x)
        x = nn.silu(x)
        x = nn.Dense(self.dim, use_bias=False, kernel_init=self.out_init, name="fc2")(x)
        return x


class GLU(nn.Module):
    dim: int
    hidden_dim: int
    base_init: nn.initializers.Initializer
    out_init: nn.initializers.Initializer

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=self.base_init, name="fc_gate")(x)
        up   = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=self.base_init, name="fc_up")(x)
        x = nn.silu(gate) * up
        x = nn.Dense(self.dim, use_bias=False, kernel_init=self.out_init, name="fc2")(x)
        return x


class MLPReluSquared(nn.Module):
    dim: int
    hidden_dim: int
    base_init: nn.initializers.Initializer
    out_init: nn.initializers.Initializer

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=self.base_init, name="fc1")(x)
        x = jnp.square(jax.nn.relu(x))
        x = nn.Dense(self.dim, use_bias=False, kernel_init=self.out_init, name="fc2")(x)
        return x


# -------------------------
# Attention (your PyTorch Attention)
# -------------------------
class Attention(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,                 # (B, T, D)
        freqs_cis: jnp.ndarray,          # (T, H/2) complex
        attn_mask: Optional[jnp.ndarray] = None,  # (B, T, T) bool or additive
    ) -> jnp.ndarray:
        d = self.cfg.dim
        n_heads = self.cfg.n_heads
        assert d % n_heads == 0
        head_dim = d // n_heads

        bsz, seqlen, _ = x.shape

        # GPT-style init std=0.02 (like your torch init)
        base_init = nn.initializers.normal(stddev=0.02)  # 
        # scale residual branches like your torch _scale_residual_branches
        resid_init = nn.initializers.normal(stddev=0.02 / math.sqrt(2 * self.cfg.n_layers))

        # (B, T, 3D)
        qkv = nn.Dense(3 * d, use_bias=False, kernel_init=base_init, name="w_qkv")(x)

        # Split to (B, T, D) each
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # (B, T, N, H)
        q = q.reshape(bsz, seqlen, n_heads, head_dim)
        k = k.reshape(bsz, seqlen, n_heads, head_dim)
        v = v.reshape(bsz, seqlen, n_heads, head_dim)

        # RoPE
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)

        # Mask handling:
        # jax.nn.dot_product_attention expects mask/bias broadcastable to (B, N, T, S). 
        mask = None
        bias = None
        is_causal = False

        if attn_mask is None:
            is_causal = True
        else:
            # (B, 1, T, T) broadcasts over heads
            expanded = attn_mask[:, None, :, :]
            if expanded.dtype == jnp.bool_:
                mask = expanded
            else:
                # additive mask goes in bias per docs 
                bias = expanded.astype(x.dtype)


        # out: (B, T, N, H)
        out = jax.nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=mask,
            bias=bias,
            is_causal=is_causal,
        )

        # (B, T, D)
        out = out.reshape(bsz, seqlen, d)

        # output proj (residual-scaled init)
        out = nn.Dense(d, use_bias=False, kernel_init=resid_init, name="w_out")(out)
        return out


# -------------------------
# Block (your PyTorch Block)
# -------------------------
class Block(nn.Module):
    cfg: ModelConfig
    layer_id: int
    normtype: Literal["rmsnorm", "layernorm", "batchnorm"] = "rmsnorm"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        freqs_cis: jnp.ndarray,
        attn_mask: Optional[jnp.ndarray],
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # Norms
        attn_norm = Norm(self.cfg.dim, normtype=self.normtype, eps=self.cfg.rmsnorm_eps, name="attn_norm")
        mlp_norm  = Norm(self.cfg.dim, normtype=self.normtype, eps=self.cfg.rmsnorm_eps, name="mlp_norm")

        # Attention
        x = x + Attention(self.cfg, name="attn")(attn_norm(x, deterministic=deterministic), freqs_cis, attn_mask)

        # MLP variant
        hidden_dim = int(self.cfg.expand * self.cfg.dim)
        base_init  = nn.initializers.normal(stddev=0.02)
        resid_init = nn.initializers.normal(stddev=0.02 / math.sqrt(2 * self.cfg.n_layers))

        if self.cfg.mlp == "mlp":
            mlp = MLP(self.cfg.dim, hidden_dim, base_init=base_init, out_init=resid_init, name="mlp")
        elif self.cfg.mlp == "glu":
            mlp = GLU(self.cfg.dim, hidden_dim, base_init=base_init, out_init=resid_init, name="mlp")
        elif self.cfg.mlp == "mlp_relu_sq":
            mlp = MLPReluSquared(self.cfg.dim, hidden_dim, base_init=base_init, out_init=resid_init, name="mlp")

        else:
            raise ValueError(f"Unknown mlp type: {self.cfg.mlp}")

        x = x + mlp(mlp_norm(x, deterministic=deterministic))
        return x


# -------------------------
# Transformer (your PyTorch Transformer)
# -------------------------
class Transformer(nn.Module):
    cfg: ModelConfig
    normtype: Literal["rmsnorm", "layernorm", "batchnorm"] = "rmsnorm"

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,                 # (B, T)
        attn_mask: Optional[jnp.ndarray] = None, # (B, T, T) or None
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        cfg = self.cfg
        if cfg.dim % cfg.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        head_dim = cfg.dim // cfg.n_heads

        base_init = nn.initializers.normal(stddev=0.02)

        # Token embedding
        embed = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.dim,
            embedding_init=base_init,
            name="embed_tokens",
        )
        x = embed(input_ids)  # (B, T, D)

        # Precompute freqs up to cfg.seq_len once per call (simple + correct).
        # If you want, you can also precompute outside and pass them in.
        freqs_cis = precompute_freqs_cis(head_dim, cfg.seq_len, theta=cfg.rope_theta)
        freqs_cis = freqs_cis[:, :x.shape[1], :, :, :]


        # Layers
        for i in range(cfg.n_layers):
            x = Block(cfg, layer_id=i, normtype=self.normtype, name=f"layers_{i}")(
                x, freqs_cis, attn_mask, deterministic=deterministic
            )

        # Output norm
        x = Norm(cfg.dim, normtype=self.normtype, eps=cfg.rmsnorm_eps, name="out_norm")(x, deterministic=deterministic)

        # LM head (optionally tie embeddings)
        if cfg.tie_embeddings:
            # Use Embed.attend for tied output projection. 
            logits = embed.attend(x)
        else:
            logits = nn.Dense(cfg.vocab_size, use_bias=False, kernel_init=base_init, name="lm_head")(x)

        return logits
