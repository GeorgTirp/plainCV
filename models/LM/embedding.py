import jax.numpy as jnp
from typing import Tuple
import jax

# -------------------------
# RoPE (complex-like)
# -------------------------
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, condense_ratio: int = 1):
    """
    Returns RoPE frequencies as cos/sin pairs.

    Output shape: (1, end, 1, dim/2, 2)
      freqs_cis[..., 0] = cos
      freqs_cis[..., 1] = sin
    """
    if dim % 2 != 0:
        raise ValueError("dim must be even (pairs of real/imag).")

    inv_freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))  # (dim/2,)
    t = (jnp.arange(end, dtype=jnp.float32) / float(condense_ratio))                # (end,)
    freqs = jnp.outer(t, inv_freqs).astype(jnp.float32)                             # (end, dim/2)

    cos = jnp.cos(freqs)[None, :, None, :]  # (1, end, 1, dim/2)
    sin = jnp.sin(freqs)[None, :, None, :]  # (1, end, 1, dim/2)

    return jnp.stack([cos, sin], axis=-1)     # (1, end, 1, dim/2, 2)

@jax.jit
def apply_rotary_emb_complex_like(
    q: jnp.ndarray,
    k: jnp.ndarray,
    freqs_cis: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    q, k: (B, T, H, D)
    freqs_cis: (1, T, 1, D/2, 2) with [...,0]=cos, [...,1]=sin

    Returns rotated q, k with same shapes as inputs.
    """
    # concatenate heads: (B, T, 2H, D)
    qk = jnp.concatenate([q, k], axis=2)

    b, t, two_h, d = qk.shape
    if d % 2 != 0:
        raise ValueError("head_dim must be even for complex-like RoPE.")

    # unflatten last dim into (D/2, 2): (B, T, 2H, D/2, 2)
    qk_r2 = qk.reshape(b, t, two_h, d // 2, 2).astype(jnp.float32)

    cos = freqs_cis[..., 0]  # (1, T, 1, D/2)
    sin = freqs_cis[..., 1]  # (1, T, 1, D/2)

    # complex multiply: (a+ib) * (cos + i sin)
    a = qk_r2[..., 0]
    b_im = qk_r2[..., 1]

    rot_a = a * cos - b_im * sin
    rot_b = b_im * cos + a * sin

    rotated = jnp.stack([rot_a, rot_b], axis=-1)      # (B, T, 2H, D/2, 2)
    rotated = rotated.reshape(b, t, two_h, d)         # (B, T, 2H, D)
    rotated = rotated.astype(q.dtype)

    h = q.shape[2]  # original n_heads
    q_rot = rotated[:, :, :h, :]
    k_rot = rotated[:, :, h:, :]
    return q_rot, k_rot