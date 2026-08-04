"""Kaon: Muon's singular *basis*, random singular *values*.

Kaon is the ablation that keeps everything Muon does -- the same momentum buffer
(``beta``, Nesterov), the same Frobenius pre-normalization, and the same
``sqrt(max(1, fan_out / fan_in))`` shape scaling -- but replaces the (near-)unit
singular values of the orthogonalized momentum by *random positive noise redrawn
every step*::

    M = U S V^T            (SVD of the momentum matrix)
    Delta_W = -eta * U diag(s_rand) V^T ,   s_rand ~ |N(0,1)|  (or Exp(1))

The left/right singular subspaces ``U, V`` (the incoherent gradient basis) are
preserved; only the magnitude profile is randomized. Because the random weights
are independent of curvature, ``E[Cov(allocation, curvature)] = 0``, so Kaon's
exposure ``A / lambda_max`` should coincide with Muon's (Corollary 4.3). The
random values are normalized to unit RMS so the step magnitude matches Muon's
orthogonal factor and Muon's learning rate transfers unchanged; exposure itself
is scale-invariant, so this normalization does not bias the measured quantity.

Non-routed parameters use the same AdamW fallback as Muon, routed by the shared
matrix routing (:func:`optim.matrix_routing.should_use_matrix_preconditioner`).
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax

from .matrix_routing import should_use_matrix_preconditioner


def _shape_factor(x: jnp.ndarray) -> jnp.ndarray:
    """Muon's shape scale for a (fan_in, fan_out) kernel: sqrt(max(1, out/in))."""
    fan_in = x.shape[0]
    fan_out = x.shape[1]
    return jnp.sqrt(jnp.maximum(1.0, jnp.asarray(fan_out / fan_in, dtype=x.dtype)))


def _kaon_orthogonalize(x: jnp.ndarray, key: jax.Array, eps: float, sv_dist: str) -> jnp.ndarray:
    """Return ``U diag(s_rand) V^T`` for the momentum matrix ``x``.

    Mirrors Muon's per-matrix normalization (transpose so rows<=cols, divide by
    Frobenius norm) but swaps the near-unit singular values for random positive
    ones with unit RMS.
    """
    transposed = x.shape[0] > x.shape[1]
    xm = x.T if transposed else x
    xm = xm / (jnp.linalg.norm(xm) + eps)
    u, _s, vt = jnp.linalg.svd(xm, full_matrices=False)
    r = u.shape[1]
    if sv_dist == "exp":
        s_rand = jax.random.exponential(key, (r,), dtype=xm.dtype)
    else:  # "halfnormal" -> |N(0,1)|
        s_rand = jnp.abs(jax.random.normal(key, (r,), dtype=xm.dtype))
    s_rand = s_rand / (jnp.sqrt(jnp.mean(jnp.square(s_rand))) + eps)
    out = (u * s_rand) @ vt
    return out.T if transposed else out


class KaonState(NamedTuple):
    count: jnp.ndarray
    mu: optax.Updates
    rng_key: jax.Array


def scale_by_kaon(
    beta: float = 0.95,
    eps: float = 1e-8,
    nesterov: bool = True,
    seed: int = 0,
    sv_dist: str = "halfnormal",
) -> optax.GradientTransformation:
    """Kaon core transform for the routed matrix parameters (2-D leaves only)."""

    def init_fn(params: optax.Params) -> KaonState:
        return KaonState(
            count=jnp.zeros([], jnp.int32),
            mu=jtu.tree_map(jnp.zeros_like, params),
            rng_key=jax.random.PRNGKey(int(seed)),
        )

    def update_fn(updates, state: KaonState, params=None):
        count_inc = optax.safe_increment(state.count)
        mu = jtu.tree_map(lambda m, g: beta * m + (1.0 - beta) * g, state.mu, updates)

        bc = 1.0 - beta ** count_inc.astype(jnp.float32)
        if nesterov:
            bc_next = 1.0 - beta ** (count_inc.astype(jnp.float32) + 1.0)
            mu_hat = jtu.tree_map(
                lambda m, g: beta * (m / bc_next) + (1.0 - beta) * (g / bc), mu, updates
            )
        else:
            mu_hat = jtu.tree_map(lambda m: m / bc, mu)

        leaves, treedef = jtu.tree_flatten(mu_hat)
        new_key, sub = jax.random.split(state.rng_key)
        leaf_keys = list(jax.random.split(sub, len(leaves))) if leaves else []
        new_leaves = [
            _kaon_orthogonalize(x, k, eps, sv_dist) for x, k in zip(leaves, leaf_keys)
        ]
        ortho = jtu.tree_unflatten(treedef, new_leaves)
        scaled = jtu.tree_map(lambda x: _shape_factor(x) * x, ortho)
        return scaled, KaonState(count=count_inc, mu=mu, rng_key=new_key)

    return optax.GradientTransformation(init_fn, update_fn)


def kaon(
    learning_rate: float,
    weight_decay: float = 0.0,
    *,
    beta: float = 0.95,
    eps: float = 1e-8,
    nesterov: bool = True,
    seed: int = 0,
    sv_dist: str = "halfnormal",
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps: float = 1e-8,
    adam_weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Kaon on routed matrices, AdamW on everything else.

    ``seed`` seeds the per-step random singular values; ``weight_decay`` is
    decoupled (multiplied by the learning rate) on the matrix parameters, matching
    Muon's convention.
    """
    kaon_tx = optax.chain(
        scale_by_kaon(beta=beta, eps=eps, nesterov=nesterov, seed=seed, sv_dist=sv_dist),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_learning_rate(learning_rate),
    )
    adam_tx = optax.adamw(
        learning_rate=learning_rate,
        b1=adam_b1,
        b2=adam_b2,
        eps=adam_eps,
        weight_decay=adam_weight_decay,
    )

    def label_fn(params):
        return jtu.tree_map_with_path(
            lambda path, p: "kaon" if should_use_matrix_preconditioner(path, p) else "adam",
            params,
        )

    return optax.multi_transform({"kaon": kaon_tx, "adam": adam_tx}, label_fn)
