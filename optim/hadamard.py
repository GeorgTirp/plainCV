"""Hadamard-Signum: Signum in a fixed random orthonormal (Walsh-Hadamard) frame.

For each matrix-shaped parameter ``W in R^{m x n}`` selected by the shared matrix
routing, we take the sign of the (Nesterov-)momentum buffer in a *fixed* random
orthonormal frame ``R`` that is incoherent with the coordinate axes (hence with
curvature), instead of in the coordinate frame as plain Signum does::

    R x     = (1/sqrt(N)) * WHT_N( D .* pad(x, N) )      # orthonormal on R^N
    u_rot   = R^T( sign( R( flatten(m) ) ) )
    Delta_W = -eta * reshape(u_rot[:m*n], (m, n))

with ``N = next_power_of_two(m*n)``, ``D in {+1,-1}^N`` a random sign diagonal
that is *fixed for the whole run* (seeded from ``(run_seed, param_path)``), and
``WHT_N`` the fast Walsh-Hadamard transform (O(N log N) butterfly, no dense
matrix). Because ``R^T R = I`` and the frame is drawn once and never re-drawn,
the coherence between the sign-allocation and curvature is destroyed while the
sign non-linearity, momentum and step-size are held identical to Signum.

Non-routed parameters (norms, embeddings, output head, 1-D params) fall back to
plain Signum (sign in the coordinate frame), so this arm differs from Signum
*only* by the frame applied to the routed hidden matrices -- the intended causal
control. ``run_seed`` (the config ``seed``) makes ``D`` reproducible per run.
"""

from __future__ import annotations

import hashlib
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import optax

from .matrix_routing import should_use_matrix_preconditioner, path_to_name


def next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def fwht(x: jnp.ndarray) -> jnp.ndarray:
    """Fast Walsh-Hadamard transform of a 1-D array whose length is a power of 2.

    Unnormalized: applying it twice yields ``N * x``. Implemented as an unrolled
    reshape/butterfly, so ``N`` must be statically known (it always is here, since
    it is derived from a parameter shape). Self-transpose: the matrix is symmetric.
    """
    N = x.shape[0]
    h = 1
    while h < N:
        x = x.reshape(N // (2 * h), 2, h)
        a = x[:, 0, :]
        b = x[:, 1, :]
        x = jnp.concatenate([a + b, a - b], axis=1).reshape(N)
        h *= 2
    return x


def _stable_leaf_seed(path) -> int:
    """Deterministic 32-bit seed from a parameter path, stable across processes."""
    name = path_to_name(path)
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), "little")


def _sign_diag(base_key: jax.Array, path, N: int) -> jnp.ndarray:
    """Fixed +/-1 diagonal for one parameter, independent of the training step."""
    leaf_key = jax.random.fold_in(base_key, _stable_leaf_seed(path))
    return jax.random.rademacher(leaf_key, (N,), dtype=jnp.float32)


def _rotated_sign_update(flat_dir: jnp.ndarray, d: jnp.ndarray, mn: int, N: int) -> jnp.ndarray:
    """Return u_rot = R^T( sign( R( flat_dir ) ) )[:mn], the Hadamard-Signum step.

    ``R x = (1/sqrt(N)) WHT(D .* pad(x))`` is orthonormal, so
    ``R^T y = D .* ((1/sqrt(N)) WHT(y))``.
    """
    inv_sqrt_n = jnp.asarray(1.0 / (N ** 0.5), dtype=flat_dir.dtype)
    x = jnp.pad(flat_dir, (0, N - mn))
    r = inv_sqrt_n * fwht(d.astype(flat_dir.dtype) * x)          # R x
    s = jnp.sign(r)
    u = d.astype(flat_dir.dtype) * (inv_sqrt_n * fwht(s))        # R^T s
    return u[:mn]


class HadamardSignumState(NamedTuple):
    momentum_buffer: optax.Updates


def hadamard_signum(
    learning_rate: float,
    momentum: float = 0.9,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    seed: int = 0,
) -> optax.GradientTransformation:
    """Signum with the sign taken in a fixed random Walsh-Hadamard frame.

    Same momentum buffer, Nesterov option and decoupled weight decay as
    :func:`optim.signum.signum`; the only difference is that routed matrix
    parameters take the sign in the random orthonormal frame ``R`` rather than in
    the coordinate frame. ``seed`` should be the run seed so the frame is
    reproducible per run.
    """
    if learning_rate < 0.0:
        raise ValueError(f"learning_rate must be >= 0, got {learning_rate}.")
    if momentum < 0.0 or momentum >= 1.0:
        raise ValueError(f"momentum must be in [0, 1), got {momentum}.")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}.")

    one_minus_momentum = 1.0 - momentum
    base_key = jax.random.PRNGKey(int(seed))

    def init_fn(params: optax.Params) -> HadamardSignumState:
        return HadamardSignumState(momentum_buffer=jtu.tree_map(jnp.zeros_like, params))

    def update_fn(
        updates: optax.Updates,
        state: HadamardSignumState,
        params: Optional[optax.Params] = None,
    ) -> tuple[optax.Updates, HadamardSignumState]:
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

        def sign_leaf(path, d_leaf):
            if should_use_matrix_preconditioner(path, d_leaf):
                mn = int(d_leaf.size)
                N = next_power_of_two(mn)
                dsign = _sign_diag(base_key, path, N)
                flat = d_leaf.reshape((mn,))
                u = _rotated_sign_update(flat, dsign, mn, N)
                return u.reshape(d_leaf.shape).astype(d_leaf.dtype)
            return jnp.sign(d_leaf)

        signed_updates = jtu.tree_map_with_path(sign_leaf, direction)

        if weight_decay > 0.0:
            if params is None:
                raise ValueError("hadamard_signum with weight_decay requires current params.")
            signed_updates = jtu.tree_map(
                lambda u, p: u + weight_decay * p, signed_updates, params
            )

        scaled_updates = jtu.tree_map(lambda u: -learning_rate * u, signed_updates)
        return scaled_updates, HadamardSignumState(momentum_buffer=momentum_buffer)

    return optax.GradientTransformation(init_fn, update_fn)
