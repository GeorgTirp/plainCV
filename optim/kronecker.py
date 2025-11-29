# optim/kronecker.py
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

Array = jax.Array
PyTree = Any

# For a single layer: curvature block ≈ R ⊗ L
KroneckerFactors = Tuple[Array, Array]  # (L, R)
MatvecFn = Callable[[Array], Array]


def build_kronecker_matrix(
    left: Array,
    right: Array,
    damping: float = 0.0,
) -> Array:
    """Return the full Kronecker-product matrix R ⊗ L.

    Args:
      left:  L ∈ R^{m x m}, e.g. "input-side" factor.
      right: R ∈ R^{n x n}, e.g. "output-side" factor.
      damping: Optional diagonal damping λ. We build
               (R + λ I_n) ⊗ (L + λ I_m).

    Returns:
      H ≈ R ⊗ L ∈ R^{(mn) x (mn)}.

    Note:
      By the identity vec(L X R^T) = (R ⊗ L) vec(X), this H acts on
      vec(W) by first reshaping to (m, n), then applying L on the left
      and R on the right.
    """
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(
            f"build_kronecker_matrix expects 2D factors, got "
            f"L.ndim={left.ndim}, R.ndim={right.ndim}"
        )

    m, m2 = left.shape
    n, n2 = right.shape
    if m != m2 or n != n2:
        raise ValueError(
            f"Factors must be square; got L.shape={left.shape}, "
            f"R.shape={right.shape}"
        )

    if damping != 0.0:
        left = left + damping * jnp.eye(m, dtype=left.dtype)
        right = right + damping * jnp.eye(n, dtype=right.dtype)

    # H = R ⊗ L so that H vec(X) = vec(L X R^T)
    return jnp.kron(right, left)


def kronecker_matvec(
    left: Array,
    right: Array,
    v: Array,
) -> Array:
    """Apply H ≈ R ⊗ L to a vector using the vec-trick.

    Args:
      left:  L ∈ R^{m x m}.
      right: R ∈ R^{n x n}.
      v:     Vector of length m * n, interpreted as vec(X) with X ∈ R^{m x n}.

    Returns:
      H v where H = R ⊗ L, reshaped back to a vector of size m * n.
    """
    m = left.shape[0]
    n = right.shape[0]

    if v.size != m * n:
        raise ValueError(
            f"kronecker_matvec: v.size={v.size} incompatible with "
            f"L.shape={left.shape}, R.shape={right.shape} "
            f"(expected {m*n})."
        )

    # Reshape vec(X) -> X ∈ R^{m x n}
    X = v.reshape(m, n)

    # Y = L X R^T, then vec(Y) = (R ⊗ L) vec(X)
    Y = left @ X @ right.T

    return Y.reshape(-1)


def make_kronecker_matvec_fn(
    left: Array,
    right: Array,
) -> MatvecFn:
    """Return a closure v ↦ (R ⊗ L) v for fixed factors.

    Useful when you want a matvec to feed into Lanczos/CG without
    materializing the full Kronecker matrix.
    """
    left = jnp.asarray(left)
    right = jnp.asarray(right)

    def mv(v: Array) -> Array:
        return kronecker_matvec(left, right, v)

    return mv


def tree_kronecker_matvec(
    factors_tree: PyTree,
    vec_tree: PyTree,
) -> PyTree:
    """Apply per-parameter Kronecker blocks to a PyTree of vectors.

    Args:
      factors_tree: PyTree matching the parameter structure for which
                    you have Kronecker factors; each leaf is (L, R).
      vec_tree:     PyTree of same structure, each leaf is an array
                    whose flattened length matches L.shape[0] * R.shape[0].

    Returns:
      PyTree of same structure with (R ⊗ L) vec applied leafwise.
    """
    def _leaf_mv(factors: KroneckerFactors, v: Array) -> Array:
        L, R = factors
        flat_v = v.reshape(-1)
        out = kronecker_matvec(L, R, flat_v)
        return out.reshape(v.shape)

    return jtu.tree_map(_leaf_mv, factors_tree, vec_tree)
