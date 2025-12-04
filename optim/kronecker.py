from typing import Any, Callable, Tuple, Optional

import jax
import jax.numpy as jnp
import jax.nn as jnn
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
    """Return a closure v ↦ (R ⊗ L) v for fixed factors."""
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


# ---------------------------------------------------------------------------
# Kronecker-based curvature from per-example gradients
# ---------------------------------------------------------------------------

def make_kronecker_factors_fn(
    model_def,
    curvature_batch: Tuple[Array, Array],
    batch_stats: Optional[PyTree] = None,
    damping: float = 1e-6,
):
    """
    Return a function `factors_fn(params) -> factors_tree` where each leaf is
    (L, R) giving a Kronecker block H_block ≈ R ⊗ L, constructed from
    per-example gradients on the curvature batch.

    Muon-style reshape:
      - scalar:      (1, 1)
      - bias (1D):   (1, n)
      - ndim >= 2:   fan_in = prod(shape[:-1]), fan_out = shape[-1]
    """
    images, labels = curvature_batch

    def single_loss(params, x, y):
        variables = {"params": params}
        if batch_stats is not None:
            variables["batch_stats"] = batch_stats

        # For curvature probes we don't want to mutate BatchNorm stats; use eval mode.
        logits = model_def.apply(
            variables,
            x[None, ...],
            train=False,
            mutable=False,
        )[0]
        num_classes = logits.shape[-1]
        one_hot = jnn.one_hot(y, num_classes)
        log_probs = jnn.log_softmax(logits)
        return -jnp.sum(one_hot * log_probs)

    per_example_grad_fn = jax.jit(
        jax.vmap(jax.grad(single_loss), in_axes=(None, 0, 0))
    )

    def factors_fn(params: PyTree) -> PyTree:
        grads = per_example_grad_fn(params, images, labels)
        # grads: PyTree, each leaf has shape (B, *param_shape)

        def leaf_factors(param_leaf, grad_leaf):
            B = grad_leaf.shape[0]
            param_shape = param_leaf.shape

            if param_leaf.ndim == 0:
                fan_in, fan_out = 1, 1
                g_mat = grad_leaf.reshape(B, fan_in, fan_out)
            elif param_leaf.ndim == 1:
                n = param_shape[0]
                fan_in, fan_out = 1, n
                g_mat = grad_leaf.reshape(B, fan_in, fan_out)
            else:
                fan_out = param_shape[-1]
                fan_in = int(jnp.prod(jnp.array(param_shape[:-1])))
                g_mat = grad_leaf.reshape(B, fan_in, fan_out)

            L = jnp.einsum("bik,bjk->ij", g_mat, g_mat) / B
            R = jnp.einsum("bki,bkj->ij", g_mat, g_mat) / B

            L = L + damping * jnp.eye(fan_in, dtype=L.dtype)
            R = R + damping * jnp.eye(fan_out, dtype=R.dtype)

            return (L, R)

        return jtu.tree_map(leaf_factors, params, grads)

    return factors_fn


def make_kronecker_ggn_matvec_fn(
    model_def,
    curvature_batch: Tuple[Array, Array],
    batch_stats: Optional[PyTree] = None,
    damping: float = 1e-6,
) -> Callable[[PyTree, PyTree, Array], PyTree]:
    """
    Build a GGN-style matvec_fn(params, v_pytree, rng) using a Kronecker
    approximation per parameter leaf.

    Under the hood this just uses make_kronecker_factors_fn + tree_kronecker_matvec.
    """
    factors_fn = make_kronecker_factors_fn(
        model_def=model_def,
        curvature_batch=curvature_batch,
        batch_stats=batch_stats,
        damping=damping,
    )

    def ggn_matvec_fn(params: PyTree, vec_pytree: PyTree, rng: Array) -> PyTree:
        factors_tree = factors_fn(params)
        return tree_kronecker_matvec(factors_tree, vec_pytree)

    return ggn_matvec_fn
