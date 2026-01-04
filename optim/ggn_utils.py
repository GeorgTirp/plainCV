# optim/ggn_utils.py
import functools
from typing import Any, Callable, Tuple, Optional

import jax
import jax.numpy as jnp
import optax

Array = jax.Array
Params = Any
PyTree = Any

# Type: matvec callable (params, vec_pytree, rng_key) -> vec_pytree
GGNMatvecFn = Callable[[Params, PyTree, Array], PyTree]
HessianMatvecFn = Callable[[Params, PyTree, Array], PyTree]


def softmax_cross_entropy_hessian_vec(
    logits: Array,
    labels: Array,
    vec_logits: Array,
) -> Array:
    """Apply Hessian of softmax cross-entropy wrt logits to a vector.

    Args:
      logits: (B, C) pre-softmax logits.
      labels: (B,) int32/int64 class indices.
      vec_logits: (B, C) direction to multiply Hessian with.

    Returns:
      Hv_logits: (B, C) Hessian(logits) @ vec_logits
    """
    # probabilities
    probs = jax.nn.softmax(logits, axis=-1)  # (B, C)

    # For cross-entropy loss ℓ = -log p_y, Hessian wrt logits:
    # H = diag(p) - p p^T, applied per sample.
    # For each sample b:
    #   H_b v_b = diag(p_b) v_b - p_b (p_b^T v_b)

    # First term: diag(p) v  -> elementwise p * v
    diag_term = probs * vec_logits  # (B, C)

    # Second term: p (p^T v)
    inner = jnp.sum(probs * vec_logits, axis=-1, keepdims=True)  # (B, 1)
    outer_term = probs * inner  # (B, C)

    hv = diag_term - outer_term  # (B, C)
    return hv


def make_ggn_matvec_fn(
    model_def: Any,
    batch_stats: Any,
    curvature_batch: Tuple[Array, Array],
) -> GGNMatvecFn:
    """Build a GGN matvec function for a Flax classifier with softmax CE.

    This uses the classic Gauss-Newton / GGN construction for
    softmax-cross-entropy:

      H_GGN(θ) v = J^T H_ℓ J v,

    where:
      - J is the Jacobian of logits wrt params,
      - H_ℓ is the Hessian of the loss wrt logits.

    Args:
      model_def: Flax Linen Module, e.g. SmallResNet or MLP.
      batch_stats: BatchNorm / similar statistics PyTree or None.
      curvature_batch: (images, labels) used to estimate curvature.
        images: (B, H, W, C)
        labels: (B,)

    Returns:
      ggn_matvec(params, vec_pytree, rng_key) -> Hv_pytree
    """
    images, labels = curvature_batch

    def logits_fn(params: Params) -> Array:
        """Forward pass returning logits only (robust to extra outputs)."""
        variables = {"params": params}
        if batch_stats is not None:
            variables["batch_stats"] = batch_stats

            out = model_def.apply(
                variables,
                images,
                train=False,
                mutable=["batch_stats"],
            )
        else:
            out = model_def.apply(
                variables,
                images,
                train=False,
            )

        # Robustly extract logits as first element if apply returns a tuple
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        return logits

    # We will use jvp + vjp around logits_fn:
    #   1) jvp(logits_fn, (params,), (vec_pytree,)) -> (logits, J v)
    #   2) apply H_ℓ to J v
    #   3) vjp(logits_fn, params) on that result to get J^T H_ℓ J v

    def ggn_matvec(params: Params, vec_pytree: PyTree, rng_key: Array) -> PyTree:
        # 1. JVP: compute J v at logits level
        logits, jvp_logits = jax.jvp(
            logits_fn,
            (params,),
            (vec_pytree,),
        )  # both (B, C)

        # 2. Apply Hessian of CE wrt logits: H_ℓ (J v)
        hv_logits = softmax_cross_entropy_hessian_vec(
            logits=logits,
            labels=labels,
            vec_logits=jvp_logits,
        )  # (B, C)

        # 3. VJP: map back to parameter space: J^T (H_ℓ J v)
        _, vjp_fun = jax.vjp(logits_fn, params)
        hv_params, = vjp_fun(hv_logits)  # PyTree with same structure as params

        return hv_params

    # Optionally JIT-compile the matvec (useful since Lanczos calls it many times)
    ggn_matvec_jit = jax.jit(ggn_matvec)

    return ggn_matvec_jit


def make_hessian_matvec_fn(
    model_def: Any,
    curvature_batch: Tuple[Array, Array],
    batch_stats: Optional[PyTree] = None,
) -> HessianMatvecFn:
    """
    Build a Hessian-vector product function Hv for the training loss,
    using the fixed curvature_batch (images, labels).

    This matches the same calling convention as make_ggn_matvec_fn:
      hvp_fn(params, v_pytree, rng) -> PyTree with same structure as params.

    Internally we use the standard JAX hvp pattern:
        hvp(f, params, v) = jax.jvp(grad(f), (params,), (v,))[1]
    """
    images, labels = curvature_batch

    # --- 1. Define the scalar loss on this fixed batch ---
    # This should match your training loss (softmax cross-entropy).
    def loss_fn(params: Params, rng: Array) -> Array:
        """Scalar loss for this curvature batch."""
        variables = {"params": params}
        if batch_stats is not None:
            variables["batch_stats"] = batch_stats

        out = model_def.apply(
            variables,
            images,
            mutable=False,          # don't update batch_stats during curvature
            train=False,            # use eval mode for curvature estimation
            rngs={"dropout": rng},  # safe even if model has no dropout
        )

        # Robustly extract logits as first element if model returns a tuple
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        num_classes = logits.shape[-1]
        one_hot = jax.nn.one_hot(labels, num_classes=num_classes)
        xent = optax.softmax_cross_entropy(logits, one_hot).mean()
        return xent

    # --- 2. Wrap loss_fn so rng is fixed during the JVP/grad call ---
    def make_loss_with_rng(rng: Array):
        def loss_with_rng(p: Params) -> Array:
            return loss_fn(p, rng)
        return loss_with_rng

    # --- 3. Build the Hessian-vector product Hv ---
    def hvp_fn(params: Params, vec_pytree: PyTree, rng: Array) -> PyTree:
        """
        Compute H(params) @ vec_pytree using forward-over-reverse:
            hvp(f, params, v) = jax.jvp(jax.grad(f), (params,), (v,))[1]
        """
        loss_wrapped = make_loss_with_rng(rng)
        # jvp returns (grad, hvp); we only need hvp
        _, hvp = jax.jvp(
            jax.grad(loss_wrapped),
            (params,),
            (vec_pytree,),
        )
        return hvp

    return hvp_fn


def make_fisher_matvec_fn(
    model_def: Any,
    curvature_batch: Tuple[Array, Array],
    batch_stats: Optional[PyTree] = None,
) -> GGNMatvecFn:
    """
    Build an empirical Fisher-information matvec F v for the training loss
    on a fixed curvature_batch (images, labels).

    Empirical Fisher:
        F(θ) ≈ (1/B) Σ_i g_i g_i^T
    where g_i = ∇_θ ℓ_i(θ) is the per-example gradient of the
    softmax cross-entropy.

    Returns:
      fisher_matvec(params, vec_pytree, rng) -> PyTree with same structure as params.
    """
    images, labels = curvature_batch
    batch_size = images.shape[0]

    # ---- 1. Per-example loss (scalar) ----
    def loss_single(params: Params, image: Array, label: Array, rng: Array) -> Array:
        """Negative log-likelihood for a single (image, label)."""
        variables = {"params": params}
        if batch_stats is not None:
            variables["batch_stats"] = batch_stats

        out = model_def.apply(
            variables,
            image[None, ...],      # add batch dimension
            mutable=False,
            train=False,           # eval mode for curvature
            rngs={"dropout": rng}, # safe even if model has no dropout
        )

        # Extract logits robustly if model returns a tuple
        if isinstance(out, tuple):
            logits = out[0]  # (1, C)
        else:
            logits = out     # (1, C)

        logits = logits[0]   # (C,)
        num_classes = logits.shape[-1]
        one_hot = jax.nn.one_hot(label, num_classes=num_classes)  # (C,)

        # optax expects (..., C), so wrap back into batch dim
        xent = optax.softmax_cross_entropy(
            logits[None, :],
            one_hot[None, :],
        ).mean()
        return xent

    # ---- 2. Per-example gradients g_i(θ) ----
    def per_example_grads(params: Params, rng: Array) -> PyTree:
        grad_single = jax.grad(loss_single)
        # One RNG per example (if dropout etc. is present)
        rngs = jax.random.split(rng, batch_size)
        # vmap over (image, label, rng); params is shared
        grads = jax.vmap(
            grad_single,
            in_axes=(None, 0, 0, 0),
        )(params, images, labels, rngs)
        # grads: PyTree whose leaves have shape (B, ...)
        return grads

    # ---- 3. Empirical Fisher matvec: F v = (1/B) Σ_i g_i (g_i^T v) ----
    def fisher_matvec(params: Params, vec_pytree: PyTree, rng: Array) -> PyTree:
        grads = per_example_grads(params, rng)  # PyTree with leading batch dim

        # a) Compute per-example inner products alpha_i = <g_i, v>
        def per_leaf_dot(g_leaf: Array, v_leaf: Array) -> Array:
            # g_leaf: (B, ...), v_leaf: (...)
            # result: (B,)
            return jnp.einsum("i..., ...->i", g_leaf, v_leaf)

        per_leaf_dots = jax.tree_util.tree_map(per_leaf_dot, grads, vec_pytree)
        # Sum contributions across all leaves to get final alpha_i
        alphas = functools.reduce(
            lambda acc, x: acc + x,
            jax.tree_util.tree_leaves(per_leaf_dots),
            jnp.zeros((batch_size,), dtype=images.dtype),
        )  # shape (B,)

        # b) Weighted sum of per-example grads:
        #    F v = (1/B) Σ_i alpha_i g_i
        def combine_leaf(g_leaf: Array) -> Array:
            # g_leaf: (B, ...)
            # result: (...)
            return jnp.einsum("i, i...->...", alphas, g_leaf) / batch_size

        fisher_v = jax.tree_util.tree_map(combine_leaf, grads)
        return fisher_v

    # Optionally JIT, since Lanczos will call this many times
    fisher_matvec_jit = jax.jit(fisher_matvec)
    return fisher_matvec_jit
