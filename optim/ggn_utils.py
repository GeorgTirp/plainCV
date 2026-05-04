# optim/ggn_utils.py
import functools
from typing import Any, Callable, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import tree_util as jtu
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
    logits = logits.astype(jnp.float32)
    vec_logits = vec_logits.astype(jnp.float32)

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


def _extract_logits(output: Any) -> Array:
    """Robustly extract logits from a model output."""
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        # Flax HF models sometimes return a FrozenDict with "logits"
        if hasattr(output, "get"):
            return output.get("logits")
    return output


def make_ggn_matvec_fn_lm(
    model_def: Any,
    curvature_batch: Tuple[Array, Array, Optional[Array]],
    batch_stats: Optional[PyTree] = None,
) -> GGNMatvecFn:
    """Build a GGN matvec function for a Flax causal LM with softmax CE.

    The LM logits are (B, T, V) and labels are (B, T).
    We flatten (B, T) into a single batch dimension for the Hessian.

    Args:
      model_def: Flax Linen Module for LM.
      curvature_batch: (input_ids, labels, attn_mask).
        input_ids: (B, T) int32
        labels: (B, T) int32
        attn_mask: (B, T, T) bool/additive or None
      batch_stats: optional BatchNorm state (typically None for LM).
    """
    input_ids, labels, attn_mask = curvature_batch

    def _single_batch_ggn_matvec(
        params: Params,
        vec_pytree: PyTree,
        batch_input_ids: Array,
        batch_labels: Array,
        batch_attn_mask: Optional[Array],
    ) -> PyTree:
        def logits_fn(params: Params) -> Array:
            variables = {"params": params}
            if batch_stats is not None:
                variables["batch_stats"] = batch_stats

            out = model_def.apply(
                variables,
                batch_input_ids,
                attn_mask=batch_attn_mask,
                deterministic=True,
            )
            return _extract_logits(out)

        logits, jvp_logits = jax.jvp(
            logits_fn,
            (params,),
            (vec_pytree,),
        )  # (B, T, V)

        b, t, v = logits.shape
        logits2 = logits.reshape(b * t, v)
        jvp2 = jvp_logits.reshape(b * t, v)

        hv2 = softmax_cross_entropy_hessian_vec(
            logits=logits2,
            labels=batch_labels.reshape(b * t),
            vec_logits=jvp2,
        )
        # Match the training objective, which averages CE over all tokens.
        hv_logits = hv2.reshape(b, t, v) / jnp.asarray(b * t, dtype=hv2.dtype)
        hv_logits = hv_logits.astype(logits.dtype)

        _, vjp_fun = jax.vjp(logits_fn, params)
        hv_params, = vjp_fun(hv_logits)
        return hv_params

    def ggn_matvec(params: Params, vec_pytree: PyTree, rng_key: Array) -> PyTree:
        del rng_key
        if input_ids.ndim == 2:
            return _single_batch_ggn_matvec(params, vec_pytree, input_ids, labels, attn_mask)

        num_probe_batches = int(input_ids.shape[0])
        hvp_sum = None
        for i in range(num_probe_batches):
            batch_attn_mask = None if attn_mask is None else attn_mask[i]
            hvp_i = _single_batch_ggn_matvec(
                params,
                vec_pytree,
                input_ids[i],
                labels[i],
                batch_attn_mask,
            )
            hvp_sum = hvp_i if hvp_sum is None else jtu.tree_map(
                lambda acc, x: acc + x,
                hvp_sum,
                hvp_i,
            )
        scale = jnp.asarray(num_probe_batches, dtype=jnp.float32)
        return jtu.tree_map(lambda x: x / scale, hvp_sum)

    return jax.jit(ggn_matvec)


def make_hessian_matvec_fn_lm(
    model_def: Any,
    curvature_batch: Tuple[Array, Array, Optional[Array]],
    batch_stats: Optional[PyTree] = None,
) -> HessianMatvecFn:
    """Build a signed Hessian matvec for a Flax causal LM."""
    input_ids, labels, attn_mask = curvature_batch

    def _single_batch_loss(
        params: Params,
        batch_input_ids: Array,
        batch_labels: Array,
        batch_attn_mask: Optional[Array],
    ) -> Array:
        variables = {"params": params}
        if batch_stats is not None:
            variables["batch_stats"] = batch_stats

        out = model_def.apply(
            variables,
            batch_input_ids,
            attn_mask=batch_attn_mask,
            deterministic=True,
        )
        logits = _extract_logits(out).astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(logits, batch_labels).mean()

    def loss_fn(params: Params) -> Array:
        if input_ids.ndim == 2:
            return _single_batch_loss(params, input_ids, labels, attn_mask)

        num_probe_batches = int(input_ids.shape[0])
        loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
        for i in range(num_probe_batches):
            batch_attn_mask = None if attn_mask is None else attn_mask[i]
            loss_sum = loss_sum + _single_batch_loss(
                params,
                input_ids[i],
                labels[i],
                batch_attn_mask,
            )
        return loss_sum / jnp.asarray(num_probe_batches, dtype=jnp.float32)

    def hvp_fn(params: Params, vec_pytree: PyTree, rng_key: Array) -> PyTree:
        del rng_key
        _, hvp = jax.jvp(
            jax.grad(loss_fn),
            (params,),
            (vec_pytree,),
        )
        return hvp

    return jax.jit(hvp_fn)


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
        hv_logits = hv_logits / jnp.asarray(logits.shape[0], dtype=hv_logits.dtype)

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


# Add to optim/ggn_utils.py

def _build_weighted_laplacian_from_probs(
    p: Array,
    adjacency: Array,
    eps: float = 1e-8,
) -> Array:
    """
    Build a probability-dependent weighted Laplacian L(p) on the class simplex.

    We use a simple (common) choice:
      w_ij(p) = a_ij * (p_i + p_j)/2
      L_ij = -w_ij for i != j
      L_ii = sum_{j != i} w_ij

    Args:
      p: (C,) probability vector, p_i >= 0, sum p_i = 1
      adjacency: (C, C) symmetric nonnegative weights a_ij, diag assumed 0
      eps: small stabilizer

    Returns:
      L: (C, C) Laplacian matrix (singular; nullspace is constants).
    """
    # Symmetrize and clear diagonal for safety
    a = 0.5 * (adjacency + adjacency.T)
    a = a * (1.0 - jnp.eye(a.shape[0], dtype=a.dtype))

    # w_ij = a_ij * (p_i + p_j)/2
    w = a * 0.5 * (p[:, None] + p[None, :])

    # Laplacian: L = diag(sum_j w_ij) - w
    d = jnp.sum(w, axis=-1)
    L = jnp.diag(d) - w

    # Small eps on diagonal (doesn't remove nullspace, just numerical safety)
    L = L + eps * jnp.eye(L.shape[0], dtype=L.dtype)
    return L


def _solve_laplacian_gauge_fixed(
    L: Array,
    b: Array,
) -> Array:
    """
    Solve L x = b with a gauge-fixing constraint to handle Laplacian singularity.

    We fix the additive constant by enforcing sum(x)=0:
      Replace last row with ones: 1^T x = 0
      Replace last rhs entry with 0

    Args:
      L: (C, C) Laplacian-like matrix (near-singular).
      b: (C,) right-hand side with sum(b)=0 (tangent to simplex).

    Returns:
      x: (C,) solution with approximately zero mean.
    """
    C = L.shape[0]
    A = L
    rhs = b

    # Gauge-fix: last row becomes all-ones; rhs last entry = 0
    A = A.at[-1, :].set(jnp.ones((C,), dtype=L.dtype))
    rhs = rhs.at[-1].set(jnp.array(0.0, dtype=rhs.dtype))

    x = jnp.linalg.solve(A, rhs)
    # Enforce zero-mean numerically
    x = x - jnp.mean(x)
    return x


def make_wasserstein_metric_matvec_fn(
    model_def: Any,
    curvature_batch: Tuple[Array, Array],
    batch_stats: Optional[PyTree] = None,
    *,
    # Provide either adjacency directly, or a cost/distance matrix to convert.
    class_adjacency: Optional[Array] = None,
    class_cost: Optional[Array] = None,
    cost_to_adj_eps: float = 1e-6,
    laplacian_eps: float = 1e-8,
) -> GGNMatvecFn:
    """
    Build a Wasserstein-metric-tensor matvec (GW) for a Flax classifier.

    GW is a PSD operator on parameter space induced by a discrete Wasserstein
    geometry on the output simplex:
        GW(θ) v = J_p(θ)^T  L(p)^{-1}  J_p(θ) v
    where p = softmax(logits), and L(p) is a probability-weighted graph Laplacian
    built from a ground geometry across classes.

    This matches the same calling convention as the other curvature matvecs:
        gw_matvec(params, vec_pytree, rng_key) -> vec_pytree

    Args:
      model_def: Flax Linen Module.
      curvature_batch: (images, labels) used for curvature estimation. Labels
        are unused here but kept for a consistent signature.
      batch_stats: BatchNorm stats or None.
      class_adjacency: (C, C) nonnegative symmetric weights a_ij encoding
        class geometry. diag should be 0.
      class_cost: (C, C) symmetric nonnegative costs/distances. If provided and
        class_adjacency is None, we set adjacency = 1 / (cost^2 + eps).
      cost_to_adj_eps: epsilon for converting cost->adjacency.
      laplacian_eps: epsilon added to L(p) diagonal for numerical stability.

    Returns:
      gw_matvec(params, vec_pytree, rng_key) -> PyTree like params
    """
    images, _labels = curvature_batch  # labels not needed for the metric tensor

    def probs_fn(params: Params, rng: Array) -> Array:
        """Forward pass returning class probabilities p = softmax(logits)."""
        variables = {"params": params}
        if batch_stats is not None:
            variables["batch_stats"] = batch_stats

        out = model_def.apply(
            variables,
            images,
            mutable=False,
            train=False,
            rngs={"dropout": rng},  # safe if model has no dropout
        )

        logits = out[0] if isinstance(out, tuple) else out
        return jax.nn.softmax(logits, axis=-1)  # (B, C)

    # Determine number of classes and build adjacency
    # (we do this lazily inside the matvec to keep JIT happy with shapes).
    def _get_adjacency(num_classes: int, dtype) -> Array:
        if class_adjacency is not None:
            A = class_adjacency
        elif class_cost is not None:
            # Convert cost/distance matrix to adjacency weights
            # Larger cost => smaller coupling weight.
            A = 1.0 / (jnp.square(class_cost) + cost_to_adj_eps)
        else:
            # Default: fully-connected unit adjacency (no geometry info)
            A = jnp.ones((num_classes, num_classes), dtype=dtype)

        A = A.astype(dtype)
        # Symmetrize & clear diagonal
        A = 0.5 * (A + A.T)
        A = A * (1.0 - jnp.eye(num_classes, dtype=dtype))
        return A

    def gw_matvec(params: Params, vec_pytree: PyTree, rng_key: Array) -> PyTree:
        # Wrap probs_fn so jvp/vjp are wrt params only (rng fixed)
        f = lambda p: probs_fn(p, rng_key)

        # 1) JVP: compute s = J_p v (tangent in simplex), shape (B, C)
        p, s = jax.jvp(f, (params,), (vec_pytree,))

        # Ensure tangent numerically: sum_c s_c = 0
        s = s - jnp.mean(s, axis=-1, keepdims=True)

        B, C = p.shape
        A = _get_adjacency(C, p.dtype)

        # 2) Solve per example: phi = L(p)^{-1} s  (with gauge fixing)
        def solve_one(p_i: Array, s_i: Array) -> Array:
            L = _build_weighted_laplacian_from_probs(p_i, A, eps=laplacian_eps)
            phi = _solve_laplacian_gauge_fixed(L, s_i)
            return phi

        phi = jax.vmap(solve_one, in_axes=(0, 0))(p, s)  # (B, C)

        # 3) VJP: map back to parameter space: J_p^T phi
        _, vjp_fun = jax.vjp(f, params)
        gw_v, = vjp_fun(phi)
        return gw_v

    return jax.jit(gw_matvec)


def _softmax_jacobian_vec_from_probs(p: Array, v: Array) -> Array:
    """Apply S(p)v = (diag(p) - pp^T)v."""
    inner = jnp.sum(p * v, axis=-1, keepdims=True)
    return p * (v - inner)


def _build_local_embedding_laplacian(
    p: Array,
    embeds: Array,
    *,
    graph_temp: float,
    graph_eps: float,
) -> Array:
    """Build K x K probability-weighted graph Laplacian on top-k token support."""
    p = p.astype(jnp.float32)
    embeds = embeds.astype(jnp.float32)

    diffs = embeds[:, None, :] - embeds[None, :, :]
    dist2 = jnp.sum(diffs * diffs, axis=-1)

    if graph_temp <= 0.0:
        tau = jnp.maximum(jnp.mean(dist2), graph_eps)
    else:
        tau = jnp.asarray(graph_temp, dtype=jnp.float32)

    A = jnp.exp(-dist2 / tau)
    K = A.shape[0]
    A = A * (1.0 - jnp.eye(K, dtype=A.dtype))

    # Mobility choice: arithmetic mean of endpoint masses.
    W = A * 0.5 * (p[:, None] + p[None, :])

    degree = jnp.sum(W, axis=-1)
    L = jnp.diag(degree) - W
    return L


def _solve_laplacian_symmetric_gauge(
    L: Array,
    b: Array,
    *,
    ridge: float,
) -> Array:
    """
    Symmetric gauge-fixed approximation to L^dagger b.

    We solve:
        (L + ridge I + 11^T/K) x = b,
    then remove the mean of x.
    """
    K = L.shape[0]
    dtype = L.dtype

    b = b - jnp.mean(b)

    I = jnp.eye(K, dtype=dtype)
    gauge = jnp.ones((K, K), dtype=dtype) / jnp.asarray(K, dtype=dtype)

    A = L + jnp.asarray(ridge, dtype=dtype) * I + gauge
    x = jnp.linalg.solve(A, b)
    x = x - jnp.mean(x)
    return x


def make_wasserstein_metric_matvec_fn_lm(
    model_def: Any,
    curvature_batch: Tuple[Array, Array, Optional[Array]],
    token_embedding_matrix: Array,
    batch_stats: Optional[PyTree] = None,
    *,
    top_k: int = 128,
    graph_temp: float = 0.0,
    graph_eps: float = 1e-6,
    laplacian_ridge: float = 1e-4,
    ignore_label: int = -100,
) -> GGNMatvecFn:
    """
    Build a matrix-vector product for the Wasserstein/Otto pullback metric.

    Operator:
        W(θ) v =
            J_z^T S(p)^T L(p)^dagger S(p) J_z v

    where:
      - z are LM logits,
      - p is the restricted top-k softmax distribution,
      - S(p) = diag(p) - pp^T,
      - L(p) is a probability-weighted graph Laplacian over top-k token embeddings.

    This is not a loss Hessian. It is a Wasserstein pullback metric-vector product,
    suitable for Lanczos exactly like your GGN/Hessian matvecs.

    Args:
      model_def: Flax LM module.
      curvature_batch: (input_ids, labels, attn_mask).
        input_ids: (B, T) or (P, B, T)
        labels: (B, T) or (P, B, T), used only for masking ignore_label.
        attn_mask: optional attention mask, matching your model.
      token_embedding_matrix: (V, D) token embeddings defining ground geometry.
      top_k: vocabulary support size per token position.
      graph_temp: RBF temperature for token graph. If <= 0, uses mean pairwise dist2.
      graph_eps: numerical epsilon for graph temperature.
      laplacian_ridge: ridge used in symmetric gauge-fixed Laplacian solve.
      ignore_label: labels equal to this are excluded from the average.

    Returns:
      wasserstein_matvec(params, vec_pytree, rng_key) -> PyTree like params.
    """
    input_ids, labels, attn_mask = curvature_batch
    token_embedding_matrix = token_embedding_matrix.astype(jnp.float32)

    def _single_batch_matvec(
        params: Params,
        vec_pytree: PyTree,
        batch_input_ids: Array,
        batch_labels: Array,
        batch_attn_mask: Optional[Array],
    ) -> PyTree:

        def logits_fn(p: Params) -> Array:
            variables = {"params": p}
            if batch_stats is not None:
                variables["batch_stats"] = batch_stats

            out = model_def.apply(
                variables,
                batch_input_ids,
                attn_mask=batch_attn_mask,
                deterministic=True,
            )
            return _extract_logits(out)

        logits, jvp_logits = jax.jvp(
            logits_fn,
            (params,),
            (vec_pytree,),
        )  # (B, T, V)

        B, T, V = logits.shape
        N = B * T

        logits2 = logits.reshape(N, V).astype(jnp.float32)
        jvp2 = jvp_logits.reshape(N, V).astype(jnp.float32)
        labels2 = batch_labels.reshape(N)

        valid = (labels2 != ignore_label).astype(jnp.float32)  # (N,)
        denom = jnp.maximum(jnp.sum(valid), 1.0)

        # Top-k predictive support per token position.
        top_vals, top_idx = jax.lax.top_k(logits2, top_k)  # (N, K), (N, K)
        dz_top = jnp.take_along_axis(jvp2, top_idx, axis=-1)  # (N, K)

        # Restricted simplex probabilities.
        p_top = jax.nn.softmax(top_vals, axis=-1)  # (N, K)

        def apply_one(p_i: Array, dz_i: Array, idx_i: Array, valid_i: Array) -> Array:
            # δp = S(p) J_z v
            dp_i = _softmax_jacobian_vec_from_probs(p_i, dz_i)

            embeds_i = token_embedding_matrix[idx_i]  # (K, D)

            L_i = _build_local_embedding_laplacian(
                p_i,
                embeds_i,
                graph_temp=graph_temp,
                graph_eps=graph_eps,
            )

            # φ = L(p)^dagger δp
            phi_i = _solve_laplacian_symmetric_gauge(
                L_i,
                dp_i,
                ridge=laplacian_ridge,
            )

            # S(p)^T φ = S(p)φ since S is symmetric.
            az_i = _softmax_jacobian_vec_from_probs(p_i, phi_i)

            return valid_i * az_i

        az_top = jax.vmap(apply_one, in_axes=(0, 0, 0, 0))(
            p_top,
            dz_top,
            top_idx,
            valid,
        )  # (N, K)

        # Scatter top-k cotangents back to full vocab logits.
        az_full = jnp.zeros_like(logits2)
        az_full = az_full.at[
            jnp.arange(N)[:, None],
            top_idx,
        ].add(az_top)

        # Match mean-over-valid-token scaling.
        az_full = az_full / denom
        az_full = az_full.reshape(B, T, V).astype(logits.dtype)

        _, vjp_fun = jax.vjp(logits_fn, params)
        out_params, = vjp_fun(az_full)
        return out_params

    def wasserstein_matvec(params: Params, vec_pytree: PyTree, rng_key: Array) -> PyTree:
        del rng_key

        if input_ids.ndim == 2:
            return _single_batch_matvec(
                params,
                vec_pytree,
                input_ids,
                labels,
                attn_mask,
            )

        num_probe_batches = int(input_ids.shape[0])
        mv_sum = None

        for i in range(num_probe_batches):
            batch_attn_mask = None if attn_mask is None else attn_mask[i]
            mv_i = _single_batch_matvec(
                params,
                vec_pytree,
                input_ids[i],
                labels[i],
                batch_attn_mask,
            )
            mv_sum = mv_i if mv_sum is None else jtu.tree_map(
                lambda a, b: a + b,
                mv_sum,
                mv_i,
            )

        scale = jnp.asarray(num_probe_batches, dtype=jnp.float32)
        return jtu.tree_map(lambda x: x / scale, mv_sum)

    return jax.jit(wasserstein_matvec)