from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import tree_util as jtu
import optax

Array = jax.Array
Params = Any
PyTree = Any

# Type: given params, direction (same PyTree structure as params), rng -> matvec(direction)
GGNMatvecFn = Callable[[Params, PyTree, Array], PyTree]


# ---------------------------------------------------------------------------
# Lanczos iterative eigen solver on a matrix-vector product
# (copied from your existing file for convenience)
# ---------------------------------------------------------------------------

def lanczos(
    matvec: Callable[[Array], Array],
    dim: int,
    num_iter: int,
    key: Array,
    eps: float = 1e-6,
) -> Tuple[Array, Array]:
    """Run Lanczos to approximate top eigenvalues/eigenvectors.

    Args:
      matvec: function v -> A @ v (A is implicit symmetric PSD matrix, e.g. GGN).
      dim: dimension of the flattened parameter vector.
      num_iter: number of Lanczos iterations (also size of Krylov subspace).
      key: RNG key for initial vector.
      eps: small value to guard against breakdown.

    Returns:
      eigenvalues: (num_iter,) approximated eigenvalues (sorted descending).
      eigenvectors: (num_iter, dim) corresponding eigenvectors (in flattened space).
    """
    v0 = jax.random.normal(key, (dim,))
    v0 = v0 / (jnp.linalg.norm(v0) + eps)

    def body_fun(carry, i):
        V, alphas, betas = carry
        v = V[i]  # (dim,)

        w = matvec(v)  # (dim,)
        alpha = jnp.vdot(v, w)
        w = w - alpha * v

        def ortho_against_prev(_w, _i):
            prev_v = V[_i]
            proj = jnp.vdot(prev_v, _w)
            return _w - proj * prev_v

        # full reorthogonalization against previous basis vectors
        w = jax.lax.fori_loop(0, i, lambda j, ww: ortho_against_prev(ww, j), w)

        beta = jnp.linalg.norm(w)
        beta = jnp.where(beta < eps, 0.0, beta)
        next_v = jnp.where(beta > 0, w / (beta + eps), jnp.zeros_like(w))

        V = V.at[i + 1].set(next_v)
        alphas = alphas.at[i].set(alpha)
        betas = betas.at[i].set(beta)

        return (V, alphas, betas), None

    V = jnp.zeros((num_iter + 1, dim))
    V = V.at[0].set(v0)
    alphas = jnp.zeros((num_iter,))
    betas = jnp.zeros((num_iter,))

    (V, alphas, betas), _ = jax.lax.scan(
        body_fun,
        (V, alphas, betas),
        jnp.arange(num_iter),
    )

    k = num_iter
    T = jnp.diag(alphas)
    if k > 1:
        T = T.at[jnp.arange(k - 1), jnp.arange(1, k)].set(betas[: k - 1])
        T = T.at[jnp.arange(1, k), jnp.arange(k - 1)].set(betas[: k - 1])

    evals, evecs_T = jnp.linalg.eigh(T)  # ascending
    idx = jnp.argsort(evals)[::-1]
    evals = evals[idx]
    evecs_T = evecs_T[:, idx]

    V_k = V[:-1]  # (k, dim)
    eigenvectors_flat = (evecs_T.T @ V_k).reshape(k, dim)

    return evals, eigenvectors_flat


# ---------------------------------------------------------------------------
# Global curvature state (shared between EigenAdam / EigenMuon / curvature Muon)
# ---------------------------------------------------------------------------

class CurvatureState(NamedTuple):
    """Global curvature state: top eigenpairs of a GGN/Hessian-like operator."""
    step: Array
    eigenvalues: Array      # (max_eigenvectors,)
    eigenvectors: Array     # (max_eigenvectors, dim)
    rng_key: Array


def init_curvature_state(params: Params, max_eigenvectors: int) -> CurvatureState:
    """Initialize curvature state with zeros (no eigenbasis yet)."""
    flat_params, _ = ravel_pytree(params)
    dim = flat_params.shape[0]

    eigenvalues = jnp.zeros((max_eigenvectors,), dtype=jnp.float32)
    eigenvectors = jnp.zeros((max_eigenvectors, dim), dtype=jnp.float32)

    rng_key = jax.random.PRNGKey(0)
    step = jnp.array(0, dtype=jnp.int32)

    return CurvatureState(
        step=step,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        rng_key=rng_key,
    )


def maybe_update_curvature_state(
    state: CurvatureState,
    params: Params,
    ggn_matvec_fn: Optional[GGNMatvecFn],
    curvature_update_every: int,
    lanczos_iters: int,
) -> CurvatureState:
    """Optionally refresh curvature eigenbasis using Lanczos."""
    step = state.step + 1
    rng_key = state.rng_key
    eigenvalues = state.eigenvalues
    eigenvectors = state.eigenvectors

    if ggn_matvec_fn is None or params is None:
        return CurvatureState(step=step, eigenvalues=eigenvalues,
                              eigenvectors=eigenvectors, rng_key=rng_key)

    should_update = (step % curvature_update_every) == 0

    def do_update(carry):
        eigenvalues, eigenvectors, rng_key, params = carry

        flat_params, unravel_params = ravel_pytree(params)
        dim = flat_params.shape[0]

        def matvec_flat(v_flat: Array) -> Array:
            v_pytree = unravel_params(v_flat)
            Hv_pytree = ggn_matvec_fn(params, v_pytree, rng_key)
            Hv_flat, _ = ravel_pytree(Hv_pytree)
            return Hv_flat

        rng_key, lanczos_key = jax.random.split(rng_key)

        evals, evecs = lanczos(
            matvec=matvec_flat,
            dim=dim,
            num_iter=lanczos_iters,
            key=lanczos_key,
        )

        k = min(eigenvalues.shape[0], evals.shape[0])
        new_evals = jnp.zeros_like(eigenvalues)
        new_evals = new_evals.at[:k].set(evals[:k])

        new_evecs = jnp.zeros_like(eigenvectors)
        new_evecs = new_evecs.at[:k, :].set(evecs[:k, :])

        return (new_evals, new_evecs, rng_key, params)

    def dont_update(carry):
        return carry

    eigenvalues, eigenvectors, rng_key, _ = jax.lax.cond(
        should_update,
        do_update,
        dont_update,
        operand=(eigenvalues, eigenvectors, rng_key, params),
    )

    return CurvatureState(
        step=step,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        rng_key=rng_key,
    )


# ---------------------------------------------------------------------------
# Muon-style matrix optimizer (very simplified)
# ---------------------------------------------------------------------------

def _muon_matrix_update(
    grad_matrix: Array,
    lr: float,
    ns_steps: int = 2,
    eps: float = 1e-6,
) -> Array:
    """Approximate Muon-style update for a 2D weight matrix.

    This is a simplified Muon-like rule:
      - Take gradient matrix G.
      - Apply a small Newton–Schulz iteration to approximate the polar factor.
      - Use the resulting Q as update direction: ΔW ≈ -lr * Q.

    Args:
      grad_matrix: [m, n] gradient of a weight matrix.
      lr: learning rate (already includes any curvature scaling).
      ns_steps: number of Newton–Schulz iterations (1–3 is typical).
      eps: small jitter for numerical stability.

    Returns:
      ΔW: update matrix with same shape as grad_matrix.
    """
    G = grad_matrix

    # If the matrix is effectively 1D or empty, just do SGD
    if G.ndim != 2:
        return -lr * G

    m, n = G.shape
    if m == 0 or n == 0:
        return jnp.zeros_like(G)

    # Normalize by Frobenius norm to keep NS stable
    frob = jnp.linalg.norm(G)
    scale = jnp.where(frob > 0, 1.0 / (frob + eps), 1.0)
    X = G * scale

    # Approximate polar factor via Newton–Schulz iteration
    # X_{k+1} = 0.5 X_k (3I - X_k^T X_k)
    def ns_body(_, Xk):
        XtX = Xk.T @ Xk
        I = jnp.eye(XtX.shape[0], dtype=XtX.dtype)
        X_next = 0.5 * Xk @ (3.0 * I - XtX)
        return X_next

    X = jax.lax.fori_loop(0, ns_steps, ns_body, X)

    # Undo Frobenius scaling. X now ~ polar factor of G (orthogonal-ish)
    Q = X
    return -lr * Q


def _muon_tree_update(
    grads: PyTree,
    lr_tree: PyTree,
    ns_steps: int = 2,
    eps: float = 1e-6,
) -> PyTree:
    """Apply Muon-style update per leaf of gradients, given per-leaf LRs.

    For non-matrix leaves, this falls back to simple scaled SGD.
    """

    def _leaf_update(g, lr):
        if g.ndim == 2:
            return _muon_matrix_update(g, lr=lr, ns_steps=ns_steps, eps=eps)
        # fallback: simple scaled gradient for non-matrix leaves
        return -lr * g

    return jtu.tree_map(_leaf_update, grads, lr_tree)


# ---------------------------------------------------------------------------
# PNS EigenMuon: curvature-aware top subspace, Muon on complement
# ---------------------------------------------------------------------------

class PnsEigenMuonState(NamedTuple):
    """State for PN-S EigenMuon (global eigenbasis + Muon complement)."""
    curvature_state: CurvatureState
    # we keep an inner optax state only for possible extra stuff; here unused
    dummy_state: Array  # placeholder, could be jnp.array(0) to satisfy Optax


def pns_eigenmuon(
    base_learning_rate: float,
    curvature_update_every: int = 200,
    max_eigenvectors: int = 16,
    lanczos_iters: Optional[int] = None,
    ggn_matvec_fn: Optional[GGNMatvecFn] = None,
    precond_damping: float = 1e-4,
    ns_steps: int = 2,
) -> optax.GradientTransformation:
    """PN-S EigenMuon with a single global eigenbasis and Muon in the complement.

    - In the top-k eigen-subspace (spanned by leading eigenvectors of GGN/Hessian),
      we apply a Newton-like scaling: 1 / (λ_i + δ).
    - In the orthogonal complement, we apply a Muon-style matrix update.

    Args:
      base_learning_rate: global LR scale for both top-subspace and Muon part.
      curvature_update_every: refresh curvature eigenbasis every N steps.
      max_eigenvectors: maximum k to keep in the global eigenbasis.
      lanczos_iters: number of Lanczos steps; defaults to max_eigenvectors.
      ggn_matvec_fn: callable(params, direction_pytree, rng) -> H v (PyTree).
      precond_damping: δ in (Λ + δI)^{-1}.
      ns_steps: Newton–Schulz steps for Muon per matrix leaf.
    """
    if lanczos_iters is None:
        lanczos_iters = max_eigenvectors

    def init_fn(params: Params) -> PnsEigenMuonState:
        curvature_state = init_curvature_state(params, max_eigenvectors)
        dummy_state = jnp.array(0, dtype=jnp.int32)
        return PnsEigenMuonState(
            curvature_state=curvature_state,
            dummy_state=dummy_state,
        )

    def update_fn(
        grads: PyTree,
        state: PnsEigenMuonState,
        params: Optional[Params] = None,
    ):
        # 1) Refresh curvature state (step increment happens inside)
        curvature_state = maybe_update_curvature_state(
            state.curvature_state,
            params=params,
            ggn_matvec_fn=ggn_matvec_fn,
            curvature_update_every=curvature_update_every,
            lanczos_iters=lanczos_iters,
        )
        eigenvalues = curvature_state.eigenvalues
        eigenvectors = curvature_state.eigenvectors

        # 2) Flatten grads and params to work in the global eigenbasis
        flat_grads, unravel_grads = ravel_pytree(grads)
        dim = flat_grads.shape[0]

        V = eigenvectors  # (k, dim)
        lambdas = eigenvalues  # (k,)

        # If no eigenbasis yet, just fall back to Muon everywhere
        if V.shape[0] == 0 or jnp.all(lambdas == 0):
            # constant LR for all leaves
            lr_tree = jtu.tree_map(lambda _: base_learning_rate, grads)
            updates = _muon_tree_update(grads, lr_tree, ns_steps=ns_steps)
            new_state = PnsEigenMuonState(
                curvature_state=curvature_state,
                dummy_state=state.dummy_state,
            )
            return updates, new_state

        # 3) Split gradient into top-k subspace and complement
        #    g_top_coords = V g, g_parallel = V^T g_top_coords
        g_top_coords = V @ flat_grads  # (k,)
        g_parallel = V.T @ g_top_coords
        g_perp_flat = flat_grads - g_parallel

        # 4) Top-k: curvature-aware scaling (partial Newton) in eigen coordinates
        scaled_coords = g_top_coords / (lambdas + precond_damping)
        step_top_flat = -base_learning_rate * (V.T @ scaled_coords)  # Δθ∥

        # 5) Complement: reshape g_perp to PyTree shape and apply Muon per leaf
        grads_perp = unravel_grads(g_perp_flat)

        # Per-leaf LR: here we just use the same base LR; you could make this
        # blockwise curvature-aware later if you want.
        lr_tree = jtu.tree_map(lambda _: base_learning_rate, grads_perp)
        step_perp_tree = _muon_tree_update(
            grads_perp,
            lr_tree,
            ns_steps=ns_steps,
        )

        # 6) Combine: top-subspace step (flat) + complement Muon step (tree)
        step_top_tree = unravel_grads(step_top_flat)

        def _combine(top_u, perp_u):
            return top_u + perp_u

        updates = jtu.tree_map(_combine, step_top_tree, step_perp_tree)

        new_state = PnsEigenMuonState(
            curvature_state=curvature_state,
            dummy_state=state.dummy_state,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Curvature-aware Muon: use λ_max to scale the Muon learning rate
# ---------------------------------------------------------------------------

class CurvatureMuonState(NamedTuple):
    """State for curvature-aware Muon (global LR scaled by top eigenvalue)."""
    curvature_state: CurvatureState
    # here we keep a running LR for interpretability
    lr_scale: Array


def curvature_muon(
    base_learning_rate: float,
    curvature_update_every: int = 200,
    max_eigenvectors: int = 8,
    lanczos_iters: Optional[int] = None,
    ggn_matvec_fn: Optional[GGNMatvecFn] = None,
    kappa_uncertainty: float = 1.0,
    min_lr_scale: float = 1e-3,
    max_lr_scale: float = 10.0,
    ns_steps: int = 2,
) -> optax.GradientTransformation:
    """Muon optimizer whose global learning rate is scaled by curvature.

    We:
      - Track top eigenvalues of a GGN/Hessian via Lanczos.
      - Form a conservative curvature estimate λ_eff from the leading eigenvalue
        and (optionally) a rough uncertainty factor.
      - Set lr_scale ≈ 1 / (λ_eff + δ), clamped to [min_lr_scale, max_lr_scale].
      - Apply a Muon-style update per matrix parameter with LR = base_lr * lr_scale.

    Args:
      base_learning_rate: base LR that is modulated by curvature.
      curvature_update_every: refresh eigenbasis every N steps.
      max_eigenvectors: number of eigenvalues to track (we use the top one).
      lanczos_iters: Lanczos iterations used for curvature; default = max_eigenvectors.
      ggn_matvec_fn: callable(params, direction_pytree, rng) -> H v (PyTree).
      kappa_uncertainty: multiply the top eigenvalue by this factor as a safety margin.
      min_lr_scale, max_lr_scale: clamp for the curvature-based LR scale.
      ns_steps: Newton–Schulz steps for Muon per matrix leaf.
    """
    if lanczos_iters is None:
        lanczos_iters = max_eigenvectors

    def init_fn(params: Params) -> CurvatureMuonState:
        curvature_state = init_curvature_state(params, max_eigenvectors)
        lr_scale = jnp.array(1.0, dtype=jnp.float32)
        return CurvatureMuonState(
            curvature_state=curvature_state,
            lr_scale=lr_scale,
        )

    def update_fn(
        grads: PyTree,
        state: CurvatureMuonState,
        params: Optional[Params] = None,
    ):
        curvature_state = maybe_update_curvature_state(
            state.curvature_state,
            params=params,
            ggn_matvec_fn=ggn_matvec_fn,
            curvature_update_every=curvature_update_every,
            lanczos_iters=lanczos_iters,
        )

        # Leading eigenvalue as curvature scale (if available)
        eigenvalues = curvature_state.eigenvalues
        # If we have k>0, take the largest; else default to 1.0
        lam0 = jnp.where(
            eigenvalues.shape[0] > 0,
            eigenvalues[0],
            jnp.array(1.0, dtype=jnp.float32),
        )

        # Conservative effective curvature: λ_eff = kappa * max(lam0, 0)
        lam_eff = kappa_uncertainty * jnp.maximum(lam0, 0.0) + 1e-6

        # Curvature-based LR scale ≈ 1 / λ_eff (clamped)
        lr_scale = 1.0 / lam_eff
        lr_scale = jnp.clip(lr_scale, a_min=min_lr_scale, a_max=max_lr_scale)

        # Apply Muon update with LR = base_lr * lr_scale for all leaves
        effective_lr = base_learning_rate * lr_scale
        lr_tree = jtu.tree_map(lambda _: effective_lr, grads)
        updates = _muon_tree_update(
            grads,
            lr_tree,
            ns_steps=ns_steps,
        )

        new_state = CurvatureMuonState(
            curvature_state=curvature_state,
            lr_scale=lr_scale,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
