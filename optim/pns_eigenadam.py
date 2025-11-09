# optim/pns_eigenadam.py
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax


Array = jax.Array
Params = Any
PyTree = Any

# Type: given params, direction (same PyTree structure as params), rng -> matvec(direction)
GGNMatvecFn = Callable[[Params, PyTree, Array], PyTree]


class PnsEigenAdamState(NamedTuple):
    """State for PN-S EigenAdam.

    - adam_state: underlying Adam state (m, v, etc.)
    - step: global step counter
    - eigenvalues: (k,) array of top-k eigenvalues (curvature)
    - eigenvectors: (k, dim) matrix whose rows are eigenvectors in flattened space
    - rng_key: RNG key for stochastic components (e.g. Lanczos init)
    """
    adam_state: optax.OptState
    step: int
    eigenvalues: Optional[Array]
    eigenvectors: Optional[Array]
    rng_key: Array


# ---------------------------------------------------------------------------
# Lanczos iterative eigen solver on a matrix-vector product
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
      eigenvalues: (k,) approximated eigenvalues (sorted descending).
      eigenvectors: (k, dim) corresponding eigenvectors (in flattened space).
    """
    # Random normalized starting vector v0
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

        # (Optional) reorthogonalization against all previous v_j (cheap for small num_iter)
        w = jax.lax.fori_loop(0, i, lambda j, ww: ortho_against_prev(ww, j), w)

        beta = jnp.linalg.norm(w)
        beta = jnp.where(beta < eps, 0.0, beta)
        next_v = jnp.where(beta > 0, w / (beta + eps), jnp.zeros_like(w))

        V = V.at[i + 1].set(next_v)
        alphas = alphas.at[i].set(alpha)
        betas = betas.at[i].set(beta)

        return (V, alphas, betas), None

    # Allocate basis + tridiagonal coefficients
    V = jnp.zeros((num_iter + 1, dim))
    V = V.at[0].set(v0)
    alphas = jnp.zeros((num_iter,))
    betas = jnp.zeros((num_iter,))

    (V, alphas, betas), _ = jax.lax.scan(
        body_fun,
        (V, alphas, betas),
        jnp.arange(num_iter),
    )

    # Build symmetric tridiagonal matrix T
    k = num_iter
    T = jnp.diag(alphas)
    T = T.at[jnp.arange(k - 1), jnp.arange(1, k)].set(betas[: k - 1])
    T = T.at[jnp.arange(1, k), jnp.arange(k - 1)].set(betas[: k - 1])

    # Eigen-decomposition of small T (k x k)
    evals, evecs_T = jnp.linalg.eigh(T)  # ascending
    # Sort descending to get largest eigenvalues first
    idx = jnp.argsort(evals)[::-1]
    evals = evals[idx]
    evecs_T = evecs_T[:, idx]  # columns are eigenvectors

    # Map eigenvectors of T back to R^dim: V_k @ evecs_T
    # V has shape (num_iter+1, dim); we only use first k rows
    V_k = V[:-1]  # (k, dim)
    # eigenvectors_flat: (k, dim) = (k, k) @ (k, dim) via transpose trick
    # We want each eigenvector as row, so:
    eigenvectors_flat = (evecs_T.T @ V_k).reshape(k, dim)

    return evals, eigenvectors_flat


# ---------------------------------------------------------------------------
# Preconditioner in eigenbasis: M = V Λ^{-1} V^T + I - V V^T
# ---------------------------------------------------------------------------

def apply_eigen_preconditioner(
    grad_flat: Array,
    eigenvalues: Optional[Array],
    eigenvectors: Optional[Array],
    damping: float = 1e-4,
) -> Array:
    """Apply partial Newton-like preconditioner in eigenbasis.

    M = V (Λ + δI)^{-1} V^T + (I - V V^T)

    - Along each eigenvector v_i, scale gradient by 1 / (λ_i + δ).
    - In directions orthogonal to span(V), leave gradient unchanged.

    Args:
      grad_flat: (dim,) flattened gradient.
      eigenvalues: (k,) eigenvalues.
      eigenvectors: (k, dim) eigenvectors (rows).
      damping: δ, numerical damping.

    Returns:
      preconditioned_grad_flat: (dim,)
    """
    if eigenvalues is None or eigenvectors is None:
        return grad_flat

    V = eigenvectors  # (k, dim)
    lambdas = eigenvalues  # (k,)

    # Project gradient into eigenbasis: g_i = v_i^T g
    proj = V @ grad_flat  # (k,)

    # Component of g in span(V)
    proj_vec = V.T @ proj  # (dim,)

    # Scale along eigenvectors
    scaled = proj / (lambdas + damping)  # (k,)
    new_subspace = V.T @ scaled  # (dim,)

    # Orthogonal component left untouched: g_perp = g - proj_vec
    g_perp = grad_flat - proj_vec

    return new_subspace + g_perp


# ---------------------------------------------------------------------------
# PN-S EigenAdam wrapper
# ---------------------------------------------------------------------------

def pns_eigenadam(
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    curvature_update_every: int = 100,
    max_eigenvectors: int = 16,
    lanczos_iters: Optional[int] = None,
    ggn_matvec_fn: Optional[GGNMatvecFn] = None,
    precond_damping: float = 1e-4,
) -> optax.GradientTransformation:
    """PN-S EigenAdam as an Optax gradient transformation.

    Args:
      learning_rate: base LR for Adam part.
      beta1, beta2, eps: Adam hyperparameters.
      weight_decay: AdamW-style weight decay.
      curvature_update_every: recompute eigenbasis every N steps.
      max_eigenvectors: number of eigen-directions to track (k).
      lanczos_iters: number of Lanczos iterations (default = max_eigenvectors).
      ggn_matvec_fn: callable(params, vec_pytree, rng_key) -> vec_pytree
        This should implement a GGN matrix-vector product using model + loss +
        some chosen mini-batch. You will wire this in later.
      precond_damping: δ in (Λ + δI)^{-1} for numerical stability.

    Returns:
      An optax.GradientTransformation usable in Flax TrainState.
    """
    if lanczos_iters is None:
        lanczos_iters = max_eigenvectors

    base_adam = optax.adamw(
        learning_rate=learning_rate,
        b1=beta1,
        b2=beta2,
        eps=eps,
        weight_decay=weight_decay,
    )

    def init_fn(params: Params) -> PnsEigenAdamState:
        adam_state = base_adam.init(params)
        rng_key = jax.random.PRNGKey(0)
        return PnsEigenAdamState(
            adam_state=adam_state,
            step=0,
            eigenvalues=None,
            eigenvectors=None,
            rng_key=rng_key,
        )

    def update_fn(
        grads: PyTree,
        state: PnsEigenAdamState,
        params: Optional[Params] = None,
    ):
        step = state.step + 1
        rng_key = state.rng_key
        eigenvalues = state.eigenvalues
        eigenvectors = state.eigenvectors

        # -------------------------------------------------------------------
        # 1. Optionally update curvature (GGN eigenbasis) every N steps
        # -------------------------------------------------------------------
        if ggn_matvec_fn is not None and params is not None:
            # JAX boolean scalar
            should_update = (step % curvature_update_every) == 0
    
            def do_update(carry):
                eigenvalues, eigenvectors, rng_key, params = carry
    
                # Flatten params to know the dimension
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
    
                k = jnp.minimum(max_eigenvectors, evals.shape[0])
                # use jnp.minimum so shapes stay JAX-friendly
                eigenvalues = evals[:k]
                eigenvectors = evecs[:k]
    
                return (eigenvalues, eigenvectors, rng_key, params)
    
            def dont_update(carry):
                # just keep current eigenvalues/eigenvectors and rng_key
                return carry
    
            eigenvalues, eigenvectors, rng_key, _ = jax.lax.cond(
                should_update,
                do_update,
                dont_update,
                operand=(eigenvalues, eigenvectors, rng_key, params),
            )

        # -------------------------------------------------------------------
        # 2. Precondition gradients in eigenbasis
        # -------------------------------------------------------------------
        # Flatten grads to apply eigen preconditioner
        flat_grads, unravel_grads = ravel_pytree(grads)
        precond_flat_grads = apply_eigen_preconditioner(
            grad_flat=flat_grads,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            damping=precond_damping,
        )
        precond_grads = unravel_grads(precond_flat_grads)

        # -------------------------------------------------------------------
        # 3. Forward to underlying AdamW on preconditioned gradients
        # -------------------------------------------------------------------
        updates, new_adam_state = base_adam.update(
            precond_grads,
            state.adam_state,
            params=params,
        )

        new_state = PnsEigenAdamState(
            adam_state=new_adam_state,
            step=step,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rng_key=rng_key,
        )

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
