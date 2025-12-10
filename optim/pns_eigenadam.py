# optim/pns_eigenadam.py

from readline import backend
from typing import Any, Callable, NamedTuple, Optional, Tuple
import time

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
# Block-wise SLU / SOAP-style helpers (used for sketched Lanczos as well)
# ---------------------------------------------------------------------------

class BlockSLUState(NamedTuple):
    """State for a single block in PN-S EigenAdam / SOAP-style update."""
    U: Array                # [d_b, k] current eigenbasis (columns orthonormal)
    mu_lambda: Array        # [k] running mean of eigenvalues
    sigma_lambda: Array     # [k] running std of eigenvalues
    m_top: Array            # [k] Adam first moment in top subspace
    v_top: Array            # [k] Adam second moment in top subspace
    m_perp: Array           # [d_b] Adam first moment in complement
    v_perp: Array           # [d_b] Adam second moment in complement
    last_refresh_step: Array  # scalar int32 step index


def init_block_slu_state(dim: int, k: int) -> BlockSLUState:
    """Initialize per-block SLU state with a trivial eigenbasis."""
    U0 = jnp.eye(dim, k)
    mu0 = jnp.zeros((k,))
    sigma0 = jnp.zeros((k,))
    m_top0 = jnp.zeros((k,))
    v_top0 = jnp.zeros((k,))
    m_perp0 = jnp.zeros((dim,))
    v_perp0 = jnp.zeros((dim,))
    return BlockSLUState(
        U=U0,
        mu_lambda=mu0,
        sigma_lambda=sigma0,
        m_top=m_top0,
        v_top=v_top0,
        m_perp=m_perp0,
        v_perp=v_perp0,
        last_refresh_step=jnp.array(-1, dtype=jnp.int32),
    )


def _sketched_lanczos_block(
    hvp_fn: Callable[[Array], Array],
    dim: int,
    k: int,
    s: int,
    rng: Array,
    eps: float = 1e-8,
) -> tuple[Array, Array]:
    """
    Run a small HVP-only (sketched) Lanczos on a single block or global vector.

    Args:
      hvp_fn: v -> H v (H is implicit symmetric PSD / symmetric matrix).
      dim: dimension of the vector space (flattened).
      k: number of Lanczos iterations / size of subspace (Ritz vectors).
      s: sketch dimension (rows of the random sketch S).
      rng: JAX PRNG key.
      eps: small numerical guard.

    Returns:
      evals: (k,) approximate eigenvalues (sorted descending).
      U: (dim, k) approximate eigenvectors as columns (orthonormal).
    """
    key_v, key_S = jax.random.split(rng)

    # 1) HVP-only Krylov / Lanczos basis V
    v = jax.random.normal(key_v, (dim,))
    v = v / (jnp.linalg.norm(v) + eps)

    V = jnp.zeros((dim, k))
    GV = jnp.zeros((dim, k))

    v_prev = jnp.zeros_like(v)
    beta_prev = 0.0

    for j in range(k):
        Hv = hvp_fn(v)  # H v

        # Classic Lanczos three-term recurrence
        if j > 0:
            Hv = Hv - beta_prev * v_prev

        alpha = jnp.dot(v, Hv)
        Hv = Hv - alpha * v

        beta = jnp.linalg.norm(Hv)

        # Store basis and H*basis
        V = V.at[:, j].set(v)
        GV = GV.at[:, j].set(Hv)

        # Next vector (simple breakdown guard)
        v_prev = v
        v = jnp.where(
            beta > eps,
            Hv / (beta + eps),
            v,  # if breakdown, just reuse v (QR later will clean up)
        )
        beta_prev = beta

    # 2) Sketch algebra: Y = S V, Z = S (H V)
    # S ~ Rademacher / sqrt(s) as a cheap CountSketch-ish matrix.
    S = jax.random.choice(key_S, jnp.array([-1.0, 1.0]), (s, dim))
    S = S / jnp.sqrt(s)

    Y = S @ V   # (s, k)
    Z = S @ GV  # (s, k)

    G = Y.T @ Y  # (k, k)
    C = Y.T @ Z  # (k, k)

    # 3) Tiny eigensolve: T = (Y^T Y)^† (Y^T Z)
    G_reg = G + 1e-6 * jnp.eye(k, dtype=G.dtype)
    G_inv = jnp.linalg.pinv(G_reg)
    T = G_inv @ C               # (k, k)
    T = 0.5 * (T + T.T)         # symmetrize

    evals, Z_eig = jnp.linalg.eigh(T)  # ascending
    idx = jnp.argsort(evals)[::-1]     # descending
    evals = evals[idx]
    Z_eig = Z_eig[:, idx]

    # Lift eigenvectors: U = V z_i
    U = V @ Z_eig  # (dim, k)

    # Orthonormalize columns of U
    U, _ = jnp.linalg.qr(U)  # (dim, k), (k, k)

    return evals, U


def refresh_block_slu_state(
    hvp_fn: Callable[[Array], Array],
    state: BlockSLUState,
    step: Array,
    rng: Array,
    k: int,
    s: int,
    pn_beta: float = 0.2,
) -> BlockSLUState:
    """Refresh eigenbasis & PN stats for a block using sketched Lanczos."""
    dim = state.m_perp.shape[0]

    evals, U_new = _sketched_lanczos_block(
        hvp_fn=hvp_fn,
        dim=dim,
        k=k,
        s=s,
        rng=rng,
    )

    # Moment transport: m~, v~ ← U_new^T U_old (m~, v~)
    R = U_new.T @ state.U           # (k, k)
    m_top_new = R @ state.m_top     # (k,)
    v_top_new = R @ state.v_top     # (k,)  (heuristic)

    # PN-style running mean/variance of eigenvalues
    mu_old = state.mu_lambda
    sigma_old = state.sigma_lambda
    var_old = sigma_old ** 2

    mu_new = (1.0 - pn_beta) * mu_old + pn_beta * evals
    diff = evals - mu_new
    var_new = (1.0 - pn_beta) * var_old + pn_beta * (diff ** 2)
    sigma_new = jnp.sqrt(jnp.maximum(var_new, 0.0) + 1e-8)

    return BlockSLUState(
        U=U_new,
        mu_lambda=mu_new,
        sigma_lambda=sigma_new,
        m_top=m_top_new,
        v_top=v_top_new,
        m_perp=state.m_perp,
        v_perp=state.v_perp,
        last_refresh_step=step,
    )


def pns_eigenadam_block_step(
    grad_block: Array,
    hvp_fn: Callable[[Array], Array],
    state: BlockSLUState,
    step: Array,            # scalar jnp.int32 from your global optimizer state
    rng: Array,
    *,
    k: int,
    s: int,
    refresh_every: int,
    lr_top: float,
    lr_perp: float,
    beta1: float,
    beta2: float,
    eps_adam: float = 1e-8,
    kappa_uncertainty: float = 0.0,
    eps_curv: float = 1e-4,
    pn_beta: float = 0.2,
    alpha_complement: Optional[float] = None,
) -> tuple[Array, BlockSLUState]:
    """
    One PN-S / SOAP-style update for a single parameter block.
    (Not used by the global pns_eigenadam transform, but kept for reference.)
    """
    dim = grad_block.shape[0]

    # --- JAX-safe refresh condition: use lax.cond, not Python if ---
    steps_since = step - state.last_refresh_step
    need_refresh = jnp.logical_or(
        steps_since >= refresh_every,
        state.last_refresh_step < 0,
    )

    def _do_refresh(st: BlockSLUState) -> BlockSLUState:
        return refresh_block_slu_state(
            hvp_fn=hvp_fn,
            state=st,
            step=step,
            rng=rng,
            k=k,
            s=s,
            pn_beta=pn_beta,
        )

    state = jax.lax.cond(
        need_refresh,
        _do_refresh,
        lambda st: st,
        state,
    )

    U = state.U                  # (d_b, k)
    mu = state.mu_lambda         # (k,)
    sigma = state.sigma_lambda   # (k,)

    # Split gradient into top-k subspace and complement
    g_top = U.T @ grad_block         # (k,)   g∥ in eigen coordinates
    g_parallel = U @ g_top           # (d_b,) projection into top subspace
    g_perp = grad_block - g_parallel # (d_b,) complement

    # ----------------------
    # 1) Top-k (curvature-aware Adam)
    # ----------------------
    m_top = beta1 * state.m_top + (1.0 - beta1) * g_top
    v_top = beta2 * state.v_top + (1.0 - beta2) * (g_top ** 2)

    # Bias correction (step is scalar array)
    t = step.astype(jnp.float32) + 1.0
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    m_top_hat = m_top / bc1
    v_top_hat = v_top / bc2

    # Curvature scaling: s_i = 1 / (μλ_i + κ σλ_i + ε)
    denom_curv = mu + kappa_uncertainty * sigma + eps_curv
    s_scale = 1.0 / denom_curv

    step_top_coords = -lr_top * s_scale * m_top_hat / (jnp.sqrt(v_top_hat) + eps_adam)
    step_top = U @ step_top_coords  # Δθ∥ = U_k Δθ~∥

    # ----------------------
    # 2) Complement (Adam or scaled identity)
    # ----------------------
    if alpha_complement is None:
        # Full diagonal Adam on g⊥
        m_perp = beta1 * state.m_perp + (1.0 - beta1) * g_perp
        v_perp = beta2 * state.v_perp + (1.0 - beta2) * (g_perp ** 2)

        m_perp_hat = m_perp / bc1
        v_perp_hat = v_perp / bc2

        step_perp = -lr_perp * m_perp_hat / (jnp.sqrt(v_perp_hat) + eps_adam)
    else:
        # Simple scaled identity in complement: Δθ⊥ = -η⊥ α g⊥
        m_perp = state.m_perp
        v_perp = state.v_perp
        step_perp = -lr_perp * alpha_complement * g_perp

    # ----------------------
    # 3) Combine
    # ----------------------
    delta_block = step_top + step_perp

    new_state = BlockSLUState(
        U=state.U,
        mu_lambda=mu,
        sigma_lambda=sigma,
        m_top=m_top,
        v_top=v_top,
        m_perp=m_perp,
        v_perp=v_perp,
        last_refresh_step=state.last_refresh_step,
    )

    return delta_block, new_state


# ---------------------------------------------------------------------------
# Global PN-S EigenAdam state and helpers
# ---------------------------------------------------------------------------

class PnsEigenAdamState(NamedTuple):
    """State for PN-S EigenAdam (global eigenbasis)."""
    adam_state: optax.OptState
    step: Array
    eigenvalues: Array      # (max_eigenvectors,)
    eigenvectors: Array     # (max_eigenvectors, dim)
    rng_key: Array
    rotation_diff: Array    # scalar, normalized Frobenius distance between bases


# ---------------------------------------------------------------------------
# Lanczos iterative eigen solver on a matrix-vector product (un-sketched)
# ---------------------------------------------------------------------------

def lanczos(
    matvec: Callable[[Array], Array],
    dim: int,
    num_iter: int,
    key: Array,
    eps: float = 1e-6,
    sort_by_abs: bool = False,   # <-- NEW
) -> Tuple[Array, Array]:
    """Run Lanczos to approximate top eigenvalues/eigenvectors.

    Args:
      matvec: function v -> A @ v (A is implicit symmetric matrix, e.g. GGN or Hessian).
      dim: dimension of the flattened parameter vector.
      num_iter: number of Lanczos iterations (also size of Krylov subspace).
      key: RNG key for initial vector.
      eps: small value to guard against breakdown.
      sort_by_abs: if True, sort eigenvalues by |λ| descending instead of λ descending.

    Returns:
      eigenvalues: (num_iter,) approximated eigenvalues (sorted).
      eigenvectors: (num_iter, dim) corresponding eigenvectors (in flattened space).
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

        # reorthogonalize against previous basis vectors
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

    if sort_by_abs:
        # Largest |λ| first (so we also capture big negatives)
        idx = jnp.argsort(jnp.abs(evals))[::-1]
    else:
        # Largest λ first (standard GGN case)
        idx = jnp.argsort(evals)[::-1]

    evals = evals[idx]
    evecs_T = evecs_T[:, idx]

    V_k = V[:-1]  # (k, dim)
    eigenvectors_flat = (evecs_T.T @ V_k).reshape(k, dim)

    return evals, eigenvectors_flat



# ---------------------------------------------------------------------------
# Preconditioners in eigenbasis
# ---------------------------------------------------------------------------

def apply_eigen_preconditioner(
    grad_flat: Array,
    eigenvalues: Array,
    eigenvectors: Array,
    damping: float = 1e-4,
    saddle_free_neg: bool = False,  # <-- NEW FLAG
) -> Array:
    """Apply partial Newton-like preconditioner in eigenbasis.

    M ≈ V diag(m_i) V^T + (I - V V^T), where:

      - If saddle_free_neg == False:
           m_i = 1 / (λ_i + δ)
      - If saddle_free_neg == True (Hessian backend):
           m_i = 1 / (|λ_i| + δ)  for all i
           (this automatically becomes "saddle-free" on negative λ_i).

    Args:
      grad_flat: (dim,) flattened gradient.
      eigenvalues: (k,) eigenvalues (can be positive or negative).
      eigenvectors: (k, dim) eigenvectors (rows).
      damping: δ, numerical damping.
      saddle_free_neg: if True, use |λ| in the denominator (saddle-free Newton).

    Returns:
      preconditioned_grad_flat: (dim,)
    """
    if eigenvalues.size == 0:
        return grad_flat

    V = eigenvectors  # (k, dim)
    lambdas = eigenvalues  # (k,)

    # Project gradient into eigenbasis: g_i = v_i^T g
    proj = V @ grad_flat  # (k,)

    # Component of g in span(V)
    proj_vec = V.T @ proj  # (dim,)

    if saddle_free_neg:
        # Saddle-free: use |λ| + δ (this equals standard scaling on λ>=0)
        lam_abs = jnp.abs(lambdas)
        scale = 1.0 / (lam_abs + damping)
    else:
        # Standard PN-S / Newton-like scaling
        scale = 1.0 / (lambdas + damping)

    scaled = proj * scale           # (k,)
    new_subspace = V.T @ scaled     # (dim,)

    # Orthogonal component left untouched
    g_perp = grad_flat - proj_vec

    return new_subspace + g_perp



# ---------------------------------------------------------------------------
# Global PN-S EigenAdam wrapper (single global eigenbasis, optional sketching,
# and optional saddle-free treatment when curvature_is_hessian=True)
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
    sketch_dim: Optional[int] = None,
    *,
    curvature_is_hessian: bool = False,
) -> optax.GradientTransformation:
    """
    PN-S EigenAdam as an Optax gradient transformation (global eigenbasis).

    - If `sketch_dim` is None (default), curvature is estimated via standard
      Lanczos in the full parameter space.

    - If `sketch_dim` is not None, curvature is estimated via a sketched Lanczos
      procedure using a random projection S ∈ R^{sketch_dim × dim}.

    - If `curvature_is_hessian=True`, we interpret the curvature operator as the
      (possibly indefinite) Hessian. We then:
        * sort the Ritz pairs by |λ| descending (largest magnitude first),
        * apply a saddle-free treatment to negative eigenvalues:
              λ_eff = λ          if λ >= 0
              λ_eff = |λ|        if λ < 0
          in the preconditioner.
      This matches the "saddle-free Newton" idea along the directions we
      explicitly track, while keeping the rest of the space as in standard Adam.
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
        flat_params, _ = ravel_pytree(params)
        dim = flat_params.shape[0]

        eigenvalues = jnp.zeros((max_eigenvectors,), dtype=jnp.float32)
        eigenvectors = jnp.zeros((max_eigenvectors, dim), dtype=jnp.float32)

        adam_state = base_adam.init(params)
        rng_key = jax.random.PRNGKey(0)
        step = jnp.array(0, dtype=jnp.int32)
        rotation_diff = jnp.array(0.0, dtype=jnp.float32)

        return PnsEigenAdamState(
            adam_state=adam_state,
            step=step,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rng_key=rng_key,
            rotation_diff=rotation_diff,
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
        rotation_diff = state.rotation_diff

        # -------------------------------------------------------------------
        # 1. Optionally update curvature (GGN or Hessian eigenbasis) every N steps
        # -------------------------------------------------------------------
        use_saddle_free = (backend == "hessian")
        sort_by_abs = (backend == "hessian")

        if ggn_matvec_fn is not None and params is not None:
            should_update = (step % curvature_update_every) == 0

            def do_update(carry):
                eigenvalues, eigenvectors, rng_key, params, rotation_diff = carry

                flat_params, unravel_params = ravel_pytree(params)
                dim = flat_params.shape[0]

                def matvec_flat(v_flat: Array) -> Array:
                    v_pytree = unravel_params(v_flat)
                    Hv_pytree = ggn_matvec_fn(params, v_pytree, rng_key)
                    Hv_flat, _ = ravel_pytree(Hv_pytree)
                    return Hv_flat

                rng_key, lanczos_key = jax.random.split(rng_key)

                # Either full Lanczos, or sketched Lanczos if sketch_dim is set.
                if sketch_dim is None:
                    evals, evecs = lanczos(
                        matvec=matvec_flat,
                        dim=dim,
                        num_iter=lanczos_iters,
                        key=lanczos_key,
                        sort_by_abs=sort_by_abs,
                    )  # evecs: (k, dim) rows are eigenvectors
                else:
                    evals, U = _sketched_lanczos_block(
                        hvp_fn=matvec_flat,  # v_flat -> H v_flat
                        dim=dim,
                        k=lanczos_iters,
                        s=sketch_dim,
                        rng=lanczos_key,
                    )
                    # _sketched_lanczos_block returns eigenvectors as columns (dim, k)
                    evecs = U.T  # shape (k, dim) to match apply_eigen_preconditioner

                # For a Hessian backend, reorder by |λ| to get largest magnitude modes
                if curvature_is_hessian:
                    order = jnp.argsort(jnp.abs(evals))[::-1]
                    evals = evals[order]
                    evecs = evecs[order, :]

                # Use a static number of eigenvectors (<= max_eigenvectors)
                k = min(eigenvalues.shape[0], evals.shape[0])

                # ---------- rotation matrix change (normalized Frobenius) ----------
                prev_vecs_k = eigenvectors[:k, :]  # (k, dim)
                new_vecs_k = evecs[:k, :]          # (k, dim)

                diff = new_vecs_k - prev_vecs_k
                frob_num = jnp.linalg.norm(diff)  # ||ΔV||_F
                frob_den = jnp.linalg.norm(prev_vecs_k)
                frob_den_safe = jnp.where(frob_den > 1e-8, frob_den, 1.0)
                rotation_diff_new = jnp.where(
                    frob_den > 1e-8,
                    frob_num / frob_den_safe,
                    jnp.array(0.0, dtype=jnp.float32),
                )

                # Store eigenvalues/eigenvectors in fixed-size arrays
                eigenvalues_new = jnp.zeros_like(eigenvalues)
                eigenvalues_new = eigenvalues_new.at[:k].set(evals[:k])

                eigenvectors_new = jnp.zeros_like(eigenvectors)
                eigenvectors_new = eigenvectors_new.at[:k, :].set(evecs[:k, :])

                return (
                    eigenvalues_new,
                    eigenvectors_new,
                    rng_key,
                    params,
                    rotation_diff_new,
                )

            def dont_update(carry):
                # No change to eigenvalues/eigenvectors/rotation_diff/rng_key
                return carry

            eigenvalues, eigenvectors, rng_key, _, rotation_diff = jax.lax.cond(
                should_update,
                do_update,
                dont_update,
                operand=(eigenvalues, eigenvectors, rng_key, params, rotation_diff),
            )

        # -------------------------------------------------------------------
        # 2. Precondition gradients in eigenbasis
        #    - PSD / GGN: standard PN-S EigenAdam (λ in denominator)
        #    - Hessian: saddle-free treatment on negative eigenvalues
        # -------------------------------------------------------------------
        flat_grads, unravel_grads = ravel_pytree(grads)
        precond_flat_grads = apply_eigen_preconditioner(
            grad_flat=flat_grads,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            damping=precond_damping,
            saddle_free_neg=use_saddle_free,
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
            rotation_diff=rotation_diff,
        )

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Profiling helpers (manual, not used inside the JITted training loop)
# ---------------------------------------------------------------------------

def profile_pns_eigenadam_curvature(
    params: Params,
    ggn_matvec_fn: GGNMatvecFn,
    max_eigenvectors: int = 16,
    lanczos_iters: Optional[int] = None,
    rng: Optional[Array] = None,
    warmup: bool = True,
) -> None:
    """
    Run a single curvature update with line_profiler + wall-clock timing.

    This is meant for manual debugging outside the JITted training loop.
    Usage example:
        from optim.pns_eigenadam import profile_pns_eigenadam_curvature
        profile_pns_eigenadam_curvature(params, ggn_matvec_fn, max_eigenvectors=8)
    """
    try:
        from line_profiler import LineProfiler
    except ImportError as exc:
        raise ImportError(
            "line_profiler is required for profiling. Install via `pip install line_profiler`."
        ) from exc

    if lanczos_iters is None:
        lanczos_iters = max_eigenvectors
    if rng is None:
        rng = jax.random.PRNGKey(0)

    flat_params, unravel_params = ravel_pytree(params)
    dim = flat_params.shape[0]
    rng, lanczos_key = jax.random.split(rng)

    def matvec_flat(v_flat: Array) -> Array:
        v_pytree = unravel_params(v_flat)
        Hv_pytree = ggn_matvec_fn(params, v_pytree, rng)
        Hv_flat, _ = ravel_pytree(Hv_pytree)
        return Hv_flat

    if warmup:
        # Trigger compilation so the profiled run focuses on execution time.
        _ = lanczos(matvec_flat, dim=dim, num_iter=lanczos_iters, key=lanczos_key)

    lp = LineProfiler()
    profiled_lanczos = lp(lanczos)
    profiled_matvec = lp(matvec_flat)

    start = time.perf_counter()
    evals, evecs = profiled_lanczos(
        profiled_matvec,
        dim=dim,
        num_iter=lanczos_iters,
        key=lanczos_key,
    )
    # Wait for device execution so timing reflects the full runtime.
    jax.block_until_ready((evals, evecs))
    elapsed = time.perf_counter() - start

    print(f"[PN-S EigenAdam] Curvature step wall-clock: {elapsed:.3f} s")
    print(f"Top eigenvalues (first 5): {jnp.asarray(evals)[:5]}")
    lp.print_stats()
