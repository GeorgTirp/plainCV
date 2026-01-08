# optim/factory.py

from typing import Any, Optional

import optax


from .pns_eigenmuon import pns_eigenmuon
from .soap import soap as soap_opt
from .pns_eigenadam import pns_eigenadam
from .ggn_utils import make_ggn_matvec_fn, make_hessian_matvec_fn, make_fisher_matvec_fn
from .muon import build_muon_dim_numbers
from .hessian_free import hessian_free as hf_opt
from .shampoo import shampoo as shampoo_opt
from .sophia import sophia as sophia_opt
from .sophia import sophia_shampoo as sophia_shampoo_opt
from .lanzos_hybrid import pns_eigen_hybrid


def maybe_wrap_schedule_free(base_tx, cfg):
    """Wrap base_tx with schedule-free, if requested in config."""
    if not getattr(cfg, "schedule_free", False):
        return base_tx

    # If you want separate hyperparams, read them from config,
    # otherwise just reuse lr from the optimizer.
    sf_lr = getattr(cfg, "schedule_free_lr", cfg.lr)
    sf_b1 = getattr(cfg, "schedule_free_b1", 0.9)
    sf_wlp = getattr(cfg, "schedule_free_weight_lr_power", 2.0)

    sf_tx = optax.contrib.schedule_free(
        base_optimizer=base_tx,
        learning_rate=sf_lr,
        b1=sf_b1,
        weight_lr_power=sf_wlp,
    )
    return sf_tx


def get_optimizer(
    cfg,
    model_def: Optional[Any] = None,
    curvature_batch: Optional[Any] = None,
    batch_stats: Optional[Any] = None,
) -> optax.GradientTransformation:
    # You can leave default "adamw" if you like; config will override anyway.
    name = getattr(cfg, "optim", "adamw").lower()
    lr = float(cfg.lr)

    # ------------------------
    # Standard Adam (NEW)
    # ------------------------
    if name == "adam":
        beta1 = getattr(cfg, "beta1", 0.9)
        beta2 = getattr(cfg, "beta2", 0.999)
        eps = getattr(cfg, "eps", 1e-8)
        weight_decay = getattr(cfg, "weight_decay", 0.0)

        adam_tx = optax.adam(
            learning_rate=lr,
            b1=beta1,
            b2=beta2,
            eps=eps,
        )

        # "Standard" Adam + optional L2-style weight decay
        if weight_decay != 0.0:
            tx = optax.chain(
                optax.add_decayed_weights(weight_decay),
                adam_tx,
            )
        else:
            tx = adam_tx

    # ------------------------
    # AdamW
    # ------------------------
    elif name == "adamw":
        weight_decay = getattr(cfg, "weight_decay", 0.0)
        beta1 = getattr(cfg, "beta1", 0.9)
        beta2 = getattr(cfg, "beta2", 0.999)
        eps = getattr(cfg, "eps", 1e-8)
        tx = optax.adamw(
            learning_rate=lr,
            b1=beta1,
            b2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )

    # ------------------------
    # PN-S EigenAdam
    # ------------------------
    elif name in {"pns_eigenadam", "pns-eigenadam"}:
        assert model_def is not None
        assert curvature_batch is not None
    
        weight_decay = getattr(cfg, "weight_decay", 0.0)
        beta1 = getattr(cfg, "beta1", 0.9)
        beta2 = getattr(cfg, "beta2", 0.999)
        eps = getattr(cfg, "eps", 1e-8)
    
        curvature_update_every = getattr(cfg, "pns_curvature_update_every", 10)
        max_eigenvectors = getattr(cfg, "curvature_eigenvectors", 16)
        lanczos_iters = getattr(cfg, "curvature_iters", None)
        precond_damping = getattr(cfg, "pns_precond_damping", 1e-4)
        backend = getattr(cfg, "pns_curvature_backend", "ggn")
        split_spaces = getattr(cfg, "pns_split_spaces", False)
    
        if backend == "ggn":
            curv_mv = make_ggn_matvec_fn(
                model_def=model_def,
                curvature_batch=curvature_batch,
                batch_stats=batch_stats,
            )

        elif backend == "hessian":
            curv_mv = make_hessian_matvec_fn(
                model_def=model_def,
                curvature_batch=curvature_batch,
                batch_stats=batch_stats,
            )
        elif backend == "fisher":
            curv_mv = make_fisher_matvec_fn(
                model_def=model_def,
                curvature_batch=curvature_batch,
                batch_stats=batch_stats,
            )

        else:
            raise ValueError(f"Unknown pns_curvature_backend: {backend}")
    
        from optim.pns_eigenadam import pns_eigenadam
    
        tx = pns_eigenadam(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            curvature_update_every=curvature_update_every,
            max_eigenvectors=max_eigenvectors,
            lanczos_iters=lanczos_iters,
            ggn_matvec_fn=curv_mv,
            precond_damping=precond_damping,
            backend=backend,
            split_spaces=split_spaces,
            lr_top = 0.1,
            lr_perp= 0.0001,
        )



    # ------------------------
    # Muon
    # ------------------------
    elif name == "muon":
        # Shared / legacy config fields
        weight_decay = getattr(cfg, "weight_decay", 0.0)
        beta1 = getattr(cfg, "beta1", 0.9)   # Adam part (non-2D params)
        beta2 = getattr(cfg, "beta2", 0.999)
        eps = getattr(cfg, "eps", 1e-8)

        # Muon-specific knobs (all optional)
        muon_beta = getattr(cfg, "muon_beta", 0.95)  # momentum for Muon
        ns_steps = getattr(cfg, "muon_ns_steps", 5)
        ns_coeffs = getattr(
            cfg,
            "muon_ns_coeffs",
            (3.4445, -4.7750, 2.0315),
        )
        nesterov = getattr(cfg, "muon_nesterov", True)
        adaptive = getattr(cfg, "muon_adaptive", False)
        adam_eps_root = getattr(cfg, "adam_eps_root", 0.0)
        consistent_rms = getattr(cfg, "muon_consistent_rms", None)

        def dim_fn(p):
            return build_muon_dim_numbers(p)

        tx = optax.contrib.muon(
            learning_rate=lr,
            ns_coeffs=ns_coeffs,
            ns_steps=ns_steps,
            beta=muon_beta,
            eps=eps,
            # L2-style weight decay on Muon (matrix) params:
            weight_decay=weight_decay,
            weight_decay_mask=None,
            mu_dtype=None,
            nesterov=nesterov,
            adaptive=adaptive,
            # Adam part for non-2D params:
            adam_b1=beta1,
            adam_b2=beta2,
            adam_eps_root=adam_eps_root,
            adam_weight_decay=weight_decay,
            # Default: Muon only on 2D params
            muon_weight_dimension_numbers=dim_fn,
            # consistent_rms=consistent_rms,
        )

        # ------------------------
    # PN-S EigenMuon
    # ------------------------
    elif name in {"pns_eigenmuon", "pns-eigenmuon"}:
        # Hyperparams (with sensible defaults)
        weight_decay = getattr(cfg, "weight_decay", 0.0)
        beta1 = getattr(cfg, "beta1", 0.9)
        beta2 = getattr(cfg, "beta2", 0.999)
        eps = getattr(cfg, "eps", 1e-8)

        max_eigenvectors = getattr(cfg, "gradient_eigenvectors", 8)
        lanczos_iters = getattr(cfg, "gradient_iters", None)
        precond_damping = getattr(cfg, "pns_precond_damping", 1e-4)
        sqrt_scaling = getattr(cfg, "pns_sqrt_scaling", False)

        if lanczos_iters is None:
            lanczos_iters = max_eigenvectors

        tx = pns_eigenmuon(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            max_eigenvectors=max_eigenvectors,
            lanczos_iters=lanczos_iters,
            precond_damping=precond_damping,
            sqrt_scaling=sqrt_scaling,
        )


        # ------------------------
    # Hybrid PN-S EigenAdam + EigenMuon
    #   - HVP-based global curvature (GGN / Hessian / Fisher)
    #   - per-layer Gram-based matrix preconditioner on 2D grads
    elif name in {"pns_eigen_hybrid", "pns-eigen-hybrid"}:
        assert model_def is not None
        assert curvature_batch is not None

        # Shared Adam-style hyperparams
        weight_decay = getattr(cfg, "weight_decay", 0.0)
        beta1 = getattr(cfg, "beta1", 0.9)
        beta2 = getattr(cfg, "beta2", 0.999)
        eps = getattr(cfg, "eps", 1e-8)

        # -------- global PN-S (Hessian/GGN/Fisher) part --------
        curvature_update_every = getattr(cfg, "pns_curvature_update_every", 10)
        global_max_eigenvectors = getattr(cfg, "curvature_eigenvectors", 16)
        global_lanczos_iters = getattr(cfg, "curvature_iters", None)
        global_precond_damping = getattr(cfg, "pns_precond_damping", 1e-4)
        backend = getattr(cfg, "pns_curvature_backend", "ggn")

        # If user didn't set curvature_iters explicitly, default to k.
        if global_lanczos_iters is None:
            global_lanczos_iters = global_max_eigenvectors

        # Decide whether to actually build a curvature matvec
        use_global_curv = (
            (global_max_eigenvectors > 0)
            and (global_lanczos_iters != 0)
            and (curvature_update_every > 0)
        )

        if use_global_curv:
            if backend == "ggn":
                curv_mv = make_ggn_matvec_fn(
                    model_def=model_def,
                    curvature_batch=curvature_batch,
                    batch_stats=batch_stats,
                )
            elif backend == "hessian":
                curv_mv = make_hessian_matvec_fn(
                    model_def=model_def,
                    curvature_batch=curvature_batch,
                    batch_stats=batch_stats,
                )
            elif backend == "fisher":
                curv_mv = make_fisher_matvec_fn(
                    model_def=model_def,
                    curvature_batch=curvature_batch,
                    batch_stats=batch_stats,
                )
            else:
                raise ValueError(f"Unknown pns_curvature_backend: {backend}")
        else:
            # Let the hybrid optimizer detect ggn_matvec_fn=None
            # and effectively disable the global curvature part.
            curv_mv = None

        # -------- per-matrix (muon-style) Gram part --------
        muon_max_eigenvectors = getattr(cfg, "gradient_eigenvectors", 8)
        muon_lanczos_iters = getattr(cfg, "gradient_iters", None)
        muon_precond_damping = getattr(
            cfg, "pns_grad_precond_damping", global_precond_damping
        )
        muon_sqrt_scaling = getattr(cfg, "pns_grad_sqrt_scaling", False)

        if muon_lanczos_iters is None:
            muon_lanczos_iters = muon_max_eigenvectors

        # NOTE: if user sets gradient_eigenvectors=0 or gradient_iters=0,
        # the hybrid optimizer should see k<=0 and effectively skip the
        # muon part on each matrix.

        tx = pns_eigen_hybrid(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            # global / Hessian part
            ggn_matvec_fn=curv_mv,
            global_max_eigenvectors=global_max_eigenvectors,
            global_lanczos_iters=global_lanczos_iters,
            global_precond_damping=global_precond_damping,
            curvature_update_every=curvature_update_every,
            backend=backend,
            # per-matrix muon part
            muon_max_eigenvectors=muon_max_eigenvectors,
            muon_lanczos_iters=muon_lanczos_iters,
            muon_precond_damping=muon_precond_damping,
            muon_sqrt_scaling=muon_sqrt_scaling,
        )


    # ------------------------
    # SOAP
    # ------------------------
    elif name == "soap":
        weight_decay = getattr(cfg, "weight_decay", 0.0)
        beta1 = getattr(cfg, "beta1", 0.9)
        beta2 = getattr(cfg, "beta2", 0.95)
        eps = getattr(cfg, "eps", 1e-8)
        precond_freq = getattr(cfg, "precondition_frequency", 10)
        shampoo_beta2 = getattr(cfg, "shampoo_beta2", None)
        log_skipped = getattr(cfg, "soap_log_skipped", False)

        tx = soap_opt(
            learning_rate=lr,
            b1=beta1,
            b2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precond_freq,
            shampoo_beta2=shampoo_beta2,
            log_skipped=log_skipped,
        )

     # ------------------------
    # Shampoo
    # ------------------------
    elif name == "shampoo":
        weight_decay = getattr(cfg, "weight_decay", 0.0)
        eps = getattr(cfg, "eps", 1e-4)
        max_dim = getattr(cfg, "shampoo_max_dim", 2048)
        exponent = getattr(cfg, "shampoo_exponent", 0.25)

        tx = shampoo_opt(
            learning_rate=lr,
            eps=eps,
            max_dim=max_dim,
            exponent=exponent,
            weight_decay=weight_decay,
        )
    
        # ------------------------
    # Sophia (diagonal Hessian, clipped) 
    # ------------------------
    elif name == "sophia":
        assert model_def is not None, "model_def required for Sophia"
        assert curvature_batch is not None, "curvature_batch required for Sophia"

        weight_decay = getattr(cfg, "weight_decay", 0.0)
        beta1 = getattr(cfg, "beta1", 0.9)
        beta2 = getattr(cfg, "beta2", 0.99)
        rho = getattr(cfg, "sophia_rho", 0.01)
        clip_threshold = getattr(cfg, "sophia_clip_threshold", 1.0)
        hessian_update_every = getattr(cfg, "sophia_hessian_update_every", 10)
        hutchinson_samples = getattr(cfg, "sophia_hutchinson_samples", 1)
        backend = getattr(cfg, "sophia_curvature_backend", "ggn")

        # Build curvature matvec (same style as PN-S / HF)
        if backend == "ggn":
            ggn_mv = make_ggn_matvec_fn(
                model_def=model_def,
                curvature_batch=curvature_batch,
                batch_stats=batch_stats,
            )
        elif backend == "kronecker":
            ggn_mv = make_hessian_matvec_fn(
                model_def=model_def,
                curvature_batch=curvature_batch,
                batch_stats=batch_stats,
            )
        else:
            raise ValueError(f"Unknown sophia_curvature_backend: {backend}")

        tx = sophia_opt(
            learning_rate=lr,
            ggn_matvec_fn=ggn_mv,
            beta1=beta1,
            beta2=beta2,
            rho=rho,
            clip_threshold=clip_threshold,
            hessian_update_every=hessian_update_every,
            hutchinson_samples=hutchinson_samples,
            weight_decay=weight_decay,
        )

        # ------------------------
    # Sophia & Sophia+Shampoo
    # ------------------------
    elif name in {"sophia", "sophia_shampoo"}:
        assert model_def is not None, "model_def required for Sophia"
        assert curvature_batch is not None, "curvature_batch required for Sophia"

        # Build Hessian-vector-product for the *training loss* on curvature_batch
        hvp_fn = make_hessian_matvec_fn(
            model_def=model_def,
            curvature_batch=curvature_batch,
            batch_stats=batch_stats,
        )

        # Shared Sophia hyperparams
        beta1 = getattr(cfg, "beta1", 0.9)
        beta2 = getattr(cfg, "beta2", 0.99)
        rho = getattr(cfg, "sophia_rho", 0.01)
        h_max = getattr(cfg, "sophia_h_max", 1e6)
        eps = getattr(cfg, "eps", 1e-8)
        hessian_update_every = getattr(cfg, "sophia_hessian_update_every", 10)

        if name == "sophia":
            tx = sophia_opt(
                learning_rate=lr,
                hessian_matvec_fn=hvp_fn,
                beta1=beta1,
                beta2=beta2,
                rho=rho,
                h_max=h_max,
                eps=eps,
                hessian_update_every=hessian_update_every,
            )
        else:  # "sophia_shampoo"
            shampoo_eps = getattr(cfg, "shampoo_eps", 1e-4)
            shampoo_max_dim = getattr(cfg, "shampoo_max_dim", 2048)
            shampoo_exponent = getattr(cfg, "shampoo_exponent", 0.25)

            tx = sophia_shampoo_opt(
                learning_rate=lr,
                hessian_matvec_fn=hvp_fn,
                beta1=beta1,
                beta2=beta2,
                rho=rho,
                h_max=h_max,
                eps=eps,
                hessian_update_every=hessian_update_every,
                shampoo_eps=shampoo_eps,
                shampoo_max_dim=shampoo_max_dim,
                shampoo_exponent=shampoo_exponent,
            )


    # ------------------------
    # Hessian-free
    # ------------------------
    elif name in {"hf", "hessian_free"}:
        assert model_def is not None, "model_def required for Hessian-free"
        assert curvature_batch is not None, "curvature_batch required for Hessian-free"

        weight_decay = getattr(cfg, "weight_decay", 0.0)
        damping = getattr(cfg, "hf_damping", 1e-3)
        cg_max_iters = getattr(cfg, "hf_cg_max_iters", 50)
        cg_tol = getattr(cfg, "hf_cg_tol", 1e-4)

        ggn_mv = make_ggn_matvec_fn(
            model_def=model_def,
            curvature_batch=curvature_batch,
            batch_stats=batch_stats,
        )

        tx = hf_opt(
            ggn_matvec_fn=ggn_mv,
            learning_rate=lr,
            weight_decay=weight_decay,
            damping=damping,
            cg_max_iters=cg_max_iters,
            cg_tol=cg_tol,
        )

    else:
        raise ValueError(f"Unknown optimizer name: {cfg.optim}")

    # Optional schedule-free wrapper
    tx = maybe_wrap_schedule_free(tx, cfg)
    return tx
