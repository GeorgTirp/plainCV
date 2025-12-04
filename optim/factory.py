# optim/factory.py

from typing import Any, Optional

import optax

from optim.kronecker import make_kronecker_ggn_matvec_fn
from .soap import soap as soap_opt
from .pns_eigenadam import pns_eigenadam
from .ggn_utils import make_ggn_matvec_fn
from .muon import build_muon_dim_numbers
from .hessian_free import hessian_free as hf_opt


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
        assert model_def is not None, "model_def required for PN-S EigenAdam"
        assert curvature_batch is not None, "curvature_batch required for PN-S EigenAdam"
    
        # Shared Adam-style stuff
        weight_decay = getattr(cfg, "weight_decay", 0.0)
        beta1 = getattr(cfg, "beta1", 0.9)
        beta2 = getattr(cfg, "beta2", 0.999)
        eps = getattr(cfg, "eps", 1e-8)
    
        # PN-S specific knobs (only read here)
        curvature_update_every = getattr(cfg, "pns_curvature_update_every", 10)
        max_eigenvectors = getattr(cfg, "pns_max_eigenvectors", 16)
        lanczos_iters = getattr(cfg, "pns_lanczos_iters", None)
        precond_damping = getattr(cfg, "pns_precond_damping", 1e-4)
        backend = getattr(cfg, "pns_curvature_backend", "ggn")
    
        # Build curvature matvec
        if backend == "ggn":
            ggn_mv = make_ggn_matvec_fn(
                model_def=model_def,
                curvature_batch=curvature_batch,
                batch_stats=batch_stats,
            )
        elif backend == "kronecker":
            from .kronecker import make_kronecker_matvec_fn
            ggn_mv = make_kronecker_ggn_matvec_fn(
                model_def=model_def,
                curvature_batch=curvature_batch,
                batch_stats=batch_stats,
            )
        else:
            raise ValueError(f"Unknown pns_curvature_backend: {backend}")
    
        tx = pns_eigenadam(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            curvature_update_every=curvature_update_every,
            max_eigenvectors=max_eigenvectors,
            lanczos_iters=lanczos_iters,
            ggn_matvec_fn=ggn_mv,
            precond_damping=precond_damping,
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
