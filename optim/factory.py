# optim/factory.py
from typing import Any, Optional

import optax

from .pns_eigenadam import pns_eigenadam
from .ggn_utils import make_ggn_matvec_fn
from .muon import build_muon_dim_numbers


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
    name = getattr(cfg, "optim", "adamw").lower()
    lr = float(cfg.lr)

    if name == "adamw":
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

    elif name in {"pns_eigenadam", "pns-eigenadam"}:
        assert model_def is not None, "model_def required for PN-S EigenAdam"
        assert curvature_batch is not None, "curvature_batch required for PN-S EigenAdam"

        weight_decay = getattr(cfg, "weight_decay", 0.0)
        beta1 = getattr(cfg, "beta1", 0.9)
        beta2 = getattr(cfg, "beta2", 0.999)
        eps = getattr(cfg, "eps", 1e-8)
        curvature_update_every = getattr(cfg, "curvature_update_every", 1)
        max_eigenvectors = getattr(cfg, "max_eigenvectors", 16)

        ggn_mv = make_ggn_matvec_fn(
            model_def=model_def,
            curvature_batch=curvature_batch,
            batch_stats=batch_stats,
        )

        tx = pns_eigenadam(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            curvature_update_every=curvature_update_every,
            max_eigenvectors=max_eigenvectors,
            ggn_matvec_fn=ggn_mv,
            params=None
        )

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
        # optax.contrib.muon:
        # - applies Muon to all 2D parameters (by default),
        # - uses AdamW-style updates for everything else. :contentReference[oaicite:0]{index=0}
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
            #consistent_rms=consistent_rms,
        )

    else:
        raise ValueError(f"Unknown optimizer name: {cfg.optim}")
    
    tx = maybe_wrap_schedule_free(tx, cfg)
    return tx