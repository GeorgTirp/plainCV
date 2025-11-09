# optim/factory.py
from typing import Any

import optax

from .pns_eigenadam import pns_eigenadam


def get_optimizer(cfg) -> optax.GradientTransformation:
    """Return an Optax gradient transformation based on config.

    Expected fields in cfg:
      - optim: string, e.g. "adamw" or "pns_eigenadam"
      - lr: float
      - (optional) beta1, beta2, eps, weight_decay, etc.
    """
    name = getattr(cfg, "optim", "adamw").lower()

    lr = float(cfg.lr)

    if name == "adamw":
        weight_decay = getattr(cfg, "weight_decay", 0.0)
        beta1 = getattr(cfg, "beta1", 0.9)
        beta2 = getattr(cfg, "beta2", 0.999)
        eps = getattr(cfg, "eps", 1e-8)
        return optax.adamw(
            learning_rate=lr,
            b1=beta1,
            b2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )

    elif name in {"pns_eigenadam", "pns-eigenadam"}:
        weight_decay = getattr(cfg, "weight_decay", 0.0)
        beta1 = getattr(cfg, "beta1", 0.9)
        beta2 = getattr(cfg, "beta2", 0.999)
        eps = getattr(cfg, "eps", 1e-8)
        curvature_update_every = getattr(cfg, "curvature_update_every", 100)
        max_eigenvectors = getattr(cfg, "max_eigenvectors", 16)

        return pns_eigenadam(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            curvature_update_every=curvature_update_every,
            max_eigenvectors=max_eigenvectors,
        )

    else:
        raise ValueError(f"Unknown optimizer name: {cfg.optim}")
