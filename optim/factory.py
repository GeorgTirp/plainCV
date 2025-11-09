# optim/factory.py
from typing import Any, Optional

import optax

from .pns_eigenadam import pns_eigenadam
from .ggn_utils import make_ggn_matvec_fn


def get_optimizer(
    cfg,
    model_def: Optional[Any] = None,
    curvature_batch: Optional[Any] = None,
) -> optax.GradientTransformation:
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
        assert model_def is not None, "model_def required for PN-S EigenAdam"
        assert curvature_batch is not None, "curvature_batch required for PN-S EigenAdam"

        weight_decay = getattr(cfg, "weight_decay", 0.0)
        beta1 = getattr(cfg, "beta1", 0.9)
        beta2 = getattr(cfg, "beta2", 0.999)
        eps = getattr(cfg, "eps", 1e-8)
        curvature_update_every = getattr(cfg, "curvature_update_every", 100)
        max_eigenvectors = getattr(cfg, "max_eigenvectors", 16)

        ggn_mv = make_ggn_matvec_fn(
            model_def=model_def,
            curvature_batch=curvature_batch,
        )

        return pns_eigenadam(
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            curvature_update_every=curvature_update_every,
            max_eigenvectors=max_eigenvectors,
            ggn_matvec_fn=ggn_mv,
        )

    else:
        raise ValueError(f"Unknown optimizer name: {cfg.optim}")
