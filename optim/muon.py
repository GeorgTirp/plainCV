# optim/muon.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import optax
import jax.tree_util as jtu

@dataclass
class MuonConfig:
    """Hyperparameters for the Muon optimizer.

    Fields are chosen to be compatible with typical plainLM-style configs, i.e.
    you can construct a MuonConfig from a config node that has at least:
      - lr
      - wd
      - beta1
      - beta2

    You can also override the Muon-specific ones if you like.
    """

    # main knobs
    learning_rate: float
    weight_decay: float = 0.0  # passed as adam_weight_decay to optax.contrib.muon

    # Muon / momentum settings
    beta: float = 0.95              # momentum for Muon itself
    ns_steps: int = 5               # Newton–Schulz iterations
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
    eps: float = 1e-8
    nesterov: bool = True
    adaptive: bool = False          # see Optax docs (dual-norm scaling)

    # AdamW part (used for non-2D params)
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps_root: float = 0.0


def _muon_config_from_generic_config(cfg: Any) -> MuonConfig:
    """Build a MuonConfig from your existing config object.

    This assumes a 'plainLM-style' config with at least:
        cfg.lr
        cfg.wd
        cfg.beta1
        cfg.beta2

    Optional Muon-specific overrides (if present):
        cfg.muon_beta
        cfg.muon_ns_steps
        cfg.muon_eps
        cfg.muon_nesterov
        cfg.muon_adaptive
    """
    return MuonConfig(
        learning_rate=float(getattr(cfg, "lr")),
        weight_decay=float(getattr(cfg, "wd", 0.0)),
        beta=float(getattr(cfg, "muon_beta", 0.95)),
        ns_steps=int(getattr(cfg, "muon_ns_steps", 5)),
        ns_coeffs=tuple(
            getattr(cfg, "muon_ns_coeffs", (3.4445, -4.7750, 2.0315))
        ),
        eps=float(getattr(cfg, "muon_eps", 1e-8)),
        nesterov=bool(getattr(cfg, "muon_nesterov", True)),
        adaptive=bool(getattr(cfg, "muon_adaptive", False)),
        adam_b1=float(getattr(cfg, "beta1", 0.9)),
        adam_b2=float(getattr(cfg, "beta2", 0.999)),
        adam_eps_root=float(getattr(cfg, "adam_eps_root", 0.0)),
    )


def build_muon_tx_from_cfg(cfg: Any) -> optax.GradientTransformation:
    """Factory used from the rest of the code: cfg -> Optax transform.

    Example usage in your training code:

        from optim import muon as muon_opt

        tx = muon_opt.build_muon_tx_from_cfg(config.optim)

        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
        )

    By default this:
      - applies Muon to all 2D parameters (ndim == 2),
      - uses AdamW (with adam_weight_decay) for everything else.
    """
    mu_cfg = _muon_config_from_generic_config(cfg)

    tx = optax.contrib.muon(
        learning_rate=mu_cfg.learning_rate,
        ns_coeffs=mu_cfg.ns_coeffs,
        ns_steps=mu_cfg.ns_steps,
        beta=mu_cfg.beta,
        eps=mu_cfg.eps,
        mu_dtype=None,  # let Optax choose (usually same as params)
        nesterov=mu_cfg.nesterov,
        adaptive=mu_cfg.adaptive,
        adam_b1=mu_cfg.adam_b1,
        adam_b2=mu_cfg.adam_b2,
        adam_eps_root=mu_cfg.adam_eps_root,
        adam_weight_decay=mu_cfg.weight_decay,
        # If you later want to apply Muon to conv kernels, you can also pass
        #   muon_weight_dimension_numbers=...
        # following the Optax Muon example notebook.
    )

    return tx

def build_muon_dim_numbers(params):
    """Return a pytree of same structure, with MuonDimensionNumbers or None."""
    def leaf_fn(path, leaf):
        # path is a tuple of keys (if you use tree_map_with_path),
        # but we can also just inspect the leaf's shape.
        if hasattr(leaf, "ndim"):
            if leaf.ndim == 2:
                # Dense weights: (in_features, out_features)
                return optax.contrib.MuonDimensionNumbers(0, 1)
            elif leaf.ndim == 4:
                # Conv weights: (Kh, Kw, Cin, Cout)
                return optax.contrib.MuonDimensionNumbers((0, 1, 2), (3,))
        # Biases, batchnorm params, etc. -> let AdamW handle them
        return None

    # If you don't care about using the path, you can just map over leaves.
    return jtu.tree_map(leaf_fn, params)


def build_muon_tx(
    learning_rate: float,
    weight_decay: float = 0.0,
    *,
    beta: float = 0.95,
    ns_steps: int = 5,
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    eps: float = 1e-8,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps_root: float = 0.0,
) -> optax.GradientTransformation:
    """Simpler functional interface if you don’t want to go through cfg.

    Example:

        tx = build_muon_tx(
            learning_rate=1e-3,
            weight_decay=0.01,
        )
    """
    cfg = MuonConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta=beta,
        ns_steps=ns_steps,
        ns_coeffs=ns_coeffs,
        eps=eps,
        nesterov=nesterov,
        adaptive=adaptive,
        adam_b1=adam_b1,
        adam_b2=adam_b2,
        adam_eps_root=adam_eps_root,
    )
    return optax.contrib.muon(
        learning_rate=cfg.learning_rate,
        ns_coeffs=cfg.ns_coeffs,
        ns_steps=cfg.ns_steps,
        beta=cfg.beta,
        eps=cfg.eps,
        mu_dtype=None,
        nesterov=cfg.nesterov,
        adaptive=cfg.adaptive,
        adam_b1=cfg.adam_b1,
        adam_b2=cfg.adam_b2,
        adam_eps_root=cfg.adam_eps_root,
        adam_weight_decay=cfg.weight_decay,
    )
