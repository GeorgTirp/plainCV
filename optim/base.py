# optim/base.py
from typing import NamedTuple, Any, Optional
import optax


# Alias: any Optax-style gradient transformation
GradientTransformation = optax.GradientTransformation


class StepState(NamedTuple):
    """Minimal generic optimizer state with step counter."""
    inner_state: Any
    step: int
