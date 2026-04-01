from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def path_to_name(path) -> str:
    parts = []
    for key in path:
        if hasattr(key, "key"):
            parts.append(str(key.key))
        elif hasattr(key, "name"):
            parts.append(str(key.name))
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return "/".join(parts)


def is_non_degenerate_2d_matrix(p: Any) -> bool:
    p_arr = jnp.asarray(p)
    return p_arr.ndim == 2 and p_arr.shape[0] > 1 and p_arr.shape[1] > 1


def should_use_matrix_preconditioner(path, p: Any) -> bool:
    """Shared routing for matrix preconditioners (SOAP/Muon/Shampoo style)."""
    if not is_non_degenerate_2d_matrix(p):
        return False

    name = path_to_name(path).lower()
    leaf = name.split("/")[-1] if name else ""
    if leaf != "kernel":
        return False
    if ("embed" in name) or ("embedding" in name) or ("lm_head" in name):
        return False
    if "norm" in name:
        return False
    return True
