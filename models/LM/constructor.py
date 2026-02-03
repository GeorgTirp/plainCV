from fractions import Fraction
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import wandb

from flax.traverse_util import flatten_dict, unflatten_dict


def _count_params(params: Dict[str, Any]) -> int:
    """Count total number of scalars in a Flax params pytree."""
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.prod(x.shape) for x in leaves))


def _count_non_embedding_params(params: Dict[str, Any]) -> int:
    """
    Heuristic "non-embedding params" count, similar to your Torch count_params(non_embedding=True).

    For your Transformer implementation, we exclude:
      - embed_tokens/* (token embedding table)
      - lm_head/* if it exists and is tied/unwanted in your accounting
    """
    flat = flatten_dict(params, sep="/")
    keep = {}
    for k, v in flat.items():
        key = k if isinstance(k, str) else "/".join(k)
        if key.startswith("embed_tokens/"):
            continue
        # If you want to exclude untied lm_head as well, uncomment:
        # if key.startswith("lm_head/"):
        #     continue
        keep[k] = v
    return _count_params(unflatten_dict(keep, sep="/"))


def construct_model(
    cfg,
    rng: Optional[jax.random.PRNGKey] = None,
    init_batch_size: int = 1,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Initialize a model from config and count parameters.

    Returns:
      model: Flax Module (or HF Flax model wrapper)
      model_cfg: model-specific config
      variables: dict with at least {"params": ...} (and possibly batch_stats, etc.)
    """
    if rng is None:
        rng = jax.random.PRNGKey(getattr(cfg, "seed", 0))

    # -------------------------
    # Transformer++ (your Flax Transformer)
    # -------------------------
    if cfg.model == "transformer":
        from .transformer import Transformer, ModelConfig

        model_cfg = ModelConfig(
            vocab_size=cfg.vocab_size,
            dim=cfg.d_model,
            expand=float(Fraction(cfg.expand)),
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            rmsnorm_eps=1e-6,
            mlp=cfg.mlp_class,
            seq_len=cfg.seq_len,
            tie_embeddings=cfg.tie_embeddings,
            rope_theta=getattr(cfg, "rope_theta", 500000.0),
        )
        model = Transformer(model_cfg)

        # dummy inputs for init
        input_ids = jnp.zeros((init_batch_size, cfg.seq_len), dtype=jnp.int32)
        attn_mask = None  # or create one if your training always uses a mask

        variables = model.init(rng, input_ids, attn_mask=attn_mask, deterministic=True)

    # -------------------------
    # Pythia (HF Transformers Flax)
    # -------------------------
    elif cfg.model.startswith("pythia"):
        # Hugging Face Flax models store params separately and often use init_weights.
        from transformers import AutoConfig, FlaxAutoModelForCausalLM

        model_cfg = AutoConfig.from_pretrained(f"EleutherAI/{cfg.model}")
        model = FlaxAutoModelForCausalLM.from_config(model_cfg, dtype=jnp.float32)

        # HF Flax models: init_weights(rng, input_shape=...) returns params
        input_shape = (init_batch_size, cfg.seq_len)
        params = model.init_weights(rng, input_shape=input_shape)
        variables = {"params": params}

    else:
        raise NotImplementedError(f"Not implemented model: {cfg.model}.")

    # -------------------------
    # Parameter counts + W&B
    # -------------------------
    if "params" in variables:
        n_params = _count_params(variables["params"])
        n_params_no_embed = _count_non_embedding_params(variables["params"])

        print(f"Number of parameters: {n_params:_}")
        print(f"Number of non-embedding parameters: {n_params_no_embed:_}")

        if wandb.run is not None:
            wandb.log({"n_params": n_params, "n_params_no_embed": n_params_no_embed})

    return model, model_cfg, variables
