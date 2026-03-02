# train_lm.py
"""Train a causal language model in JAX/Flax."""
import os
import sys
import time
from typing import Iterable, Tuple, Optional

import yaml

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _parse_config_path(argv) -> Optional[str]:
    for i, arg in enumerate(argv):
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _apply_preimport_config():
    if os.environ.get("JAX_PLATFORM_NAME") or os.environ.get("JAX_PLATFORMS"):
        return

    force = os.environ.get("FORCE_CPU", "").lower() in {"1", "true", "yes"}
    matmul_precision = None
    cfg_path = _parse_config_path(sys.argv)

    if cfg_path:
        try:
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            force = bool(cfg.get("force_cpu", force))
            matmul_precision = cfg.get("matmul_precision", None)
        except Exception:
            pass

    if force:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if matmul_precision and "JAX_DEFAULT_MATMUL_PRECISION" not in os.environ:
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = str(matmul_precision)


_apply_preimport_config()


from absl import app, flags
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import tree_util as jtu

from data.lm_loader import get_dataloaders
from data.datasets.data_prep_utils import intra_doc_causal_mask
from models.LM.constructor import construct_model
from optim.eigentools import init_eigentracking, track_eigenstate
from optim.factory import get_optimizer, build_curvature_matvec_fn
from utils import (
    load_config,
    maybe_make_dir,
    init_wandb,
    log_scalar_dict,
    init_eigen_tracking_csv,
    append_eigen_tracking_row,
)

FLAGS = flags.FLAGS


def _define_flags():
    if "config" not in FLAGS:
        flags.DEFINE_string("config", "config/lm.yaml", "Path to LM config.yaml file.")
    if "exp_name" not in FLAGS:
        flags.DEFINE_string(
            "exp_name",
            None,
            "Override exp_name from the config for the output folder.",
        )


_define_flags()


class LMTrainState(train_state.TrainState):
    pass


def _trim_last_token(doc_boundaries):
    trimmed = list(doc_boundaries)
    if not trimmed:
        return trimmed
    trimmed[-1] -= 1
    if trimmed[-1] <= 0:
        trimmed.pop()
    return trimmed


def _build_attn_mask(batch, seq_len: int) -> jnp.ndarray:
    docs_lengths = batch.get("docs_lengths", None)
    if docs_lengths is None:
        raise ValueError("intra_doc_masking=True but docs_lengths not found in batch.")

    full_len = batch["input_ids"].shape[1]
    if full_len != seq_len + 1:
        raise ValueError(
            f"Expected input_ids length {seq_len + 1} (seq_len+1) but got {full_len}."
        )

    masks = []
    for boundaries in docs_lengths:
        boundaries = _trim_last_token(boundaries)
        if sum(boundaries) != seq_len:
            raise ValueError(
                f"Sum(doc_boundaries)={sum(boundaries)} != seq_len={seq_len}."
            )
        mask = intra_doc_causal_mask(boundaries, seq_len, device="cpu")
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        mask = jnp.asarray(mask, dtype=jnp.bool_)
        masks.append(mask)

    return jnp.stack(masks, axis=0)


def _prepare_batch(batch, seq_len: int, use_doc_mask: bool):
    input_ids = jnp.asarray(batch["input_ids"], dtype=jnp.int32)
    if input_ids.shape[1] != seq_len + 1:
        raise ValueError(
            f"Expected input_ids length {seq_len + 1} (seq_len+1) but got {input_ids.shape[1]}."
        )

    inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:]

    attn_mask = None
    if use_doc_mask:
        attn_mask = _build_attn_mask(batch, seq_len)

    return inputs, labels, attn_mask


def _clip_grads(grads, max_norm: Optional[float]):
    if max_norm is None:
        return grads
    g_norm = optax.global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (g_norm + 1e-6))
    return jtu.tree_map(lambda g: g * scale, grads)


def _make_train_fns(model, vocab_size: int, use_doc_mask: bool):
    if use_doc_mask:

        @jax.jit
        def compute_grads(params, inputs, labels, attn_mask):
            def loss_fn(p):
                logits = model.apply(
                    {"params": p},
                    inputs,
                    attn_mask=attn_mask,
                    deterministic=True,
                )
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
                acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
                return loss, acc

            (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            return grads, loss, acc

        @jax.jit
        def eval_step(params, inputs, labels, attn_mask):
            logits = model.apply(
                {"params": params},
                inputs,
                attn_mask=attn_mask,
                deterministic=True,
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
            return loss, acc

        return compute_grads, eval_step

    @jax.jit
    def compute_grads(params, inputs, labels):
        def loss_fn(p):
            logits = model.apply(
                {"params": p},
                inputs,
                attn_mask=None,
                deterministic=True,
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
            return loss, acc

        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return grads, loss, acc

    @jax.jit
    def eval_step(params, inputs, labels):
        logits = model.apply(
            {"params": params},
            inputs,
            attn_mask=None,
            deterministic=True,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
        return loss, acc

    return compute_grads, eval_step


def _next_batch(it, loader):
    try:
        batch = next(it)
        return batch, it
    except StopIteration:
        it = iter(loader)
        batch = next(it)
        return batch, it


def run(cfg):
    if cfg.model != "transformer" and not str(cfg.model).startswith("pythia"):
        raise ValueError(f"LM training expects model='transformer' or 'pythia*', got {cfg.model}.")

    use_doc_mask = bool(getattr(cfg, "intra_doc_masking", False))

    maybe_make_dir(cfg)
    init_wandb(cfg)

    trainloader, validloader = get_dataloaders(cfg)

    # Build a deterministic curvature batch (first batch) for curvature-based optimizers.
    # This is safe for all optimizers and required for PNS/Sophia/HF-like methods.
    curvature_batch = None
    try:
        curv_raw = next(iter(trainloader))
        curv_inputs, curv_labels, curv_attn_mask = _prepare_batch(curv_raw, cfg.seq_len, use_doc_mask)
        curvature_batch = (curv_inputs, curv_labels, curv_attn_mask)
    except Exception as exc:
        # If an optimizer needs curvature, it'll raise later with a clearer message.
        print(f"Warning: could not build curvature_batch: {exc}")

    rng = jax.random.PRNGKey(getattr(cfg, "seed", 0))
    model, model_cfg, variables = construct_model(cfg, rng=rng, init_batch_size=cfg.micro_batch_size)
    params = variables["params"]

    tx = get_optimizer(cfg, model_def=model, curvature_batch=curvature_batch, batch_stats=None)
    state = LMTrainState.create(apply_fn=model.apply, params=params, tx=tx)

    eigen_tracking_enabled = bool(getattr(cfg, "eigen_tracking_enabled", False))
    eigen_tracking_state = None
    eigen_tracking_csv_path = None
    eigen_tracking_every = int(getattr(cfg, "eigen_tracking_every", 100))
    if eigen_tracking_enabled:
        if curvature_batch is None:
            raise ValueError(
                "eigen_tracking_enabled=True requires a valid curvature_batch."
            )

        eigen_tracking_topk = int(
            getattr(
                cfg,
                "eigen_tracking_topk",
                getattr(cfg, "curvature_eigenvectors", 8),
            )
        )
        if eigen_tracking_topk <= 0:
            raise ValueError("eigen_tracking_topk must be >= 1 when tracking is enabled.")

        eigen_tracking_backend = getattr(
            cfg,
            "eigen_tracking_backend",
            getattr(cfg, "pns_curvature_backend", "ggn"),
        )
        eigen_tracking_iters = getattr(cfg, "eigen_tracking_lanczos_iters", None)
        if eigen_tracking_iters is None:
            eigen_tracking_iters = eigen_tracking_topk
        eigen_tracking_sort_by_abs = getattr(cfg, "eigen_tracking_sort_by_abs", None)
        if eigen_tracking_sort_by_abs is None:
            eigen_tracking_sort_by_abs = (
                str(eigen_tracking_backend).lower() in {"hessian", "fisher"}
            )
        eigen_tracking_light_ortho = bool(
            getattr(cfg, "eigen_tracking_light_ortho", True)
        )
        eigen_tracking_light_ortho_every = int(
            getattr(cfg, "eigen_tracking_light_ortho_every", 4)
        )
        eigen_tracking_matvec_fn = build_curvature_matvec_fn(
            cfg,
            model_def=model,
            curvature_batch=curvature_batch,
            batch_stats=None,
            backend=eigen_tracking_backend,
        )
        eigen_tracking_state = init_eigentracking(
            state.params,
            k=eigen_tracking_topk,
            seed=int(getattr(cfg, "seed", 0)),
        )
        eigen_tracking_csv_path = init_eigen_tracking_csv(
            cfg,
            eigen_tracking_topk,
        )

        @jax.jit
        def run_eigen_tracking(params, grads, updates, step, tracking_state):
            return track_eigenstate(
                params=params,
                grads=grads,
                updates=updates,
                step=step,
                eigen_state=tracking_state,
                matvec_fn=eigen_tracking_matvec_fn,
                num_iter=eigen_tracking_iters,
                sort_by_abs=eigen_tracking_sort_by_abs,
                use_light_ortho=eigen_tracking_light_ortho,
                light_ortho_every=eigen_tracking_light_ortho_every,
            )

    compute_grads, eval_step = _make_train_fns(model, cfg.vocab_size, use_doc_mask)

    steps_budget = int(getattr(cfg, "steps_budget", 100))
    grad_accum_steps = int(getattr(cfg, "grad_accumulation_steps", 1))
    log_every = int(getattr(cfg, "log_every_steps", 10))
    eval_every = getattr(cfg, "eval_every_steps", None)
    eval_every = int(eval_every) if eval_every is not None else None
    grad_clip = getattr(cfg, "grad_clip", None)

    train_iter = iter(trainloader)
    global_step = 0
    start_time = time.time()

    while global_step < steps_budget:
        grads_accum = None
        loss_accum = 0.0
        acc_accum = 0.0

        for _ in range(grad_accum_steps):
            batch, train_iter = _next_batch(train_iter, trainloader)
            inputs, labels, attn_mask = _prepare_batch(batch, cfg.seq_len, use_doc_mask)

            if use_doc_mask:
                grads, loss, acc = compute_grads(state.params, inputs, labels, attn_mask)
            else:
                grads, loss, acc = compute_grads(state.params, inputs, labels)

            grads_accum = grads if grads_accum is None else jtu.tree_map(
                lambda a, b: a + b, grads_accum, grads
            )
            loss_accum += loss
            acc_accum += acc

        grads_accum = jtu.tree_map(lambda g: g / grad_accum_steps, grads_accum)
        grads_accum = _clip_grads(grads_accum, grad_clip)
        params_before = state.params if eigen_tracking_enabled else None
        updates, new_opt_state = state.tx.update(
            grads_accum,
            state.opt_state,
            state.params,
        )
        new_params = optax.apply_updates(state.params, updates)
        state = state.replace(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )

        global_step += 1

        if eigen_tracking_enabled and global_step % eigen_tracking_every == 0:
            eigen_tracking_state = run_eigen_tracking(
                params_before,
                grads_accum,
                updates,
                state.step,
                eigen_tracking_state,
            )
            append_eigen_tracking_row(
                eigen_tracking_csv_path,
                eigen_tracking_state,
            )

        if global_step % log_every == 0:
            loss_val = float(jax.device_get(loss_accum / grad_accum_steps))
            acc_val = float(jax.device_get(acc_accum / grad_accum_steps))
            metrics = {
                "step": global_step,
                "train_loss": loss_val,
                "train_acc": acc_val,
                "train_ppl": float(jnp.exp(loss_val)),
                "elapsed_s": time.time() - start_time,
            }
            log_scalar_dict(cfg, metrics)

        if eval_every is not None and validloader is not None and global_step % eval_every == 0:
            eval_loss = 0.0
            eval_acc = 0.0
            n_batches = 0
            for batch in validloader:
                inputs, labels, attn_mask = _prepare_batch(batch, cfg.seq_len, use_doc_mask)
                if use_doc_mask:
                    loss, acc = eval_step(state.params, inputs, labels, attn_mask)
                else:
                    loss, acc = eval_step(state.params, inputs, labels)
                eval_loss += loss
                eval_acc += acc
                n_batches += 1
            eval_loss = float(jax.device_get(eval_loss / max(1, n_batches)))
            eval_acc = float(jax.device_get(eval_acc / max(1, n_batches)))
            metrics = {
                "step": global_step,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "eval_ppl": float(jnp.exp(eval_loss)),
            }
            log_scalar_dict(cfg, metrics)

    print("Training complete.")


def main(_argv):
    cfg, _ = load_config(FLAGS.config)
    run(cfg)


if __name__ == "__main__":
    app.run(main)
