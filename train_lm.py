# train_lm.py
"""Train a causal language model in JAX/Flax."""
import os
import sys
import time
from functools import partial
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
import numpy as np
import optax
from flax import jax_utils as flax_jax_utils
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


def _reshape_for_pmap(x: Optional[jnp.ndarray], n_devices: int):
    if x is None:
        return None
    if x.shape[0] % n_devices != 0:
        raise ValueError(
            f"Leading batch dimension {x.shape[0]} is not divisible by n_devices={n_devices}."
        )
    per_device = x.shape[0] // n_devices
    return x.reshape((n_devices, per_device) + x.shape[1:])


def _prepare_batch_for_devices(batch, seq_len: int, use_doc_mask: bool, n_devices: int):
    inputs, labels, attn_mask = _prepare_batch(batch, seq_len, use_doc_mask)
    if n_devices <= 1:
        return inputs, labels, attn_mask
    return (
        _reshape_for_pmap(inputs, n_devices),
        _reshape_for_pmap(labels, n_devices),
        _reshape_for_pmap(attn_mask, n_devices),
    )


def _clip_grads(grads, max_norm: Optional[float]):
    if max_norm is None:
        return grads
    g_norm = optax.global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (g_norm + 1e-6))
    return jtu.tree_map(lambda g: g * scale, grads)


def _loss_and_acc(logits, labels):
    # Keep numerically sensitive softmax-cross-entropy in fp32 regardless of model compute dtype.
    logits_f32 = logits.astype(jnp.float32)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits_f32, labels).mean()
    acc = jnp.mean(jnp.argmax(logits_f32, axis=-1) == labels)
    return loss, acc


def _make_train_fns(model, vocab_size: int, use_doc_mask: bool, use_pmap: bool):
    del vocab_size

    if use_doc_mask:
        if use_pmap:

            @partial(jax.pmap, axis_name="data")
            def compute_grads(params, inputs, labels, attn_mask):
                def loss_fn(p):
                    logits = model.apply(
                        {"params": p},
                        inputs,
                        attn_mask=attn_mask,
                        deterministic=True,
                    )
                    return _loss_and_acc(logits, labels)

                (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
                grads = jax.lax.pmean(grads, axis_name="data")
                loss = jax.lax.pmean(loss, axis_name="data")
                acc = jax.lax.pmean(acc, axis_name="data")
                return grads, loss, acc

            @partial(jax.pmap, axis_name="data")
            def eval_step(params, inputs, labels, attn_mask):
                logits = model.apply(
                    {"params": params},
                    inputs,
                    attn_mask=attn_mask,
                    deterministic=True,
                )
                loss, acc = _loss_and_acc(logits, labels)
                loss = jax.lax.pmean(loss, axis_name="data")
                acc = jax.lax.pmean(acc, axis_name="data")
                return loss, acc

            return compute_grads, eval_step

        @jax.jit
        def compute_grads(params, inputs, labels, attn_mask):
            def loss_fn(p):
                logits = model.apply(
                    {"params": p},
                    inputs,
                    attn_mask=attn_mask,
                    deterministic=True,
                )
                return _loss_and_acc(logits, labels)

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
            loss, acc = _loss_and_acc(logits, labels)
            return loss, acc

        return compute_grads, eval_step

    if use_pmap:

        @partial(jax.pmap, axis_name="data")
        def compute_grads(params, inputs, labels):
            def loss_fn(p):
                logits = model.apply(
                    {"params": p},
                    inputs,
                    attn_mask=None,
                    deterministic=True,
                )
                return _loss_and_acc(logits, labels)

            (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            grads = jax.lax.pmean(grads, axis_name="data")
            loss = jax.lax.pmean(loss, axis_name="data")
            acc = jax.lax.pmean(acc, axis_name="data")
            return grads, loss, acc

        @partial(jax.pmap, axis_name="data")
        def eval_step(params, inputs, labels):
            logits = model.apply(
                {"params": params},
                inputs,
                attn_mask=None,
                deterministic=True,
            )
            loss, acc = _loss_and_acc(logits, labels)
            loss = jax.lax.pmean(loss, axis_name="data")
            acc = jax.lax.pmean(acc, axis_name="data")
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
            return _loss_and_acc(logits, labels)

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
        loss, acc = _loss_and_acc(logits, labels)
        return loss, acc

    return compute_grads, eval_step


def _make_apply_grads_fn(grad_clip: Optional[float], use_pmap: bool):
    if use_pmap:

        @jax.pmap
        def apply_grads(state, grads):
            grads = _clip_grads(grads, grad_clip)
            updates, new_opt_state = state.tx.update(
                grads,
                state.opt_state,
                state.params,
            )
            new_params = optax.apply_updates(state.params, updates)
            new_state = state.replace(
                step=state.step + 1,
                params=new_params,
                opt_state=new_opt_state,
            )
            return new_state, updates

        return apply_grads

    @jax.jit
    def apply_grads(state, grads):
        grads = _clip_grads(grads, grad_clip)
        updates, new_opt_state = state.tx.update(
            grads,
            state.opt_state,
            state.params,
        )
        new_params = optax.apply_updates(state.params, updates)
        new_state = state.replace(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )
        return new_state, updates

    return apply_grads


def _merge_batches(batches):
    if len(batches) == 1:
        return batches[0]

    merged = {
        "input_ids": np.concatenate(
            [np.asarray(batch["input_ids"]) for batch in batches],
            axis=0,
        )
    }
    if "docs_lengths" in batches[0]:
        merged["docs_lengths"] = []
        for batch in batches:
            merged["docs_lengths"].extend(batch["docs_lengths"])
    return merged


def _next_batch(it, loader, num_batches: int = 1):
    batches = []
    for _ in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batches.append(batch)
    return _merge_batches(batches), it


def _iter_grouped_batches(loader, num_batches: int = 1):
    if num_batches == 1:
        yield from loader
        return

    it = iter(loader)
    while True:
        batches = []
        for _ in range(num_batches):
            try:
                batches.append(next(it))
            except StopIteration:
                return
        yield _merge_batches(batches)


def _unwrap_replicated(x, use_pmap: bool):
    if not use_pmap:
        return x
    return flax_jax_utils.unreplicate(x)


def _should_run_eigen_tracking_for_step(cfg, completed_step: int) -> bool:
    """Return whether eigen tracking should run after the completed train step."""
    eigen_tracking_every = int(getattr(cfg, "eigen_tracking_every", 100))
    if eigen_tracking_every <= 0:
        raise ValueError("eigen_tracking_every must be >= 1 when tracking is enabled.")

    if not bool(getattr(cfg, "eigen_tracking_post_soap_refresh", False)):
        return (completed_step % eigen_tracking_every) == 0

    if str(getattr(cfg, "optim", "")).lower() != "soap":
        raise ValueError(
            "eigen_tracking_post_soap_refresh=True is only supported with optim='soap'."
        )

    precondition_frequency = int(getattr(cfg, "precondition_frequency", 0))
    if precondition_frequency <= 0:
        raise ValueError(
            "eigen_tracking_post_soap_refresh=True requires precondition_frequency >= 1."
        )
    if eigen_tracking_every % precondition_frequency != 0:
        raise ValueError(
            "With eigen_tracking_post_soap_refresh=True, eigen_tracking_every must "
            "be a positive multiple of precondition_frequency."
        )

    # SOAP initializes the basis on the first optimizer step without applying an
    # actual update. The first step that uses a refreshed basis therefore occurs
    # one train step after the first QR refresh.
    first_post_refresh_step = precondition_frequency + 2
    return (
        completed_step >= first_post_refresh_step
        and ((completed_step - first_post_refresh_step) % eigen_tracking_every) == 0
    )


def _probe_pmap_collectives(n_devices: int) -> tuple[bool, Optional[str]]:
    """Return whether pmap collectives are usable on this host.

    Some cluster setups expose multiple GPUs but lack a usable NCCL runtime.
    In that case, pmap + psum/pmean fails at runtime. We probe once and
    gracefully fall back to single-device execution if needed.
    """
    if n_devices <= 1:
        return False, None

    try:
        test = jnp.arange(n_devices, dtype=jnp.float32)

        @partial(jax.pmap, axis_name="data")
        def _psum(x):
            return jax.lax.psum(x, axis_name="data")

        _ = jax.device_get(_psum(test))
        return True, None
    except Exception as exc:
        return False, str(exc)


def run(cfg):
    if cfg.model != "transformer" and not str(cfg.model).startswith("pythia"):
        raise ValueError(f"LM training expects model='transformer' or 'pythia*', got {cfg.model}.")

    wb_run = init_wandb(cfg)
    if wb_run is not None:
        wb_run.define_metric("step")
        wb_run.define_metric("*", step_metric="step")
        wb_run.define_metric("eval_loss", summary="min")

    use_doc_mask = bool(getattr(cfg, "intra_doc_masking", False))
    n_local_devices = jax.local_device_count()
    requested_use_pmap = getattr(cfg, "use_pmap", None)
    force_single_device = bool(getattr(cfg, "force_single_device", False))

    if requested_use_pmap is None:
        use_pmap = n_local_devices > 1 and not force_single_device
    else:
        use_pmap = bool(requested_use_pmap) and n_local_devices > 1 and not force_single_device

    if use_pmap:
        pmap_ok, pmap_err = _probe_pmap_collectives(n_local_devices)
        if not pmap_ok:
            print(
                "Disabling pmap and falling back to single-device training because "
                f"multi-device collectives are unavailable (likely NCCL issue): {pmap_err}"
            )
            use_pmap = False

    n_devices = n_local_devices if use_pmap else 1
    grouped_batches = n_devices if use_pmap else 1

    if use_pmap:
        print(f"Using pmap over {n_devices} local devices.")
    else:
        if n_local_devices > 1:
            print(
                f"Using single-device mode on 1/{n_local_devices} visible devices "
                "(pmap disabled)."
            )
        else:
            print("Using single-device mode.")

    maybe_make_dir(cfg)

    trainloader, validloader = get_dataloaders(cfg)

    # Build a deterministic curvature batch (first batch) for curvature-based optimizers.
    # This is safe for all optimizers and required for PNS/Sophia/HF-like methods.
    curvature_batch = None
    try:
        curv_raw, _ = _next_batch(iter(trainloader), trainloader, num_batches=grouped_batches)
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
    if use_pmap:
        state = flax_jax_utils.replicate(state)

    eigen_tracking_enabled = bool(getattr(cfg, "eigen_tracking_enabled", False))
    eigen_tracking_state = None
    eigen_tracking_csv_path = None
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
        eigen_tracking_extra_modes = int(
            getattr(cfg, "eigen_tracking_extra_modes", 0)
        )
        if eigen_tracking_extra_modes < 0:
            raise ValueError("eigen_tracking_extra_modes must be >= 0.")

        eigen_tracking_backend = getattr(
            cfg,
            "eigen_tracking_backend",
            getattr(cfg, "pns_curvature_backend", "ggn"),
        )
        eigen_tracking_iters = getattr(cfg, "eigen_tracking_lanczos_iters", None)
        if eigen_tracking_iters is None:
            eigen_tracking_iters = (
                eigen_tracking_topk + eigen_tracking_extra_modes
            )
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
        eigen_params = _unwrap_replicated(state.params, use_pmap)
        eigen_tracking_state = init_eigentracking(
            eigen_params,
            k=eigen_tracking_topk,
            extra_modes=eigen_tracking_extra_modes,
            seed=int(getattr(cfg, "seed", 0)),
        )
        eigen_tracking_csv_path = init_eigen_tracking_csv(
            cfg,
            eigen_tracking_topk,
            extra_modes=eigen_tracking_extra_modes,
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

    compute_grads, eval_step = _make_train_fns(
        model,
        cfg.vocab_size,
        use_doc_mask,
        use_pmap=use_pmap,
    )

    steps_budget = int(getattr(cfg, "steps_budget", 100))
    grad_accum_steps = int(getattr(cfg, "grad_accumulation_steps", 1))
    log_every = int(getattr(cfg, "log_every_steps", 10))
    eval_every = getattr(cfg, "eval_every_steps", None)
    eval_every = int(eval_every) if eval_every is not None else None
    grad_clip = getattr(cfg, "grad_clip", None)
    world_size = int(jax.process_count()) * int(n_devices)
    tokens_per_step = (
        int(cfg.seq_len)
        * int(cfg.micro_batch_size)
        * grad_accum_steps
        * world_size
    )
    apply_grads = _make_apply_grads_fn(grad_clip, use_pmap=use_pmap)

    train_iter = iter(trainloader)
    global_step = 0
    start_time = time.time()

    while global_step < steps_budget:
        grads_accum = None
        loss_accum = None
        acc_accum = None

        for _ in range(grad_accum_steps):
            batch, train_iter = _next_batch(train_iter, trainloader, num_batches=grouped_batches)
            inputs, labels, attn_mask = _prepare_batch_for_devices(
                batch,
                cfg.seq_len,
                use_doc_mask,
                n_devices,
            )

            if use_doc_mask:
                grads, loss, acc = compute_grads(state.params, inputs, labels, attn_mask)
            else:
                grads, loss, acc = compute_grads(state.params, inputs, labels)

            grads_accum = grads if grads_accum is None else jtu.tree_map(
                lambda a, b: a + b, grads_accum, grads
            )
            loss_accum = loss if loss_accum is None else loss_accum + loss
            acc_accum = acc if acc_accum is None else acc_accum + acc

        grads_accum = jtu.tree_map(lambda g: g / grad_accum_steps, grads_accum)
        params_before = _unwrap_replicated(state.params, use_pmap) if eigen_tracking_enabled else None
        state, updates = apply_grads(state, grads_accum)

        global_step += 1

        if eigen_tracking_enabled and _should_run_eigen_tracking_for_step(cfg, global_step):
            eigen_tracking_state = run_eigen_tracking(
                params_before,
                _unwrap_replicated(grads_accum, use_pmap),
                _unwrap_replicated(updates, use_pmap),
                _unwrap_replicated(state.step, use_pmap),
                eigen_tracking_state,
            )
            append_eigen_tracking_row(
                eigen_tracking_csv_path,
                eigen_tracking_state,
            )

        if global_step % log_every == 0:
            loss_val = float(
                jax.device_get(_unwrap_replicated(loss_accum / grad_accum_steps, use_pmap))
            )
            acc_val = float(
                jax.device_get(_unwrap_replicated(acc_accum / grad_accum_steps, use_pmap))
            )
            tokens_seen = int(global_step * tokens_per_step)
            metrics = {
                "step": global_step,
                "tokens_seen": tokens_seen,
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
            for batch in _iter_grouped_batches(validloader, num_batches=grouped_batches):
                inputs, labels, attn_mask = _prepare_batch_for_devices(
                    batch,
                    cfg.seq_len,
                    use_doc_mask,
                    n_devices,
                )
                if use_doc_mask:
                    loss, acc = eval_step(state.params, inputs, labels, attn_mask)
                else:
                    loss, acc = eval_step(state.params, inputs, labels)
                eval_loss += _unwrap_replicated(loss, use_pmap)
                eval_acc += _unwrap_replicated(acc, use_pmap)
                n_batches += 1
            eval_loss = float(jax.device_get(eval_loss / max(1, n_batches)))
            eval_acc = float(jax.device_get(eval_acc / max(1, n_batches)))
            tokens_seen = int(global_step * tokens_per_step)
            metrics = {
                "step": global_step,
                "tokens_seen": tokens_seen,
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
