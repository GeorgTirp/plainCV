# train_lm.py
"""Train a causal language model in JAX/Flax."""
import os
import sys
import time
from functools import partial
from typing import Iterable, Tuple, Optional
import yaml

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
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

from models.LM.constructor import construct_model
from optim.eigentools import init_eigentracking, track_eigenstate
from optim.factory import get_optimizer, build_lm_dynamic_curvature_matvec_fn
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
    from data.datasets.data_prep_utils import intra_doc_causal_mask

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


def _shard_batch_axis_for_pmap(x: Optional[jnp.ndarray], n_devices: int):
    if x is None:
        return None
    return _reshape_for_pmap(x, n_devices)


def _shard_probe_axis_batch_for_pmap(x: Optional[jnp.ndarray], n_devices: int):
    """Shard arrays shaped (num_probe_batches, global_batch, ...)."""
    if x is None:
        return None
    num_probe_batches = x.shape[0]
    global_batch = x.shape[1]
    if global_batch % n_devices != 0:
        raise ValueError(
            "Probe batch dimension "
            f"{global_batch} is not divisible by n_devices={n_devices}."
        )
    per_device = global_batch // n_devices
    reshaped = x.reshape(
        (num_probe_batches, n_devices, per_device) + x.shape[2:]
    )
    return jnp.swapaxes(reshaped, 0, 1)


def _shard_curvature_batch_for_pmap(curvature_batch, n_devices: int):
    input_ids, labels, attn_mask = curvature_batch
    if n_devices <= 1:
        return curvature_batch
    if input_ids.ndim == 2:
        return (
            _shard_batch_axis_for_pmap(input_ids, n_devices),
            _shard_batch_axis_for_pmap(labels, n_devices),
            _shard_batch_axis_for_pmap(attn_mask, n_devices),
        )
    if input_ids.ndim == 3:
        return (
            _shard_probe_axis_batch_for_pmap(input_ids, n_devices),
            _shard_probe_axis_batch_for_pmap(labels, n_devices),
            _shard_probe_axis_batch_for_pmap(attn_mask, n_devices),
        )
    if input_ids.ndim >= 4 and input_ids.shape[0] == n_devices:
        return curvature_batch
    raise ValueError(
        "Expected LM curvature input_ids to have shape (batch, seq), "
        "(num_probe_batches, batch, seq), or already-sharded leading device "
        f"axis; got {input_ids.shape}."
    )


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


def _make_probe_measurement_fn(
    model,
    tx,
    curvature_batch,
    grad_clip: Optional[float],
    use_doc_mask: bool,
    use_pmap: bool = False,
    n_devices: int = 1,
):
    probe_inputs, probe_labels, probe_attn_mask = curvature_batch

    def probe_loss_acc_for_batch(params, inputs, labels, attn_mask):
        logits = model.apply(
            {"params": params},
            inputs,
            attn_mask=attn_mask if use_doc_mask else None,
            deterministic=True,
        )
        return _loss_and_acc(logits, labels)

    def probe_grad_for_batch(params, inputs, labels, attn_mask):
        def loss_fn(p):
            return probe_loss_acc_for_batch(p, inputs, labels, attn_mask)

        (_loss, _acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return grads

    if use_pmap:
        sharded_probe_inputs, sharded_probe_labels, sharded_probe_attn_mask = (
            _shard_curvature_batch_for_pmap(curvature_batch, n_devices)
        )

        @partial(jax.pmap, axis_name="data")
        def compute_probe_measurement_pmapped(
            params,
            opt_state,
            local_probe_inputs,
            local_probe_labels,
            local_probe_attn_mask,
        ):
            if local_probe_inputs.ndim == 2:
                local_grads = probe_grad_for_batch(
                    params,
                    local_probe_inputs,
                    local_probe_labels,
                    local_probe_attn_mask,
                )
                local_loss_before, local_acc_before = probe_loss_acc_for_batch(
                    params,
                    local_probe_inputs,
                    local_probe_labels,
                    local_probe_attn_mask,
                )
            else:
                num_probe_batches = int(local_probe_inputs.shape[0])
                grads_sum = None
                loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
                acc_sum = jnp.asarray(0.0, dtype=jnp.float32)
                for i in range(num_probe_batches):
                    batch_attn_mask = (
                        None if local_probe_attn_mask is None else local_probe_attn_mask[i]
                    )
                    grads_i = probe_grad_for_batch(
                        params,
                        local_probe_inputs[i],
                        local_probe_labels[i],
                        batch_attn_mask,
                    )
                    loss_i, acc_i = probe_loss_acc_for_batch(
                        params,
                        local_probe_inputs[i],
                        local_probe_labels[i],
                        batch_attn_mask,
                    )
                    grads_sum = grads_i if grads_sum is None else jtu.tree_map(
                        lambda acc, x: acc + x,
                        grads_sum,
                        grads_i,
                    )
                    loss_sum = loss_sum + loss_i
                    acc_sum = acc_sum + acc_i

                scale = jnp.asarray(num_probe_batches, dtype=jnp.float32)
                local_grads = jtu.tree_map(lambda g: g / scale, grads_sum)
                local_loss_before = loss_sum / scale
                local_acc_before = acc_sum / scale

            probe_grads = jax.lax.pmean(local_grads, axis_name="data")
            probe_grads = _clip_grads(probe_grads, grad_clip)
            probe_updates, _new_opt_state = tx.update(probe_grads, opt_state, params)
            probe_params_after = optax.apply_updates(params, probe_updates)

            if local_probe_inputs.ndim == 2:
                local_loss_after, local_acc_after = probe_loss_acc_for_batch(
                    probe_params_after,
                    local_probe_inputs,
                    local_probe_labels,
                    local_probe_attn_mask,
                )
            else:
                num_probe_batches = int(local_probe_inputs.shape[0])
                loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
                acc_sum = jnp.asarray(0.0, dtype=jnp.float32)
                for i in range(num_probe_batches):
                    batch_attn_mask = (
                        None if local_probe_attn_mask is None else local_probe_attn_mask[i]
                    )
                    loss_i, acc_i = probe_loss_acc_for_batch(
                        probe_params_after,
                        local_probe_inputs[i],
                        local_probe_labels[i],
                        batch_attn_mask,
                    )
                    loss_sum = loss_sum + loss_i
                    acc_sum = acc_sum + acc_i
                scale = jnp.asarray(num_probe_batches, dtype=jnp.float32)
                local_loss_after = loss_sum / scale
                local_acc_after = acc_sum / scale

            loss_before = jax.lax.pmean(local_loss_before, axis_name="data")
            acc_before = jax.lax.pmean(local_acc_before, axis_name="data")
            loss_after = jax.lax.pmean(local_loss_after, axis_name="data")
            acc_after = jax.lax.pmean(local_acc_after, axis_name="data")
            measurement_metrics = {
                "probe_loss_before": loss_before,
                "probe_loss_after": loss_after,
                "probe_loss_reduction": loss_before - loss_after,
                "probe_acc_before": acc_before,
                "probe_acc_after": acc_after,
                "probe_acc_gain": acc_after - acc_before,
            }
            return probe_grads, probe_updates, measurement_metrics

        def compute_probe_measurement(params, opt_state):
            return compute_probe_measurement_pmapped(
                params,
                opt_state,
                sharded_probe_inputs,
                sharded_probe_labels,
                sharded_probe_attn_mask,
            )

        return compute_probe_measurement

    def probe_loss_acc(params):
        if probe_inputs.ndim == 2:
            return probe_loss_acc_for_batch(
                params,
                probe_inputs,
                probe_labels,
                probe_attn_mask,
            )

        num_probe_batches = int(probe_inputs.shape[0])
        loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
        acc_sum = jnp.asarray(0.0, dtype=jnp.float32)
        for i in range(num_probe_batches):
            batch_attn_mask = None if probe_attn_mask is None else probe_attn_mask[i]
            loss_i, acc_i = probe_loss_acc_for_batch(
                params,
                probe_inputs[i],
                probe_labels[i],
                batch_attn_mask,
            )
            loss_sum = loss_sum + loss_i
            acc_sum = acc_sum + acc_i
        scale = jnp.asarray(num_probe_batches, dtype=jnp.float32)
        return loss_sum / scale, acc_sum / scale

    @jax.jit
    def compute_probe_measurement(params, opt_state):
        if probe_inputs.ndim == 2:
            probe_grads = probe_grad_for_batch(
                params,
                probe_inputs,
                probe_labels,
                probe_attn_mask,
            )
        else:
            num_probe_batches = int(probe_inputs.shape[0])
            grads_sum = None
            for i in range(num_probe_batches):
                batch_attn_mask = None if probe_attn_mask is None else probe_attn_mask[i]
                grads_i = probe_grad_for_batch(
                    params,
                    probe_inputs[i],
                    probe_labels[i],
                    batch_attn_mask,
                )
                grads_sum = grads_i if grads_sum is None else jtu.tree_map(
                    lambda acc, x: acc + x,
                    grads_sum,
                    grads_i,
                )
            scale = jnp.asarray(num_probe_batches, dtype=jnp.float32)
            probe_grads = jtu.tree_map(lambda g: g / scale, grads_sum)

        probe_grads = _clip_grads(probe_grads, grad_clip)
        probe_updates, _new_opt_state = tx.update(probe_grads, opt_state, params)
        probe_params_after = optax.apply_updates(params, probe_updates)
        loss_before, acc_before = probe_loss_acc(params)
        loss_after, acc_after = probe_loss_acc(probe_params_after)
        measurement_metrics = {
            "probe_loss_before": loss_before,
            "probe_loss_after": loss_after,
            "probe_loss_reduction": loss_before - loss_after,
            "probe_acc_before": acc_before,
            "probe_acc_after": acc_after,
            "probe_acc_gain": acc_after - acc_before,
        }
        return probe_grads, probe_updates, measurement_metrics

    return compute_probe_measurement


def _make_effective_update_transform_fn(
    tx,
    params,
    opt_state,
    base_grads,
    *,
    base_mode: str,
    direction_scale: float,
    normalize_lr: bool,
    learning_rate: float,
):
    if base_mode == "zero":
        reference_grads = jtu.tree_map(jnp.zeros_like, base_grads)
    elif base_mode == "grad":
        reference_grads = base_grads
    else:
        raise ValueError(
            "effective_curvature_transform_base must be one of {'zero', 'grad'}."
        )

    reference_updates, _ = tx.update(reference_grads, opt_state, params)

    lr_scale = 1.0
    if normalize_lr:
        lr_scale = max(abs(float(learning_rate)), 1e-30)
    denom = float(direction_scale) * lr_scale

    def transform(direction):
        if base_mode == "zero":
            probe_grads = jtu.tree_map(
                lambda d: direction_scale * d,
                direction,
            )
        else:
            probe_grads = jtu.tree_map(
                lambda g, d: g + direction_scale * d,
                base_grads,
                direction,
            )

        probe_updates, _ = tx.update(probe_grads, opt_state, params)
        return jtu.tree_map(
            lambda probe, reference: (probe - reference) / denom,
            probe_updates,
            reference_updates,
        )

    return transform


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


def _stack_optional(values):
    if values[0] is None:
        return None
    return jnp.stack(values, axis=0)


def _stack_curvature_microbatches(inputs_list, labels_list, attn_mask_list):
    if len(inputs_list) == 1:
        return inputs_list[0], labels_list[0], attn_mask_list[0]
    return (
        jnp.stack(inputs_list, axis=0),
        jnp.stack(labels_list, axis=0),
        _stack_optional(attn_mask_list),
    )


def _build_curvature_probe_batch(
    cfg,
    loader,
    *,
    seq_len: int,
    use_doc_mask: bool,
    grouped_batches: int,
):
    num_probe_batches = getattr(cfg, "eigen_tracking_curvature_batches", None)
    if num_probe_batches is None:
        num_probe_batches = 1
    num_probe_batches = int(num_probe_batches)
    if num_probe_batches <= 0:
        raise ValueError("eigen_tracking_curvature_batches must be >= 1.")

    inputs_list = []
    labels_list = []
    attn_mask_list = []
    probe_iter = iter(loader)
    for _ in range(num_probe_batches):
        curv_raw, probe_iter = _next_batch(
            probe_iter,
            loader,
            num_batches=grouped_batches,
        )
        curv_inputs, curv_labels, curv_attn_mask = _prepare_batch(
            curv_raw,
            seq_len,
            use_doc_mask,
        )
        inputs_list.append(curv_inputs)
        labels_list.append(curv_labels)
        attn_mask_list.append(curv_attn_mask)

    if num_probe_batches == 1:
        return inputs_list[0], labels_list[0], attn_mask_list[0]

    return (
        jnp.stack(inputs_list, axis=0),
        jnp.stack(labels_list, axis=0),
        _stack_optional(attn_mask_list),
    )


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


def _unwrap_measurement_metrics(metrics, use_pmap: bool):
    if not use_pmap or metrics is None:
        return metrics
    return {key: flax_jax_utils.unreplicate(value) for key, value in metrics.items()}


def _should_run_eigen_tracking_for_step(cfg, completed_step: int) -> bool:
    """Return whether eigen tracking should run after the completed train step."""
    should_run, _phase_offset, _phase_index, _phase_frac = _eigen_tracking_phase_info(
        cfg,
        completed_step,
    )
    return should_run


def _eigen_tracking_phase_info(cfg, completed_step: int) -> tuple[bool, float, float, float]:
    """Return tracking decision plus optional SOAP-staleness phase metadata."""
    schedule = str(getattr(cfg, "eigen_tracking_schedule", "regular")).lower()
    eigen_tracking_every = int(getattr(cfg, "eigen_tracking_every", 100))
    if eigen_tracking_every <= 0:
        raise ValueError("eigen_tracking_every must be >= 1 when tracking is enabled.")

    if schedule in {"soap_phases", "soap_phase", "soap_triplet", "soap_triplets"}:
        if str(getattr(cfg, "optim", "")).lower() != "soap":
            raise ValueError("eigen_tracking_schedule='soap_phases' requires optim='soap'.")

        precondition_frequency = int(getattr(cfg, "precondition_frequency", 0))
        if precondition_frequency <= 0:
            raise ValueError(
                "eigen_tracking_schedule='soap_phases' requires "
                "precondition_frequency >= 1."
            )

        cycle_stride = int(
            getattr(cfg, "eigen_tracking_cycle_stride", eigen_tracking_every)
        )
        if cycle_stride <= 0:
            raise ValueError("eigen_tracking_cycle_stride must be >= 1.")
        if cycle_stride % precondition_frequency != 0:
            raise ValueError(
                "For eigen_tracking_schedule='soap_phases', "
                "eigen_tracking_cycle_stride must be a multiple of "
                "precondition_frequency."
            )

        phase_offsets = getattr(cfg, "eigen_tracking_phase_offsets", None)
        if phase_offsets is None:
            phase_offsets = [0, precondition_frequency // 2, precondition_frequency - 1]
        phase_offsets = [int(offset) for offset in phase_offsets]
        for offset in phase_offsets:
            if offset < 0 or offset >= precondition_frequency:
                raise ValueError(
                    "SOAP phase offsets must be in "
                    f"[0, precondition_frequency - 1], got {offset}."
                )

        # SOAP's first update initializes the basis and returns a zero update.
        # A QR refresh then occurs after step_new % precondition_frequency == 0;
        # the first completed training step that uses that refreshed basis is +1.
        first_post_refresh_step = precondition_frequency + 2
        rel_step = completed_step - first_post_refresh_step
        if rel_step < 0:
            return False, float("nan"), float("nan"), float("nan")

        for phase_index, phase_offset in enumerate(phase_offsets):
            if rel_step >= phase_offset and (
                (rel_step - phase_offset) % cycle_stride
            ) == 0:
                phase_frac = phase_offset / max(1, precondition_frequency - 1)
                return True, float(phase_offset), float(phase_index), float(phase_frac)

        return False, float("nan"), float("nan"), float("nan")

    if schedule not in {"regular", "default", "post_refresh"}:
        raise ValueError(f"Unknown eigen_tracking_schedule: {schedule}")

    if not bool(getattr(cfg, "eigen_tracking_post_soap_refresh", False)):
        should_run = (completed_step % eigen_tracking_every) == 0
        return should_run, float("nan"), float("nan"), float("nan")

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
    ), 0.0, 0.0, 0.0


def _probe_pmap_collectives(n_devices: int) -> tuple[bool, Optional[str]]:
    """Return whether pmap collectives are usable on this host.

    Some cluster setups expose multiple GPUs but lack a usable NCCL runtime.
    In that case, pmap + psum/pmean fails at runtime. We probe once and
    gracefully fall back to single-device execution if needed.
    """
    if n_devices <= 1:
        return False, None

    try:
        print(
            "train_lm pmap collective probe | "
            f"torch_imported={'torch' in sys.modules}"
        )
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

    from data.lm_loader import get_dataloaders

    trainloader, validloader = get_dataloaders(cfg)

    # Build a deterministic fixed curvature probe for curvature-based optimizers.
    # Multiple microbatches are averaged inside the LM GGN matvec to reduce
    # one-batch curvature noise without changing training batches.
    curvature_batch = None
    try:
        curvature_batch = _build_curvature_probe_batch(
            cfg,
            trainloader,
            seq_len=cfg.seq_len,
            use_doc_mask=use_doc_mask,
            grouped_batches=grouped_batches,
        )
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

    grad_clip = getattr(cfg, "grad_clip", None)
    eigen_tracking_enabled = bool(getattr(cfg, "eigen_tracking_enabled", False))
    effective_curvature_enabled = bool(
        getattr(
            cfg,
            "effective_curvature_tracking_enabled",
            getattr(cfg, "eigen_tracking_effective_curvature", False),
        )
    )
    effective_curvature_transform_base = str(
        getattr(cfg, "effective_curvature_transform_base", "zero")
    ).lower()
    if effective_curvature_transform_base not in {"zero", "grad"}:
        raise ValueError(
            "effective_curvature_transform_base must be one of {'zero', 'grad'}."
        )
    effective_curvature_direction_scale = float(
        getattr(cfg, "effective_curvature_direction_scale", 1.0)
    )
    if effective_curvature_direction_scale <= 0.0:
        raise ValueError("effective_curvature_direction_scale must be > 0.")
    effective_curvature_normalize_lr = bool(
        getattr(cfg, "effective_curvature_normalize_lr", True)
    )
    effective_curvature_gram_ridge = float(
        getattr(cfg, "effective_curvature_gram_ridge", 1e-8)
    )
    if effective_curvature_gram_ridge < 0.0:
        raise ValueError("effective_curvature_gram_ridge must be >= 0.")
    if effective_curvature_enabled and not eigen_tracking_enabled:
        raise ValueError(
            "effective_curvature_tracking_enabled=True requires "
            "eigen_tracking_enabled=True."
        )
    eigen_tracking_state = None
    eigen_tracking_csv_path = None
    eigen_tracking_measurement_mode = str(
        getattr(cfg, "eigen_tracking_measurement_mode", "actual_update")
    ).lower()
    if eigen_tracking_measurement_mode not in {"actual_update", "probe_update"}:
        raise ValueError(
            "eigen_tracking_measurement_mode must be one of "
            "{'actual_update', 'probe_update'}."
        )
    compute_probe_measurement = None
    if eigen_tracking_enabled:
        if (
            eigen_tracking_measurement_mode == "probe_update"
            and curvature_batch is None
        ):
            raise ValueError(
                "eigen_tracking_measurement_mode='probe_update' requires a "
                "valid curvature_batch."
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
        eigen_tracking_signed_split_enabled = (
            str(eigen_tracking_backend).lower() == "hessian"
        )
        eigen_tracking_light_ortho = bool(
            getattr(cfg, "eigen_tracking_light_ortho", True)
        )
        eigen_tracking_light_ortho_every = int(
            getattr(cfg, "eigen_tracking_light_ortho_every", 4)
        )
        eigen_tracking_matvec_fn = build_lm_dynamic_curvature_matvec_fn(
            cfg,
            model_def=model,
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
        if use_pmap:
            eigen_tracking_state = flax_jax_utils.replicate(eigen_tracking_state)
        eigen_tracking_csv_path = init_eigen_tracking_csv(
            cfg,
            eigen_tracking_topk,
            extra_modes=eigen_tracking_extra_modes,
        )

        if use_pmap:

            @partial(jax.pmap, axis_name="data")
            def run_eigen_tracking(
                params,
                opt_state,
                grads,
                updates,
                step,
                tracking_state,
                tracking_curvature_batch,
            ):
                def matvec_fn(matvec_params, vec_pytree, rng_key):
                    local_mv = eigen_tracking_matvec_fn(
                        matvec_params,
                        vec_pytree,
                        rng_key,
                        tracking_curvature_batch,
                    )
                    return jax.lax.pmean(local_mv, axis_name="data")

                effective_transform_fn = None
                if effective_curvature_enabled:
                    effective_transform_fn = _make_effective_update_transform_fn(
                        tx,
                        params,
                        opt_state,
                        grads,
                        base_mode=effective_curvature_transform_base,
                        direction_scale=effective_curvature_direction_scale,
                        normalize_lr=effective_curvature_normalize_lr,
                        learning_rate=float(cfg.lr),
                    )

                return track_eigenstate(
                    params=params,
                    grads=grads,
                    updates=updates,
                    step=step,
                    eigen_state=tracking_state,
                    matvec_fn=matvec_fn,
                    num_iter=eigen_tracking_iters,
                    sort_by_abs=eigen_tracking_sort_by_abs,
                    use_light_ortho=eigen_tracking_light_ortho,
                    light_ortho_every=eigen_tracking_light_ortho_every,
                    learning_rate=float(cfg.lr),
                    signed_split_enabled=eigen_tracking_signed_split_enabled,
                    effective_transform_fn=effective_transform_fn,
                    effective_curvature_gram_ridge=effective_curvature_gram_ridge,
                )

        else:

            @jax.jit
            def run_eigen_tracking(
                params,
                opt_state,
                grads,
                updates,
                step,
                tracking_state,
                tracking_curvature_batch,
            ):
                def matvec_fn(matvec_params, vec_pytree, rng_key):
                    return eigen_tracking_matvec_fn(
                        matvec_params,
                        vec_pytree,
                        rng_key,
                        tracking_curvature_batch,
                    )

                effective_transform_fn = None
                if effective_curvature_enabled:
                    effective_transform_fn = _make_effective_update_transform_fn(
                        tx,
                        params,
                        opt_state,
                        grads,
                        base_mode=effective_curvature_transform_base,
                        direction_scale=effective_curvature_direction_scale,
                        normalize_lr=effective_curvature_normalize_lr,
                        learning_rate=float(cfg.lr),
                    )

                return track_eigenstate(
                    params=params,
                    grads=grads,
                    updates=updates,
                    step=step,
                    eigen_state=tracking_state,
                    matvec_fn=matvec_fn,
                    num_iter=eigen_tracking_iters,
                    sort_by_abs=eigen_tracking_sort_by_abs,
                    use_light_ortho=eigen_tracking_light_ortho,
                    light_ortho_every=eigen_tracking_light_ortho_every,
                    learning_rate=float(cfg.lr),
                    signed_split_enabled=eigen_tracking_signed_split_enabled,
                    effective_transform_fn=effective_transform_fn,
                    effective_curvature_gram_ridge=effective_curvature_gram_ridge,
                )

        if eigen_tracking_measurement_mode == "probe_update":
            compute_probe_measurement = _make_probe_measurement_fn(
                model,
                tx,
                curvature_batch,
                grad_clip=grad_clip,
                use_doc_mask=use_doc_mask,
                use_pmap=use_pmap,
                n_devices=n_devices,
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
        tracking_this_step = (
            eigen_tracking_enabled
            and _should_run_eigen_tracking_for_step(cfg, global_step + 1)
        )
        tracking_inputs_list = (
            []
            if tracking_this_step
            and eigen_tracking_measurement_mode == "actual_update"
            else None
        )
        tracking_labels_list = [] if tracking_inputs_list is not None else None
        tracking_attn_mask_list = [] if tracking_inputs_list is not None else None
        grads_accum = None
        loss_accum = None
        acc_accum = None

        for _ in range(grad_accum_steps):
            batch, train_iter = _next_batch(train_iter, trainloader, num_batches=grouped_batches)
            if tracking_inputs_list is not None:
                curv_inputs, curv_labels, curv_attn_mask = _prepare_batch(
                    batch,
                    cfg.seq_len,
                    use_doc_mask,
                )
                tracking_inputs_list.append(curv_inputs)
                tracking_labels_list.append(curv_labels)
                tracking_attn_mask_list.append(curv_attn_mask)

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
        params_before = state.params if eigen_tracking_enabled else None
        opt_state_before = state.opt_state if eigen_tracking_enabled else None
        state, updates = apply_grads(state, grads_accum)

        global_step += 1

        if tracking_this_step:
            measurement_metrics = None
            if eigen_tracking_measurement_mode == "probe_update":
                tracking_grads, tracking_updates, measurement_metrics = compute_probe_measurement(
                    params_before,
                    opt_state_before,
                )
                tracking_curvature_batch = (
                    _shard_curvature_batch_for_pmap(curvature_batch, n_devices)
                    if use_pmap
                    else curvature_batch
                )
            else:
                if use_pmap:
                    tracking_grads = flax_jax_utils.replicate(
                        _clip_grads(
                            _unwrap_replicated(grads_accum, True),
                            grad_clip,
                        )
                    )
                    tracking_updates = updates
                else:
                    tracking_grads = _clip_grads(grads_accum, grad_clip)
                    tracking_updates = updates
                tracking_curvature_batch = _stack_curvature_microbatches(
                    tracking_inputs_list,
                    tracking_labels_list,
                    tracking_attn_mask_list,
                )
                if use_pmap:
                    tracking_curvature_batch = _shard_curvature_batch_for_pmap(
                        tracking_curvature_batch,
                        n_devices,
                    )

            eigen_tracking_state = run_eigen_tracking(
                params_before,
                opt_state_before,
                tracking_grads,
                tracking_updates,
                state.step,
                eigen_tracking_state,
                tracking_curvature_batch,
            )
            should_track_now, phase_offset, phase_index, phase_frac = (
                _eigen_tracking_phase_info(cfg, global_step)
            )
            del should_track_now
            measurement_metrics = dict(
                _unwrap_measurement_metrics(measurement_metrics, use_pmap) or {}
            )
            measurement_metrics.update(
                {
                    "tracking_phase_offset": phase_offset,
                    "tracking_phase_index": phase_index,
                    "tracking_phase_frac": phase_frac,
                }
            )
            append_eigen_tracking_row(
                eigen_tracking_csv_path,
                _unwrap_replicated(eigen_tracking_state, use_pmap),
                measurement_metrics=measurement_metrics,
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
