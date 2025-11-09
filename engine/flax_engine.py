# engine/flax_engine.py
from dataclasses import dataclass
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from optim.factory import get_optimizer


def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def compute_metrics(logits, labels) -> Dict[str, jnp.ndarray]:
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {"loss": loss, "accuracy": accuracy}


class TrainState(train_state.TrainState):
    batch_stats: Any = None  # for BatchNorm (mutable collections)


def create_train_state(rng, model_def, learning_rate: float, image_shape, num_classes: int, cfg=None):
    dummy_batch = jnp.zeros(image_shape, dtype=jnp.float32)
    variables = model_def.init(rng, dummy_batch, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats")

    if cfg is None:
        # Fallback to simple AdamW if no config passed
        tx = optax.adamw(learning_rate)
    else:
        tx = get_optimizer(cfg)

    state = TrainState.create(
        apply_fn=model_def.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )
    return state


def _apply_model(state: TrainState, images, labels, rng, train: bool):
    """Forward + loss + metrics, handling mutable batch_stats."""
    variables = {"params": state.params}
    if state.batch_stats is not None:
        variables["batch_stats"] = state.batch_stats

    if train and state.batch_stats is not None:
        (logits, new_vars) = state.apply_fn(
            variables, images, train=True, mutable=["batch_stats"], rngs={"dropout": rng}
        )
        new_batch_stats = new_vars["batch_stats"]
    else:
        logits = state.apply_fn(variables, images, train=False, mutable=False)
        new_batch_stats = state.batch_stats

    loss = cross_entropy_loss(logits, labels)
    metrics = compute_metrics(logits, labels)
    return loss, (metrics, new_batch_stats)


def make_train_step():
    @jax.jit
    def train_step(state: TrainState, batch, rng):
        images, labels = batch

        grad_fn = jax.value_and_grad(lambda params: _apply_model(
            state.replace(params=params), images, labels, rng, train=True
        )[0])

        loss, grads = grad_fn(state.params)

        # Recompute to get metrics + updated batch_stats
        loss2, (metrics, new_batch_stats) = _apply_model(
            state, images, labels, rng, train=True
        )

        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=new_batch_stats)
        metrics = {"loss": loss2, **metrics}
        return state, metrics

    return train_step


def make_eval_step():
    @jax.jit
    def eval_step(state: TrainState, batch):
        images, labels = batch
        loss, (metrics, _) = _apply_model(state, images, labels, rng=None, train=False)
        metrics = {"loss": loss, **metrics}
        return metrics

    return eval_step
