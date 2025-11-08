# train.py
"""Train a small CNN or MLP on Fashion-MNIST in JAX/Flax."""
import time
from absl import app, flags

import jax
import jax.numpy as jnp

from data.fashion_mnist import get_datasets
from engine.flax_engine import create_train_state, make_train_step, make_eval_step
from models.mlp import MLP
from models.resnet_small import SmallResNet
from utils import load_config, maybe_make_dir, init_wandb, log_scalar_dict

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "config/config.yaml", "Path to config.yaml file.")


def construct_model(cfg):
    # cfg.model is a string in a flat config: "mlp" or "resnet_small"
    if cfg.model == "mlp":
        return MLP(num_classes=cfg.num_classes)
    elif cfg.model == "resnet_small":
        return SmallResNet(num_classes=cfg.num_classes)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")


def main(_argv):
    # Now FLAGS.config is parsed and safe to use
    cfg, _ = load_config(FLAGS.config)

    maybe_make_dir(cfg)
    init_wandb(cfg)

    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng, step_rng = jax.random.split(rng, 3)

    # Data
    train_ds, test_ds = get_datasets(
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    # Model
    model_def = construct_model(cfg)

    # Example image shape (B, H, W, C)
    image_shape = (
        cfg.batch_size,
        cfg.image_size,
        cfg.image_size,
        cfg.num_channels,
    )

    state = create_train_state(
        init_rng,
        model_def=model_def,
        learning_rate=cfg.lr,
        image_shape=image_shape,
        num_classes=cfg.num_classes,
    )

    train_step = make_train_step()
    eval_step = make_eval_step()

    # Training loop
    for epoch in range(1, cfg.num_epochs + 1):
        start_time = time.time()

        # Train
        train_metrics = []
        for batch in train_ds:
            rng, batch_rng = jax.random.split(step_rng)
            state, metrics = train_step(state, batch, batch_rng)
            train_metrics.append(metrics)

        # Eval
        eval_metrics = []
        for batch in test_ds:
            metrics = eval_step(state, batch)
            eval_metrics.append(metrics)

        def stack_metrics(metrics_list):
            return {
                k: jnp.mean(jnp.array([m[k] for m in metrics_list]))
                for k in metrics_list[0]
            }

        train_summary = stack_metrics(train_metrics)
        eval_summary = stack_metrics(eval_metrics)

        epoch_time = time.time() - start_time

        log_scalar_dict(
            cfg,
            {
                "epoch": epoch,
                "train/loss": float(train_summary["loss"]),
                "train/accuracy": float(train_summary["accuracy"]),
                "eval/loss": float(eval_summary["loss"]),
                "eval/accuracy": float(eval_summary["accuracy"]),
                "epoch_time": epoch_time,
            },
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_summary['loss']:.4f}, "
            f"train acc {train_summary['accuracy']:.4f} | "
            f"eval loss {eval_summary['loss']:.4f}, "
            f"eval acc {eval_summary['accuracy']:.4f} | "
            f"time {epoch_time:.2f}s"
        )


if __name__ == "__main__":
    app.run(main)
