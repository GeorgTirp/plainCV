# train.py
"""Train a small CNN or MLP on Fashion-MNIST in JAX/Flax."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
from absl import app, flags

import jax
import jax.numpy as jnp
import sys

print("DEBUG sys.executable:", sys.executable)
print("DEBUG jax version:", jax.__version__)
import jaxlib
print("DEBUG jaxlib version:", jaxlib.__version__)
print("DEBUG devices from train.py:", jax.devices())
print("DEBUG backend from train.py:", jax.default_backend())
from absl import app, flags


from flax.metrics.tensorboard import SummaryWriter

from data.fashion_mnist import get_datasets
from engine.flax_engine import create_train_state, make_train_step, make_eval_step
from models.mlp import MLP
from models.resnet_small import SmallResNet
from models.vit_small import VisionTransformer
from utils import (
    load_config,
    maybe_make_dir,
    init_wandb,
    log_scalar_dict,
    get_exp_dir_path,
    save_loss_curves,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "config/config.yaml", "Path to config.yaml file.")


def construct_model(cfg):
    """Build model from config."""
    if cfg.model == "mlp":
        return MLP(num_classes=cfg.num_classes)
    elif cfg.model == "resnet_small":
        return SmallResNet(num_classes=cfg.num_classes)
    elif cfg.model in {"vit", "vit_small", "vision_transformer"}:
        return VisionTransformer(
            num_classes=cfg.num_classes,
            patch_size=getattr(cfg, "vit_patch_size", 4),
            hidden_size=getattr(cfg, "vit_hidden_size", 128),
            mlp_dim=getattr(cfg, "vit_mlp_dim", 256),
            num_layers=getattr(cfg, "vit_layers", 4),
            num_heads=getattr(cfg, "vit_heads", 4),
            dropout_rate=getattr(cfg, "vit_dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model}")


def main(_argv):
    # Parse config
    cfg, _ = load_config(FLAGS.config)

    # Experiment dir + logging
    maybe_make_dir(cfg)
    init_wandb(cfg)
    exp_dir = get_exp_dir_path(cfg)
    writer = SummaryWriter(log_dir=exp_dir)

    # RNG setup
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)

    # Model
    model_def = construct_model(cfg)

    # Example image shape (B, H, W, C)
    image_shape = (
        cfg.batch_size,
        cfg.image_size,
        cfg.image_size,
        cfg.num_channels,
    )

    # Curvature batch for GGN / PN-EigenAdam (one deterministic batch)
    curv_train_ds, _ = get_datasets(
        batch_size=cfg.batch_size,
        seed=cfg.seed,  # fixed seed for reproducibility
    )
    curv_images, curv_labels = next(iter(curv_train_ds))
    curvature_batch = (curv_images, curv_labels)

    # Create train state (params + optimizer)
    state = create_train_state(
        rng=init_rng,
        model_def=model_def,
        learning_rate=cfg.lr,
        image_shape=image_shape,
        num_classes=cfg.num_classes,
        cfg=cfg,
        curvature_batch=curvature_batch,
    )

    train_step = make_train_step()
    eval_step = make_eval_step()

    # For curves: wall-clock vs loss and iteration vs loss
    wall_times = []
    iters = []
    train_losses = []
    eval_losses = []
    train_accs = []
    eval_accs = []

    training_start_time = time.time()

    # --------------------
    # Training loop
    # --------------------
    for epoch in range(1, cfg.num_epochs + 1):
        start_time = time.time()

        # Fresh dataset iterators each epoch
        train_ds, test_ds = get_datasets(
            batch_size=cfg.batch_size,
            seed=cfg.seed + epoch,  # optional: vary seed per epoch
        )

        # ---- Train ----
        train_metrics = []
        for batch in train_ds:
            rng, batch_rng = jax.random.split(rng)
            state, metrics = train_step(state, batch, batch_rng)
            train_metrics.append(metrics)

        # ---- Eval ----
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

        # Accumulate data for curves
        # inside epoch loop, after computing train_summary / eval_summary:
        elapsed_from_start = time.time() - training_start_time
        wall_times.append(float(elapsed_from_start))
        iters.append(int(epoch))
        train_losses.append(float(train_summary["loss"]))
        eval_losses.append(float(eval_summary["loss"]))
        train_accs.append(float(train_summary["accuracy"]))
        eval_accs.append(float(eval_summary["accuracy"]))

        # Console + optional wandb logging
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

        # TensorBoard logging
        writer.scalar("train/loss", float(train_summary["loss"]), step=epoch)
        writer.scalar("train/accuracy", float(train_summary["accuracy"]), step=epoch)
        writer.scalar("eval/loss", float(eval_summary["loss"]), step=epoch)
        writer.scalar("eval/accuracy", float(eval_summary["accuracy"]), step=epoch)
        writer.scalar("epoch_time", float(epoch_time), step=epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_summary['loss']:.4f}, "
            f"train acc {train_summary['accuracy']:.4f} | "
            f"eval loss {eval_summary['loss']:.4f}, "
            f"eval acc {eval_summary['accuracy']:.4f} | "
            f"time {epoch_time:.2f}s"
        )

    # Save CSVs + plots for wall-clock vs loss and iteration vs loss
    save_loss_curves(
        cfg=cfg,
        optimizer_name=cfg.optim,
        wall_times=wall_times,
        iterations=iters,
        train_losses=train_losses,
        eval_losses=eval_losses,
        train_accuracies=train_accs,
        eval_accuracies=eval_accs,
    )

    writer.flush()
    writer.close()
    
    

if __name__ == "__main__":
    app.run(main)
