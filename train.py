# train.py
"""Train a small CNN or MLP on Fashion-MNIST in JAX/Flax."""
import os

from optim.pns_eigenadam import profile_pns_eigenadam_curvature
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
from absl import app, flags

import jax
import jax.numpy as jnp
import sys
from jax import tree_util as jtu

print("DEBUG sys.executable:", sys.executable)
print("DEBUG jax version:", jax.__version__)
import jaxlib
print("DEBUG jaxlib version:", jaxlib.__version__)
print("DEBUG devices from train.py:", jax.devices())
print("DEBUG backend from train.py:", jax.default_backend())
from absl import app, flags


from flax.metrics.tensorboard import SummaryWriter

from data import get_datasets
from engine.flax_engine import create_train_state, make_train_step, make_eval_step
from optim.ggn_utils import make_ggn_matvec_fn

from models.mlp import MLP
from models.resnet import SmallResNet, ResNet30, ResNet18
from models.vit_small import VisionTransformer
from utils import (
    load_config,
    maybe_make_dir,
    init_wandb,
    log_scalar_dict,
    get_exp_dir_path,
    save_loss_curves,
    _sanitize_name,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "config/config.yaml", "Path to config.yaml file.")


def construct_model(cfg):
    """Build model from config."""
    if cfg.model == "mlp":
        return MLP(num_classes=cfg.num_classes)
    elif cfg.model == "resnet_small":
        return SmallResNet(
            num_classes=cfg.num_classes,
            use_bn=getattr(cfg, "resnet_use_batchnorm", True),
        )
    elif cfg.model == "resnet30":
        return ResNet30(
            num_classes=cfg.num_classes,
            use_bn=getattr(cfg, "resnet_use_batchnorm", True),
        )
    elif cfg.model == "resnet18":
        return ResNet18(
            num_classes=cfg.num_classes,
            use_bn=getattr(cfg, "resnet_use_batchnorm", True),
        )
    elif cfg.model in {"vit", "vit_small", "vision_transformer"}:
        return VisionTransformer(
            num_classes=cfg.num_classes,
            patch_size=getattr(cfg, "vit_patch_size", 4),
            hidden_size=getattr(cfg, "vit_hidden_size", 128),
            mlp_dim=getattr(cfg, "vit_mlp_dim", 256),
            num_layers=getattr(cfg, "vit_layers", 4),
            num_heads=getattr(cfg, "vit_heads", 4),
            dropout_rate=getattr(cfg, "vit_dropout", 0.1),
            use_layernorm=getattr(cfg, "vit_use_layernorm", True),
            use_batchnorm=getattr(cfg, "vit_use_batchnorm", False),
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

    # --- Optional: curvature spectrum logging ---
    log_curv = getattr(cfg, "pns_log_curvature", False)
    use_pns = cfg.optim in {"pns_eigenadam", "pns-eigenadam"}
    use_muon = cfg.optim in {"pns_eigenmuon", "pns-eigenmuon"}
    curvature_csv_path = None
    max_eigs = getattr(cfg, "pns_max_eigenvectors", 16)
    max_neg_eigs = getattr(cfg, "pns_negcurv_iters", 0) or 0 
    muon_max_eigs = getattr(cfg, "gradient_eigenvectors", max_eigs)

    if log_curv and use_pns:
        curvature_csv_path = os.path.join(exp_dir, "curvature.csv")
        header = (
            ["epoch", "global_step"]
            + [f"eig_{i}" for i in range(max_eigs)]
            + ["rotation_diff_pos"]
            + [f"eig_neg_{i}" for i in range(max_neg_eigs)]
            + ["rotation_diff_neg"]
        )
        with open(curvature_csv_path, "w") as f:
            f.write(",".join(header) + "\n")


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
        dataset=cfg.dataset,
        batch_size=cfg.batch_size,
        seed=cfg.seed,  # fixed seed for reproducibility
        image_size=getattr(cfg, "image_size", None),
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

    muon_log_files = {}
    if log_curv and use_muon:
        muon_curv_dir = os.path.join(exp_dir, "gradient_eigenvalues")
        os.makedirs(muon_curv_dir, exist_ok=True)
        muon_state = state.opt_state
        eig_leaves = [
            leaf for leaf in jtu.tree_leaves(muon_state.eigenvalues)
            if isinstance(leaf, jax.Array)
        ]
        if eig_leaves:
            muon_max_eigs = int(eig_leaves[0].shape[0])
        header = ["epoch", "global_step"] + [f"eig_{i}" for i in range(muon_max_eigs)]

        def register_layer(path, leaf):
            if hasattr(leaf, "ndim") and leaf.ndim == 2:
                layer_name = "/".join(str(k) for k in path)
                file_name = f"{_sanitize_name(layer_name)}.csv"
                file_path = os.path.join(muon_curv_dir, file_name)
                muon_log_files[layer_name] = file_path
                with open(file_path, "w") as f:
                    f.write(",".join(header) + "\n")
            return None

        jtu.tree_map_with_path(register_layer, state.params)

    # --------- OPTIONAL: profile PN-S curvature step once ---------
    # Only makes sense if you're using a curvature-based optimizer
    # (pns_eigenadam, pns_eigenmuon, curvature_muon, hf, etc.)
    #if cfg.optim in {"pns_eigenadam", "pns-eigenadam",
    #                 "pns_eigenmuon", "pns-eigenmuon",
    #                 "curvature_muon", "curvature-muon"}:
    #    # Build the same GGN matvec that the optimizer factory uses
    #    ggn_mv = make_ggn_matvec_fn(
    #        model_def=model_def,
    #        curvature_batch=curvature_batch,
    #        batch_stats=state.batch_stats,
    #    )
#
    #    # Use config knobs if they exist, else fall back to defaults
    #    max_eigs = getattr(cfg, "pns_max_eigenvectors", 16)
    #    lanczos_iters = getattr(cfg, "pns_lanczos_iters", None)
#
    #    # Use the main rng or split a fresh one
    #    rng, prof_rng = jax.random.split(rng)
#
    #    profile_pns_eigenadam_curvature(
    #        params=state.params,
    #        ggn_matvec_fn=ggn_mv,
    #        max_eigenvectors=max_eigs,
    #        lanczos_iters=lanczos_iters,
    #        rng=prof_rng,
    #        warmup=True,
    #    )
    ## --------------------------------------------------------------

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
            dataset=cfg.dataset,
            batch_size=cfg.batch_size,
            seed=cfg.seed + epoch,  # optional: vary seed per epoch
            image_size=getattr(cfg, "image_size", None),
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

        if log_curv and use_pns and curvature_csv_path is not None:
            pns_state = state.opt_state 

            # Positive spectrum
            ev_pos = jnp.asarray(pns_state.eigenvalues)
            rot_pos = float(pns_state.rotation_diff)
            step_opt = int(pns_state.step)
            eig_pos_list = [float(x) for x in ev_pos[:max_eigs]]

            # Negative spectrum (if present)
            if max_neg_eigs > 0:
                ev_neg = jnp.asarray(pns_state.neg_eigenvalues)
                rot_neg = float(pns_state.neg_rotation_diff)
                eig_neg_list = [float(x) for x in ev_neg[:max_neg_eigs]]
            else:
                rot_neg = 0.0
                eig_neg_list = []

            row = [epoch, step_opt] + eig_pos_list + [rot_pos] + eig_neg_list + [rot_neg]

            with open(curvature_csv_path, "a") as f:
                f.write(",".join(str(x) for x in row) + "\n")

        if log_curv and use_muon and muon_log_files:
            muon_state = state.opt_state
            step_opt = int(muon_state.step)

            def write_muon_row(path, eigs):
                if eigs is None:
                    return None
                layer_name = "/".join(str(k) for k in path)
                file_path = muon_log_files.get(layer_name)
                if file_path is None:
                    return None
                eig_vals = [float(x) for x in jnp.asarray(eigs)[:muon_max_eigs]]
                row = [epoch, step_opt] + eig_vals
                with open(file_path, "a") as f:
                    f.write(",".join(str(x) for x in row) + "\n")
                return None

            jtu.tree_map_with_path(write_muon_row, muon_state.eigenvalues)


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
