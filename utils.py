import os
import math
import shutil
import yaml
import csv
from typing import Sequence
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from itertools import product
from collections import namedtuple

from absl import flags

# Try to import wandb, but keep everything working if it's not installed.
try:
    import wandb
except ImportError:
    wandb = None

FLAGS = flags.FLAGS

# These are used by cluster/slurm scripts in the original plainLM repo.
# It's harmless to have them here; on local runs they'll just be None.
flags.DEFINE_integer(
    "job_idx",
    None,
    "Index for hyperparameter sweep (0..n-1). "
    "If None, use the config as-is.",
)
flags.DEFINE_string(
    "job_cluster",
    None,
    "Optional name of the cluster for logging / bookkeeping.",
)


# ---------------------------------------------------------------------------
# Configuration loading / hyperparameter sweeps
# ---------------------------------------------------------------------------

def load_config(path: str):
    """
    Parse a YAML config file and return the corresponding config as a namedtuple.

    - If the YAML contains only scalars, we just wrap them.
    - If some values are lists and FLAGS.job_idx is not None, we interpret the
      YAML as a *sweep definition*: we take the Cartesian product over all
      list-valued entries and pick the combination with index job_idx.

      Example:
        lr: [1e-3, 1e-4]
        batch_size: [64, 128]
        model: "resnet_small"

      This defines 4 configs. job_idx=0..3 selects each one.
    """
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    # We keep the original "flat dict" style from plainLM.
    # If you want nested dicts, just avoid lists and sweeps will be skipped.
    keys = list(config_dict.keys())
    Config = namedtuple("Config", keys)

    if FLAGS.job_idx is None:
        # No sweep: use YAML as-is
        cfg = config_dict
        sweep_size = 1
    else:
        # Interpret YAML as sweep definition
        values = [
            v if isinstance(v, list) else [v]
            for v in config_dict.values()
        ]
        combinations = list(product(*values))

        sweep_size = len(combinations)
        if FLAGS.job_idx >= sweep_size:
            raise ValueError(
                f"job_idx={FLAGS.job_idx} exceeds number of "
                f"combinations={sweep_size}."
            )

        combo = combinations[FLAGS.job_idx]
        cfg = {keys[i]: combo[i] for i in range(len(keys))}

    return Config(**cfg), sweep_size


# ---------------------------------------------------------------------------
# Weights & Biases (optional)
# ---------------------------------------------------------------------------

def init_wandb(cfg):
    """
    Optionally initialize a Weights & Biases run.

    Expects (optionally) the following fields in cfg:
      - use_wandb: bool (default False)
      - wandb_project: str
      - wandb_run_name: str
      - wandb_dir: str

    If wandb is not installed or use_wandb is False, this is a no-op.
    """
    if wandb is None:
        return

    use_wandb = getattr(cfg, "use_wandb", False)
    if not use_wandb:
        return

    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.environ["WANDB_SILENT"] = "true"

    project = getattr(cfg, "wandb_project", "plainCV")
    run_name = getattr(cfg, "wandb_run_name", getattr(cfg, "exp_name", "run"))
    wandb_dir = getattr(cfg, "wandb_dir", "./wandb")

    # Optional safety check: if you really want dedup detection, implement here.
    # For now we just always start a run.
    wandb.init(
        project=project,
        name=run_name,
        dir=wandb_dir,
        config=cfg._asdict(),
    )


def log_job_info():
    """Logs basic cluster job info (if available) to stdout and wandb."""
    job_cluster = getattr(FLAGS, "job_cluster", None)
    job_idx = getattr(FLAGS, "job_idx", None)

    if job_cluster is None or job_idx is None:
        return

    msg = (
        f"JOB_CLUSTER = {job_cluster}\n"
        f"JOB_INDEX   = {job_idx}\n"
        f"JOB_ID      = {job_cluster}.{job_idx}"
    )
    print(msg)

    if wandb is not None:
        wandb.log(
            {
                "JOB_CLUSTER": job_cluster,
                "JOB_INDEX": job_idx,
                "JOB_ID": f"{job_cluster}.{job_idx}",
            }
        )


# ---------------------------------------------------------------------------
# Experiment directories / checkpointing
# ---------------------------------------------------------------------------

def get_exp_dir_path(cfg) -> str:
    """
    Build an experiment directory path from config.

    Uses:
      - cfg.out_dir (default: "./exp")
      - cfg.exp_name (default: "run_{optim}_{model}" for clarity)

    If FLAGS.job_idx is set, we create a subfolder "job_idx_X".
    """
    out_dir = getattr(cfg, "out_dir", "./exp")
    # If user left the generic name, derive a more descriptive default.
    default_name = f"run_{getattr(cfg, 'optim', 'optim')}_{getattr(cfg, 'model', 'model')}"
    exp_name = getattr(cfg, "exp_name", None) or default_name
    if exp_name == "run":
        exp_name = default_name
    exp_dir = os.path.join(out_dir, exp_name)

    if FLAGS.job_idx is not None:
        exp_dir = os.path.join(exp_dir, f"job_idx_{FLAGS.job_idx}")

    return exp_dir


def maybe_make_dir(cfg):
    """
    Create experiment directory and save the config YAML.

    This is a simplified, JAX-agnostic version of the plainLM helper:
      - If cfg.over_write is False and the directory exists, we raise.
      - Otherwise we create (and optionally overwrite) the directory.
      - We always save the resolved config to 'config.yaml' inside the exp dir.

    Optional expected fields in cfg:
      - out_dir: base directory (default "./exp")
      - exp_name: experiment name (default "run")
      - over_write: bool (default True)
    """
    exp_dir = get_exp_dir_path(cfg)
    over_write = getattr(cfg, "over_write", True)

    if os.path.exists(exp_dir):
        if not over_write:
            raise ValueError(f"Found existing exp_dir at {exp_dir}.")
        print(f"Removing existing experiment dir: {exp_dir}")
        shutil.rmtree(exp_dir)

    print(f"Creating experiment directory: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save resolved config (after sweep selection) for reproducibility.
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg._asdict(), f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Logging helpers (generic, JAX-friendly)
# ---------------------------------------------------------------------------

def log_scalar_dict(cfg, metrics: dict):
    """
    Generic metric logger: prints to console and optionally logs to wandb.

    Args:
      cfg: config namedtuple (only cfg.use_wandb and cfg.print_progress are used).
      metrics: dict of scalar_name -> scalar_value.
    """
    print_progress = getattr(cfg, "print_progress", True)

    if print_progress:
        parts = []
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.4e}")
            else:
                parts.append(f"{k}: {v}")
        print(" | ".join(parts))

    if getattr(cfg, "use_wandb", False) and wandb is not None:
        wandb.log(metrics)


def print_master(msg: str):
    """
    Print only on the master process if using multi-GPU / multi-host.

    We reuse the original DDP-style convention: if the env var RANK is unset
    or 0, we print; otherwise we stay silent.
    """
    rank = os.environ.get("RANK", None)
    if rank is None:
        # Single-process case
        print(msg)
        return

    try:
        rank_val = int(rank)
    except ValueError:
        # If RANK is some weird non-int, just print.
        print(msg)
        return

    if rank_val == 0:
        print(msg)


# ---------------------------------------------------------------------------
# Curve saving helpers (CSV + plots)
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Make a string safe to use in filenames."""
    return "".join(
        c if (c.isalnum() or c in ("-", "_")) else "_"
        for c in str(name)
    )


def save_loss_curves(
    cfg,
    optimizer_name: str,
    wall_times,
    iterations,
    train_losses,
    eval_losses,
    train_accuracies,
    eval_accuracies,
) -> None:
    """
    Save a single CSV and plots for training curves.

    CSV columns:
      - iteration
      - wall_time_sec
      - train_loss
      - eval_loss
      - train_accuracy
      - eval_accuracy

    Plots (still saved separately):
      - wallclock time vs eval loss
      - iteration vs eval loss

    All files go into the experiment directory, e.g. ./exp/run or
    ./exp/run/job_idx_X (the same place as your config + TB logs).
    """
    n = len(iterations)
    if not (
        len(wall_times) == n
        and len(train_losses) == n
        and len(eval_losses) == n
        and len(train_accuracies) == n
        and len(eval_accuracies) == n
    ):
        raise ValueError(
            "All metric sequences must have the same length "
            "(iterations, wall_times, train/eval losses, train/eval accuracies)."
        )

    # This is typically something like "./exp/run" (or with job_idx suffix)
    exp_dir = get_exp_dir_path(cfg)
    os.makedirs(exp_dir, exist_ok=True)

    opt_name_safe = _sanitize_name(optimizer_name)

    # -------- Single CSV with all metrics --------
    metrics_csv_path = os.path.join(
        exp_dir, f"{opt_name_safe}_metrics.csv"
    )
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iteration",
                "wall_time_sec",
                "train_loss",
                "eval_loss",
                "train_accuracy",
                "eval_accuracy",
            ]
        )
        for it, t, tr_l, ev_l, tr_a, ev_a in zip(
            iterations,
            wall_times,
            train_losses,
            eval_losses,
            train_accuracies,
            eval_accuracies,
        ):
            writer.writerow(
                [
                    int(it),
                    float(t),
                    float(tr_l),
                    float(ev_l),
                    float(tr_a),
                    float(ev_a),
                ]
            )

    # -------- Plots (if matplotlib is available) --------
    if plt is None:
        print(
            "matplotlib not installed; saved CSV but skipped PNG plots "
            f"for optimizer {optimizer_name}."
        )
        return

    # Time vs eval loss plot
    time_png_path = os.path.join(
        exp_dir, f"{opt_name_safe}_time_vs_eval_loss.png"
    )
    plt.figure()
    plt.plot(wall_times, eval_losses)
    plt.xlabel("Wall-clock time [s]")
    plt.ylabel("Eval loss")
    plt.title(f"{optimizer_name} – time vs eval loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(time_png_path)
    plt.close()

    # Iter vs eval loss plot
    iter_png_path = os.path.join(
        exp_dir, f"{opt_name_safe}_iter_vs_eval_loss.png"
    )
    plt.figure()
    plt.plot(iterations, eval_losses)
    plt.xlabel("Iteration (epoch)")
    plt.ylabel("Eval loss")
    plt.title(f"{optimizer_name} – iteration vs eval loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(iter_png_path)
    plt.close()

    print(f"Saved metrics CSV + eval-loss plots in {exp_dir}")
