import os
import math
import shutil
import re
import yaml
import csv
from datetime import datetime
from typing import Optional, Sequence
import jax
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from itertools import product

from absl import flags

# Try to import wandb, but keep everything working if it's not installed.
try:
    import wandb
except ImportError:
    wandb = None

try:
    import fcntl
except ImportError:
    fcntl = None

FLAGS = flags.FLAGS

_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


class Config(dict):
    """Mutable config with attribute access and namedtuple-like helpers."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def _asdict(self):
        return dict(self)

    def to_dict(self):
        return dict(self)

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


def _coerce_yaml_scalar(value):
    """Convert numeric-like YAML strings (e.g. '1e-3') into numbers."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    lowered = stripped.lower()

    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null", "~"}:
        return None
    if _NUMERIC_RE.fullmatch(stripped):
        if any(ch in lowered for ch in (".", "e")):
            return float(stripped)
        return int(stripped)
    return value


def _coerce_yaml_values(value):
    """Recursively coerce YAML-loaded values to more useful Python scalars."""
    if isinstance(value, dict):
        return {k: _coerce_yaml_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_yaml_values(v) for v in value]
    return _coerce_yaml_scalar(value)

def load_config(path: str):
    """
    Parse a YAML config file and return the corresponding config as a mutable mapping.

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
        config_dict = _coerce_yaml_values(yaml.safe_load(f))

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
        keys = list(config_dict.keys())
        cfg = {keys[i]: combo[i] for i in range(len(keys))}

    return Config(cfg), sweep_size


# ---------------------------------------------------------------------------
# Weights & Biases (optional)
# ---------------------------------------------------------------------------

def _wandb_cfg_to_dict(wb_cfg):
    if hasattr(wb_cfg, "as_dict"):
        return wb_cfg.as_dict()
    return dict(wb_cfg)


def _next_wandb_run_index(counter_path: str) -> int:
    """Allocate a monotonically increasing run index persisted on disk."""
    counter_dir = os.path.dirname(counter_path)
    if counter_dir:
        os.makedirs(counter_dir, exist_ok=True)

    with open(counter_path, "a+", encoding="utf-8") as counter_file:
        if fcntl is not None:
            fcntl.flock(counter_file.fileno(), fcntl.LOCK_EX)

        try:
            counter_file.seek(0)
            raw = counter_file.read().strip()
            try:
                current = int(raw)
            except (TypeError, ValueError):
                current = 0
            next_index = current + 1

            counter_file.seek(0)
            counter_file.truncate()
            counter_file.write(str(next_index))
            counter_file.flush()
            try:
                os.fsync(counter_file.fileno())
            except OSError:
                pass
        finally:
            if fcntl is not None:
                fcntl.flock(counter_file.fileno(), fcntl.LOCK_UN)

    return next_index


def init_wandb(cfg):
    """
    Optionally initialize a Weights & Biases run.

    Expects (optionally) the following fields in cfg:
      - use_wandb: bool (default False)
      - wandb_project: str
      - wandb_run_name: str
      - wandb_dir: str

    If wandb is not installed and no sweep is active, this is a no-op.
    """
    if wandb is None:
        return

    use_wandb = getattr(cfg, "use_wandb", False)
    sweep_id = os.environ.get("WANDB_SWEEP_ID")
    sweep_active = sweep_id is not None
    if not use_wandb and not sweep_active:
        return

    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.environ["WANDB_SILENT"] = "true"

    # During sweeps, prefer agent-provided routing metadata.
    project = os.environ.get("WANDB_PROJECT", getattr(cfg, "wandb_project", "plainCV"))
    entity = os.environ.get("WANDB_ENTITY", getattr(cfg, "wandb_entity", None))
    run_name_base = getattr(cfg, "wandb_run_name", getattr(cfg, "exp_name", "run"))
    wandb_dir = os.path.abspath(getattr(cfg, "wandb_dir", "./wandb"))
    exp_dir = os.path.abspath(get_exp_dir_path(cfg))
    local_config_path = os.path.join(exp_dir, "config.yaml")

    run_name = run_name_base
    run_index = None
    if run_name and not sweep_active and bool(getattr(cfg, "wandb_unique_names", True)):
        run_index = _next_wandb_run_index(os.path.join(wandb_dir, ".run_counter"))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{run_index:05d}_{timestamp}_{run_name_base}"

    cfg["local_exp_dir"] = exp_dir
    cfg["local_config_path"] = local_config_path
    cfg["wandb_run_name_base"] = run_name_base
    if run_index is not None:
        cfg["wandb_run_index"] = run_index
    cfg["wandb_run_name_resolved"] = run_name

    if wandb.run is None:
        init_kwargs = {
            "project": project,
            "dir": wandb_dir,
            "config": cfg._asdict(),
        }
        if entity:
            init_kwargs["entity"] = entity
        # Let W&B agent name sweep runs unless this is a regular run.
        if run_name and not sweep_active:
            init_kwargs["name"] = run_name

        try:
            run = wandb.init(**init_kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if sweep_active and "sweep" in msg and "not found" in msg:
                raise RuntimeError(
                    f"W&B sweep '{sweep_id}' not found for project='{project}'"
                    + (f", entity='{entity}'" if entity else "")
                    + ". Ensure the agent was started with the full path "
                      "'entity/project/sweep_id' and that the sweep still exists."
                ) from exc
            raise
    else:
        run = wandb.run

    if run is not None:
        run.summary["local_exp_dir"] = exp_dir
        run.summary["local_config_path"] = local_config_path
        if run_index is not None:
            run.summary["wandb_run_index"] = run_index

    for key, value in _wandb_cfg_to_dict(run.config).items():
        if key.startswith("_"):
            continue
        cfg[key] = value

    return run


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
    flag_exp_name = getattr(FLAGS, "exp_name", None)
    exp_name = flag_exp_name or getattr(cfg, "exp_name", None) or default_name
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

    if wandb is not None and (getattr(cfg, "use_wandb", False) or wandb.run is not None):
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


def _nu_regression(
    log_alpha: np.ndarray,
    log_lambda: np.ndarray,
    weights: Optional[np.ndarray] = None,
    spread_threshold: float = 0.1,
) -> tuple[float, float]:
    """OLS regression of log|alpha| on log|lambda|, returning (nu_hat, r2).

    Fits the model  log|α_i| = c − ν log|λ_i|  and returns the slope
    ν̂ = −Cov(log|λ|, log|α|) / Var(log|λ|)  plus R².

    Returns (nan, nan) when fewer than 2 finite points are available or when
    Var(log|λ|) < spread_threshold (eigenvalues too tightly clustered to
    recover a reliable scaling exponent).  This avoids the per-direction
    division-by-small-log instability of the ratio formula.
    """
    finite = np.isfinite(log_alpha) & np.isfinite(log_lambda)
    if finite.sum() < 2:
        return float("nan"), float("nan")
    x = log_lambda[finite]
    y = log_alpha[finite]
    if weights is not None:
        w = weights[finite]
        s = float(w.sum())
        if s < 1e-30:
            return float("nan"), float("nan")
        w = w / s
        mx = float(np.dot(w, x))
        my = float(np.dot(w, y))
        dx = x - mx
        var_x = float(np.dot(w, dx * dx))
        cov_xy = float(np.dot(w, dx * (y - my)))
        ss_tot = float(np.dot(w, (y - my) ** 2))
    else:
        mx = float(np.mean(x))
        my = float(np.mean(y))
        dx = x - mx
        n = len(x)
        var_x = float(np.dot(dx, dx)) / n
        cov_xy = float(np.dot(dx, y - my)) / n
        ss_tot = float(np.dot(y - my, y - my))
    if var_x < spread_threshold:
        return float("nan"), float("nan")
    nu_hat = float(-cov_xy / var_x)
    resid = y - (my - nu_hat * dx)
    if weights is not None:
        ss_res = float(np.dot(w, resid * resid))
    else:
        ss_res = float(np.dot(resid, resid))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else float("nan")
    return nu_hat, r2


def init_eigen_tracking_csv(
    cfg,
    top_k: int,
    extra_modes: int = 0,
    filename: str = "eigen_tracking.csv",
    measurement_momentum: bool = False,
) -> str:
    exp_dir = get_exp_dir_path(cfg)
    os.makedirs(exp_dir, exist_ok=True)
    csv_path = os.path.join(exp_dir, filename)

    header = (
        [
            "global_step",
            "tracking_phase_offset",
            "tracking_phase_index",
            "tracking_phase_frac",
            "rotation_diff",
            "eff_cond",
            "effective_curvature_cond",
            "actual_update_rayleigh",
            "topk_eos_rho_max",
            "topk_eos_rho_mean",
            "topk_eos_rho_update_weighted",
            "topk_eos_rho_over_2_max",
            "topk_eos_rho_over_2_update_weighted",
            "probe_loss_before",
            "probe_loss_after",
            "probe_loss_reduction",
            "probe_acc_before",
            "probe_acc_after",
            "probe_acc_gain",
            "grad_norm",
            "update_norm",
            "tracked_update_energy_frac",
            "tracked_grad_energy_frac",
            "tracked_update_grad_cosine",
            "full_update_grad_cosine",
            "full_update_grad_cos2",
            "pos_update_energy_frac",
            "neg_update_energy_frac",
            "pos_grad_energy_frac",
            "neg_grad_energy_frac",
            "pos_update_grad_cosine",
            "neg_update_grad_cosine",
            # Regression-based nu (OLS slope of log|alpha| on log|lambda|)
            "nu_reg",
            "nu_reg_r2",
            "nu_reg_var_log_lambda",
            "nu_reg_ew",
            "nu_reg_ew_r2",
            "nu_reg_dao",
            "nu_reg_dao_r2",
            "nu_reg_dao_ew",
            "nu_reg_dao_ew_r2",
        ]
        # Momentum-reference columns (only when measurement_momentum=True)
        + (
            [
                # Scale-dependent nu (momentum reference, analogous to nu_reg)
                "nu_reg_mom",
                "nu_reg_mom_r2",
                "nu_reg_mom_ew",
                "nu_reg_mom_ew_r2",
                # Direction-alignment-only nu (momentum reference, analogous to nu_reg_dao)
                "nu_reg_dao_mom",
                "nu_reg_dao_mom_r2",
                "nu_reg_dao_mom_ew",
                "nu_reg_dao_mom_ew_r2",
                "m_norm",
            ]
            if measurement_momentum
            else []
        )
        + [f"eig_{i}" for i in range(top_k)]
        + [f"extra_eig_{i}" for i in range(extra_modes)]
        + [f"ritz_resid_{i}" for i in range(top_k)]
        + [f"extra_ritz_resid_{i}" for i in range(extra_modes)]
        + [f"g_proj_{i}" for i in range(top_k)]
        + [f"extra_g_proj_{i}" for i in range(extra_modes)]
        + [f"d_proj_{i}" for i in range(top_k)]
        + [f"extra_d_proj_{i}" for i in range(extra_modes)]
        + ([f"m_proj_{i}" for i in range(top_k)] if measurement_momentum else [])
        + ([f"extra_m_proj_{i}" for i in range(extra_modes)] if measurement_momentum else [])
        + [f"update_energy_frac_{i}" for i in range(top_k)]
        + [f"extra_update_energy_frac_{i}" for i in range(extra_modes)]
        + [f"alpha_valid_{i}" for i in range(top_k)]
        + [f"extra_alpha_valid_{i}" for i in range(extra_modes)]
        + [f"alpha_{i}" for i in range(top_k)]
        + [f"extra_alpha_{i}" for i in range(extra_modes)]
        + [f"phi_{i}" for i in range(top_k)]
        + [f"extra_phi_{i}" for i in range(extra_modes)]
        + [f"eos_rho_{i}" for i in range(top_k)]
        + [f"extra_eos_rho_{i}" for i in range(extra_modes)]
        + [f"eff_curv_eig_{i}" for i in range(top_k)]
        + [f"extra_eff_curv_eig_{i}" for i in range(extra_modes)]
        + [f"damped_eff_curv_eig_{i}" for i in range(top_k)]
        + [f"extra_damped_eff_curv_eig_{i}" for i in range(extra_modes)]
        + [f"precond_dir_gain_{i}" for i in range(top_k)]
        + [f"extra_precond_dir_gain_{i}" for i in range(extra_modes)]
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    return csv_path


def append_eigen_tracking_row(
    csv_path: str,
    tracking_state,
    measurement_metrics=None,
    measurement_momentum: bool = False,
) -> None:
    measurement_metrics = measurement_metrics or {}

    def _metric(name: str) -> float:
        value = measurement_metrics.get(name, float("nan"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    # Avoid device_get on the whole state: it contains full eigenvector matrices
    # that can be hundreds of GiB for LM-scale models.
    scalar_step, scalar_rotation, scalar_eff_cond = jax.device_get(
        (
            tracking_state.step,
            tracking_state.rotation_diff,
            tracking_state.eff_cond,
        )
    )
    (
        eigenvalues,
        extra_eigenvalues,
        ritz_residual,
        extra_ritz_residual,
        g_proj,
        extra_g_proj,
        d_proj,
        extra_d_proj,
        update_energy_frac,
        extra_update_energy_frac,
        alpha_valid,
        extra_alpha_valid,
        alpha,
        extra_alpha,
        phi,
        extra_phi,
        eos_rho,
        extra_eos_rho,
        grad_norm,
        update_norm,
        tracked_update_energy_frac,
        tracked_grad_energy_frac,
        tracked_update_grad_cosine,
        full_update_grad_cosine,
        full_update_grad_cos2,
        pos_update_energy_frac,
        neg_update_energy_frac,
        pos_grad_energy_frac,
        neg_grad_energy_frac,
        pos_update_grad_cosine,
        neg_update_grad_cosine,
        effective_curvature_cond,
        actual_update_rayleigh,
        topk_eos_rho_max,
        topk_eos_rho_mean,
        topk_eos_rho_update_weighted,
        topk_eos_rho_over_2_max,
        topk_eos_rho_over_2_update_weighted,
        effective_curvature_eigenvalues,
        extra_effective_curvature_eigenvalues,
        damped_effective_curvature_eigenvalues,
        extra_damped_effective_curvature_eigenvalues,
        preconditioned_dir_gain,
        extra_preconditioned_dir_gain,
        m_proj,
        extra_m_proj,
        m_norm,
    ) = jax.device_get(
        (
            tracking_state.eigenvalues,
            getattr(tracking_state, "extra_eigenvalues", ()),
            getattr(tracking_state, "ritz_residual", ()),
            getattr(tracking_state, "extra_ritz_residual", ()),
            getattr(tracking_state, "g_proj", ()),
            getattr(tracking_state, "extra_g_proj", ()),
            getattr(tracking_state, "d_proj", ()),
            getattr(tracking_state, "extra_d_proj", ()),
            getattr(tracking_state, "update_energy_frac", ()),
            getattr(tracking_state, "extra_update_energy_frac", ()),
            getattr(tracking_state, "alpha_valid", ()),
            getattr(tracking_state, "extra_alpha_valid", ()),
            tracking_state.alpha,
            getattr(tracking_state, "extra_alpha", ()),
            getattr(tracking_state, "phi", ()),
            getattr(tracking_state, "extra_phi", ()),
            getattr(tracking_state, "eos_rho", ()),
            getattr(tracking_state, "extra_eos_rho", ()),
            getattr(tracking_state, "grad_norm", float("nan")),
            getattr(tracking_state, "update_norm", float("nan")),
            getattr(tracking_state, "tracked_update_energy_frac", float("nan")),
            getattr(tracking_state, "tracked_grad_energy_frac", float("nan")),
            getattr(tracking_state, "tracked_update_grad_cosine", float("nan")),
            getattr(tracking_state, "full_update_grad_cosine", float("nan")),
            getattr(tracking_state, "full_update_grad_cos2", float("nan")),
            getattr(tracking_state, "pos_update_energy_frac", float("nan")),
            getattr(tracking_state, "neg_update_energy_frac", float("nan")),
            getattr(tracking_state, "pos_grad_energy_frac", float("nan")),
            getattr(tracking_state, "neg_grad_energy_frac", float("nan")),
            getattr(tracking_state, "pos_update_grad_cosine", float("nan")),
            getattr(tracking_state, "neg_update_grad_cosine", float("nan")),
            getattr(tracking_state, "effective_curvature_cond", float("nan")),
            getattr(tracking_state, "actual_update_rayleigh", float("nan")),
            getattr(tracking_state, "topk_eos_rho_max", float("nan")),
            getattr(tracking_state, "topk_eos_rho_mean", float("nan")),
            getattr(tracking_state, "topk_eos_rho_update_weighted", float("nan")),
            getattr(tracking_state, "topk_eos_rho_over_2_max", float("nan")),
            getattr(
                tracking_state,
                "topk_eos_rho_over_2_update_weighted",
                float("nan"),
            ),
            getattr(tracking_state, "effective_curvature_eigenvalues", ()),
            getattr(tracking_state, "extra_effective_curvature_eigenvalues", ()),
            getattr(tracking_state, "damped_effective_curvature_eigenvalues", ()),
            getattr(tracking_state, "extra_damped_effective_curvature_eigenvalues", ()),
            getattr(tracking_state, "preconditioned_dir_gain", ()),
            getattr(tracking_state, "extra_preconditioned_dir_gain", ()),
            getattr(tracking_state, "m_proj", ()),
            getattr(tracking_state, "extra_m_proj", ()),
            getattr(tracking_state, "m_norm", float("nan")),
        )
    )

    # --- Regression-based nu (computed on CPU from already-fetched arrays) ---
    _LOG_EPS = 1e-30
    _all_lambda = np.concatenate([np.asarray(eigenvalues, dtype=float),
                                   np.asarray(extra_eigenvalues, dtype=float)])
    _all_alpha  = np.concatenate([np.asarray(alpha, dtype=float),
                                   np.asarray(extra_alpha, dtype=float)])
    _all_valid  = np.concatenate([np.asarray(alpha_valid, dtype=float),
                                   np.asarray(extra_alpha_valid, dtype=float)]) > 0.5
    _all_energy = np.concatenate([np.asarray(update_energy_frac, dtype=float),
                                   np.asarray(extra_update_energy_frac, dtype=float)])
    _all_g_proj = np.concatenate([np.asarray(g_proj, dtype=float),
                                   np.asarray(extra_g_proj, dtype=float)])
    _all_d_proj = np.concatenate([np.asarray(d_proj, dtype=float),
                                   np.asarray(extra_d_proj, dtype=float)])

    _vmask = _all_valid & np.isfinite(_all_lambda) & np.isfinite(_all_alpha) & (np.abs(_all_lambda) > _LOG_EPS)
    _log_lam = np.where(_vmask, np.log(np.abs(_all_lambda) + _LOG_EPS), np.nan)
    _log_alp = np.where(_vmask, np.log(np.abs(_all_alpha) + _LOG_EPS), np.nan)
    _ew       = np.where(_vmask, np.abs(_all_energy), 0.0)

    _log_lam_valid = _log_lam[_vmask & np.isfinite(_log_lam)]
    _nu_reg_var_log_lambda = float(np.var(_log_lam_valid)) if len(_log_lam_valid) >= 2 else float("nan")

    nu_reg,    nu_reg_r2    = _nu_regression(_log_alp, _log_lam)
    nu_reg_ew, nu_reg_ew_r2 = _nu_regression(_log_alp, _log_lam, weights=_ew)

    # DAO alpha regression (uses globally-normalised projections)
    _grad_n = float(grad_norm)
    _upd_n  = float(update_norm)
    _dao_eps = 1e-10
    if _grad_n > _dao_eps and _upd_n > _dao_eps:
        _g_hat = _all_g_proj / _grad_n
        _d_hat = _all_d_proj / _upd_n
        _dao_vmask = (np.isfinite(_g_hat) & np.isfinite(_d_hat)
                      & (np.abs(_g_hat) > _dao_eps)
                      & np.isfinite(_all_lambda)
                      & (np.abs(_all_lambda) > _LOG_EPS))
        _dao_alpha   = np.where(_dao_vmask, -_d_hat / _g_hat, np.nan)
        _log_lam_dao = np.where(_dao_vmask, np.log(np.abs(_all_lambda) + _LOG_EPS), np.nan)
        _log_dao_alp = np.where(_dao_vmask, np.log(np.abs(_dao_alpha)  + _LOG_EPS), np.nan)
        _ew_dao      = np.where(_dao_vmask, np.abs(_all_energy), 0.0)
        nu_reg_dao,    nu_reg_dao_r2    = _nu_regression(_log_dao_alp, _log_lam_dao)
        nu_reg_dao_ew, nu_reg_dao_ew_r2 = _nu_regression(_log_dao_alp, _log_lam_dao, weights=_ew_dao)
    else:
        nu_reg_dao = nu_reg_dao_r2 = nu_reg_dao_ew = nu_reg_dao_ew_r2 = float("nan")

    # Momentum-reference nu (post-momentum / pre-preconditioning direction)
    _m_n = float(m_norm) if measurement_momentum else float("nan")
    if measurement_momentum and _m_n > _dao_eps and _upd_n > _dao_eps:
        _all_m_proj = np.concatenate([np.asarray(m_proj, dtype=float),
                                       np.asarray(extra_m_proj, dtype=float)])

        # Scale-dependent version: alpha_mom = -d_proj / m_proj  (same formula as nu_reg
        # but using the momentum buffer instead of the raw gradient as reference)
        _mom_ref = np.abs(_all_m_proj)
        _mom_ref_max = np.nanmax(_mom_ref) if np.any(np.isfinite(_mom_ref)) else 0.0
        _mom_tol = max(1e-10, 1e-3 * float(_mom_ref_max))
        _mom_valid = (np.isfinite(_all_m_proj) & (_mom_ref > _mom_tol)
                      & np.isfinite(_all_lambda) & (np.abs(_all_lambda) > _LOG_EPS))
        _safe_m_proj = np.where(_mom_valid, _all_m_proj, 1.0)
        _alpha_mom   = np.where(_mom_valid, -_all_d_proj / _safe_m_proj, np.nan)
        _log_lam_mom = np.where(_mom_valid, np.log(np.abs(_all_lambda) + _LOG_EPS), np.nan)
        _log_alp_mom = np.where(_mom_valid, np.log(np.abs(_alpha_mom)  + _LOG_EPS), np.nan)
        _ew_mom      = np.where(_mom_valid, np.abs(_all_energy), 0.0)
        nu_reg_mom,    nu_reg_mom_r2    = _nu_regression(_log_alp_mom, _log_lam_mom)
        nu_reg_mom_ew, nu_reg_mom_ew_r2 = _nu_regression(_log_alp_mom, _log_lam_mom, weights=_ew_mom)

        # DAO version: normalise both update and momentum reference before computing alpha
        _m_hat = _all_m_proj / _m_n
        _d_hat_mom = _all_d_proj / _upd_n
        _dao_mom_vmask = (np.isfinite(_m_hat) & np.isfinite(_d_hat_mom)
                          & (np.abs(_m_hat) > _dao_eps)
                          & np.isfinite(_all_lambda)
                          & (np.abs(_all_lambda) > _LOG_EPS))
        _dao_mom_alpha   = np.where(_dao_mom_vmask, -_d_hat_mom / _m_hat, np.nan)
        _log_lam_dao_mom = np.where(_dao_mom_vmask, np.log(np.abs(_all_lambda) + _LOG_EPS), np.nan)
        _log_dao_mom_alp = np.where(_dao_mom_vmask, np.log(np.abs(_dao_mom_alpha) + _LOG_EPS), np.nan)
        _ew_dao_mom      = np.where(_dao_mom_vmask, np.abs(_all_energy), 0.0)
        nu_reg_dao_mom,    nu_reg_dao_mom_r2    = _nu_regression(_log_dao_mom_alp, _log_lam_dao_mom)
        nu_reg_dao_mom_ew, nu_reg_dao_mom_ew_r2 = _nu_regression(_log_dao_mom_alp, _log_lam_dao_mom, weights=_ew_dao_mom)
    else:
        nu_reg_mom = nu_reg_mom_r2 = nu_reg_mom_ew = nu_reg_mom_ew_r2 = float("nan")
        nu_reg_dao_mom = nu_reg_dao_mom_r2 = nu_reg_dao_mom_ew = nu_reg_dao_mom_ew_r2 = float("nan")

    row = (
        [
            int(scalar_step),
            _metric("tracking_phase_offset"),
            _metric("tracking_phase_index"),
            _metric("tracking_phase_frac"),
            float(scalar_rotation),
            float(scalar_eff_cond),
            float(effective_curvature_cond),
            float(actual_update_rayleigh),
            float(topk_eos_rho_max),
            float(topk_eos_rho_mean),
            float(topk_eos_rho_update_weighted),
            float(topk_eos_rho_over_2_max),
            float(topk_eos_rho_over_2_update_weighted),
            _metric("probe_loss_before"),
            _metric("probe_loss_after"),
            _metric("probe_loss_reduction"),
            _metric("probe_acc_before"),
            _metric("probe_acc_after"),
            _metric("probe_acc_gain"),
            float(grad_norm),
            float(update_norm),
            float(tracked_update_energy_frac),
            float(tracked_grad_energy_frac),
            float(tracked_update_grad_cosine),
            float(full_update_grad_cosine),
            float(full_update_grad_cos2),
            float(pos_update_energy_frac),
            float(neg_update_energy_frac),
            float(pos_grad_energy_frac),
            float(neg_grad_energy_frac),
            float(pos_update_grad_cosine),
            float(neg_update_grad_cosine),
            nu_reg,
            nu_reg_r2,
            _nu_reg_var_log_lambda,
            nu_reg_ew,
            nu_reg_ew_r2,
            nu_reg_dao,
            nu_reg_dao_r2,
            nu_reg_dao_ew,
            nu_reg_dao_ew_r2,
        ]
        + (
            [
                nu_reg_mom,
                nu_reg_mom_r2,
                nu_reg_mom_ew,
                nu_reg_mom_ew_r2,
                nu_reg_dao_mom,
                nu_reg_dao_mom_r2,
                nu_reg_dao_mom_ew,
                nu_reg_dao_mom_ew_r2,
                float(m_norm),
            ]
            if measurement_momentum
            else []
        )
        + [float(x) for x in eigenvalues]
        + [float(x) for x in extra_eigenvalues]
        + [float(x) for x in ritz_residual]
        + [float(x) for x in extra_ritz_residual]
        + [float(x) for x in g_proj]
        + [float(x) for x in extra_g_proj]
        + [float(x) for x in d_proj]
        + [float(x) for x in extra_d_proj]
        + ([float(x) for x in m_proj] if measurement_momentum else [])
        + ([float(x) for x in extra_m_proj] if measurement_momentum else [])
        + [float(x) for x in update_energy_frac]
        + [float(x) for x in extra_update_energy_frac]
        + [int(bool(x)) for x in alpha_valid]
        + [int(bool(x)) for x in extra_alpha_valid]
        + [float(x) for x in alpha]
        + [float(x) for x in extra_alpha]
        + [float(x) for x in phi]
        + [float(x) for x in extra_phi]
        + [float(x) for x in eos_rho]
        + [float(x) for x in extra_eos_rho]
        + [float(x) for x in effective_curvature_eigenvalues]
        + [float(x) for x in extra_effective_curvature_eigenvalues]
        + [float(x) for x in damped_effective_curvature_eigenvalues]
        + [float(x) for x in extra_damped_effective_curvature_eigenvalues]
        + [float(x) for x in preconditioned_dir_gain]
        + [float(x) for x in extra_preconditioned_dir_gain]
    )

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


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
