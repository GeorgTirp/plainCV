#!/usr/bin/env python3
"""Plot eigen-tracking phi and nu trajectories for LLM optimizer runs.

By default this script uses the experiment folders from the current comparison
and writes PNG files into ./figures/eigen_tracking.

The eigen-tracking CSVs currently contain lambda-like columns (eig_*) and
phi-like columns (phi_*), but no psi_* columns. Nu is therefore inferred from
phi by default. Since phi is defined as alpha * lambda / learning_rate, the
exponent interpretation is:

    |phi_i^t| = |lambda_i^t| ** (1 - nu_i^t)
    nu_i^t = 1 - log |phi_i^t| / log |lambda_i^t|

The previous plotting formula clipped this value into [0, 1], which made most
runs look like a few isolated spikes whenever |phi| was much larger than the
eigenvalue scale. Use --clip-nu to restore that bounded diagnostic. If future
CSVs include psi_* columns, pass --psi-prefix psi.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUN_DIRS = [
    "exp/llm/paper_runs/GGN_muon",
    "exp/llm/paper_runs/GGN_soap",
    "exp/llm/paper_runs/GGN_ni_soap",
    "exp/llm/paper_runs/GGN_adam",
    "exp/llm/paper_runs/GGN_signum",
]

_OPTIMIZER_FRIENDLY = {
    "muon": "Muon",
    "soap": "SOAP",
    "ni_soap": "NI-SOAP",
    "adam": "Adam",
    "adamw": "Adam",
    "signum": "Signum",
}


@dataclass(frozen=True)
class Run:
    label: str
    optimizer: str
    path: Path
    steps: np.ndarray
    top_lambdas: np.ndarray
    extra_lambdas: np.ndarray
    top_phi: np.ndarray
    extra_phi: np.ndarray
    top_eos_rho: np.ndarray
    extra_eos_rho: np.ndarray
    topk_eos_rho_max: np.ndarray
    topk_eos_rho_mean: np.ndarray
    topk_eos_rho_update_weighted: np.ndarray
    topk_eos_rho_over_2_max: np.ndarray
    topk_eos_rho_over_2_update_weighted: np.ndarray
    top_nu: np.ndarray
    extra_nu: np.ndarray
    top_dao_nu: np.ndarray
    extra_dao_nu: np.ndarray
    top_update_energy_frac: np.ndarray
    extra_update_energy_frac: np.ndarray
    probe_loss_before: np.ndarray
    probe_loss_reduction: np.ndarray
    probe_acc_gain: np.ndarray
    tracked_update_energy_frac: np.ndarray
    tracked_update_grad_cosine: np.ndarray
    nu_reg: np.ndarray
    nu_reg_r2: np.ndarray
    nu_reg_var_log_lambda: np.ndarray
    nu_reg_ew: np.ndarray
    nu_reg_ew_r2: np.ndarray
    nu_reg_dao: np.ndarray
    nu_reg_dao_r2: np.ndarray
    nu_reg_dao_ew: np.ndarray
    nu_reg_dao_ew_r2: np.ndarray
    # Effective curvature fields (present when effective_curvature_tracking_enabled=true)
    top_eff_curv_eig: np.ndarray
    extra_eff_curv_eig: np.ndarray
    top_precond_dir_gain: np.ndarray
    extra_precond_dir_gain: np.ndarray
    eff_cond: np.ndarray
    effective_curvature_cond: np.ndarray
    actual_update_rayleigh: np.ndarray


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_run_dir(path: str) -> Path:
    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    root = _repo_root()
    if raw_path.parts and raw_path.parts[0] == root.name:
        raw_path = Path(*raw_path.parts[1:])
    return (root / raw_path).resolve()


def _dedupe_paths(paths: Iterable[str]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = _resolve_run_dir(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _read_simple_yaml_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None

    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.*?)\s*(?:#.*)?$")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip().strip("'\"") or None
    return None


def _optimizer_from_path(run_dir: Path) -> str:
    optim = _read_simple_yaml_value(run_dir / "config.yaml", "optim")
    if optim:
        return _OPTIMIZER_FRIENDLY.get(optim, optim)

    match = re.search(r"lm_([^_/]+)_job", run_dir.name)
    if match:
        raw = match.group(1)
        return _OPTIMIZER_FRIENDLY.get(raw, raw)
    return run_dir.name


def _numbered_columns(fieldnames: list[str], prefix: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    indexed: list[tuple[int, str]] = []
    for fieldname in fieldnames:
        match = pattern.match(fieldname)
        if match:
            indexed.append((int(match.group(1)), fieldname))
    return [name for _, name in sorted(indexed)]


def _read_columns(csv_path: Path, columns: list[str]) -> np.ndarray:
    values: list[list[float]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            current_row: list[float] = []
            for column in columns:
                raw_value = row.get(column, "")
                try:
                    current_row.append(float(raw_value))
                except ValueError:
                    current_row.append(float("nan"))
            values.append(current_row)
    return np.asarray(values, dtype=float)


def _compute_nu(
    psi: np.ndarray,
    lambdas: np.ndarray,
    eps: float,
    clip_nu: bool,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = np.log(np.abs(psi))
        denominator = np.log(np.abs(lambdas))
        denominator = np.where(np.abs(denominator) > eps, denominator, np.nan)
        nu = 1.0 - numerator / denominator
    if clip_nu:
        nu = np.clip(nu, 0.0, 1.0)
    return np.where(np.isfinite(nu), nu, np.nan)


def _compute_dao_phi(
    lambdas: np.ndarray,
    g_proj: np.ndarray,
    d_proj: np.ndarray,
    grad_norm: np.ndarray,
    update_norm: np.ndarray,
    alpha_valid: np.ndarray,
    eps: float,
) -> np.ndarray:
    if lambdas.size == 0:
        return np.empty_like(lambdas)
    if (
        g_proj.shape != lambdas.shape
        or d_proj.shape != lambdas.shape
        or grad_norm.ndim != 1
        or update_norm.ndim != 1
        or grad_norm.shape[0] != lambdas.shape[0]
        or update_norm.shape[0] != lambdas.shape[0]
    ):
        return np.full_like(lambdas, np.nan, dtype=float)

    grad = grad_norm[:, None]
    update = update_norm[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        g_hat = g_proj / grad
        d_hat = d_proj / update
        alpha_dir = -d_hat / g_hat
        phi_dir = alpha_dir * lambdas

    valid = np.isfinite(phi_dir)
    valid = np.logical_and(valid, np.isfinite(g_hat))
    valid = np.logical_and(valid, np.isfinite(d_hat))
    valid = np.logical_and(valid, np.abs(g_hat) > eps)
    valid = np.logical_and(valid, np.abs(grad) > eps)
    valid = np.logical_and(valid, np.abs(update) > eps)
    if alpha_valid.shape == lambdas.shape:
        valid = np.logical_and(valid, alpha_valid > 0.5)
    return np.where(valid, phi_dir, np.nan)


def _move_bulk_top_modes_to_extra(
    *,
    top_lambdas: np.ndarray,
    top_phi: np.ndarray,
    top_eos_rho: np.ndarray,
    top_nu: np.ndarray,
    top_dao_nu: np.ndarray,
    top_update_energy_frac: np.ndarray,
    extra_lambdas: np.ndarray,
    extra_phi: np.ndarray,
    extra_eos_rho: np.ndarray,
    extra_nu: np.ndarray,
    extra_dao_nu: np.ndarray,
    extra_update_energy_frac: np.ndarray,
    eig_threshold: float | None,
    relative_threshold: float | None,
    min_fraction: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if (eig_threshold is None and relative_threshold is None) or top_lambdas.size == 0:
        return (
            top_lambdas,
            top_phi,
            top_eos_rho,
            top_nu,
            top_dao_nu,
            top_update_energy_frac,
            extra_lambdas,
            extra_phi,
            extra_eos_rho,
            extra_nu,
            extra_dao_nu,
            extra_update_energy_frac,
        )

    finite = np.isfinite(top_lambdas)
    below = np.zeros_like(top_lambdas, dtype=bool)
    if eig_threshold is not None:
        below = np.logical_or(below, np.abs(top_lambdas) < eig_threshold)

    if relative_threshold is not None:
        if relative_threshold <= 0.0:
            raise ValueError("--bulk-eig-relative-threshold must be positive.")
        reference = np.abs(top_lambdas[:, [0]])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(top_lambdas) / reference
        relative_below = np.logical_and(
            np.isfinite(ratio),
            np.logical_and(reference > 0.0, ratio < relative_threshold),
        )
        below = np.logical_or(below, relative_below)

    counts = finite.sum(axis=0)
    below_counts = np.logical_and(finite, below).sum(axis=0)
    fractions = np.divide(
        below_counts,
        counts,
        out=np.zeros_like(below_counts, dtype=float),
        where=counts > 0,
    )
    move_mask = fractions >= min_fraction
    keep_mask = ~move_mask

    moved_lambdas = top_lambdas[:, move_mask]
    moved_phi = top_phi[:, move_mask]
    moved_eos_rho = top_eos_rho[:, move_mask]
    moved_nu = top_nu[:, move_mask]
    moved_dao_nu = top_dao_nu[:, move_mask]
    moved_update_energy_frac = top_update_energy_frac[:, move_mask]
    kept_lambdas = top_lambdas[:, keep_mask]
    kept_phi = top_phi[:, keep_mask]
    kept_eos_rho = top_eos_rho[:, keep_mask]
    kept_nu = top_nu[:, keep_mask]
    kept_dao_nu = top_dao_nu[:, keep_mask]
    kept_update_energy_frac = top_update_energy_frac[:, keep_mask]

    if extra_lambdas.size:
        new_extra_lambdas = np.concatenate([extra_lambdas, moved_lambdas], axis=1)
        new_extra_phi = np.concatenate([extra_phi, moved_phi], axis=1)
        new_extra_eos_rho = np.concatenate([extra_eos_rho, moved_eos_rho], axis=1)
        new_extra_nu = np.concatenate([extra_nu, moved_nu], axis=1)
        new_extra_dao_nu = np.concatenate([extra_dao_nu, moved_dao_nu], axis=1)
        new_extra_update_energy_frac = np.concatenate(
            [extra_update_energy_frac, moved_update_energy_frac],
            axis=1,
        )
    else:
        new_extra_lambdas = moved_lambdas
        new_extra_phi = moved_phi
        new_extra_eos_rho = moved_eos_rho
        new_extra_nu = moved_nu
        new_extra_dao_nu = moved_dao_nu
        new_extra_update_energy_frac = moved_update_energy_frac

    return (
        kept_lambdas,
        kept_phi,
        kept_eos_rho,
        kept_nu,
        kept_dao_nu,
        kept_update_energy_frac,
        new_extra_lambdas,
        new_extra_phi,
        new_extra_eos_rho,
        new_extra_nu,
        new_extra_dao_nu,
        new_extra_update_energy_frac,
    )


def _load_run(
    run_dir: Path,
    psi_prefix: str,
    eps: float,
    clip_nu: bool,
    bulk_eig_threshold: float | None,
    bulk_eig_relative_threshold: float | None,
    bulk_eig_min_fraction: float,
) -> Run:
    csv_path = run_dir / "eigen_tracking.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing eigen_tracking.csv in {run_dir}")

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

    top_lambda_cols = _numbered_columns(fieldnames, "eig")
    extra_lambda_cols = _numbered_columns(fieldnames, "extra_eig")
    top_phi_cols = _numbered_columns(fieldnames, "phi")
    extra_phi_cols = _numbered_columns(fieldnames, "extra_phi")
    top_eos_rho_cols = _numbered_columns(fieldnames, "eos_rho")
    extra_eos_rho_cols = _numbered_columns(fieldnames, "extra_eos_rho")
    top_g_proj_cols = _numbered_columns(fieldnames, "g_proj")
    extra_g_proj_cols = _numbered_columns(fieldnames, "extra_g_proj")
    top_d_proj_cols = _numbered_columns(fieldnames, "d_proj")
    extra_d_proj_cols = _numbered_columns(fieldnames, "extra_d_proj")
    top_update_energy_frac_cols = _numbered_columns(fieldnames, "update_energy_frac")
    extra_update_energy_frac_cols = _numbered_columns(fieldnames, "extra_update_energy_frac")
    top_alpha_valid_cols = _numbered_columns(fieldnames, "alpha_valid")
    extra_alpha_valid_cols = _numbered_columns(fieldnames, "extra_alpha_valid")
    top_psi_cols = _numbered_columns(fieldnames, psi_prefix)
    extra_psi_cols = _numbered_columns(fieldnames, f"extra_{psi_prefix}")
    top_eff_curv_eig_cols = _numbered_columns(fieldnames, "eff_curv_eig")
    extra_eff_curv_eig_cols = _numbered_columns(fieldnames, "extra_eff_curv_eig")
    top_precond_dir_gain_cols = _numbered_columns(fieldnames, "precond_dir_gain")
    extra_precond_dir_gain_cols = _numbered_columns(fieldnames, "extra_precond_dir_gain")

    if not top_lambda_cols or not top_phi_cols:
        raise ValueError(f"{csv_path} does not contain the expected eig_* and phi_* columns.")
    if not top_psi_cols:
        top_psi_cols = top_phi_cols
    if not extra_psi_cols:
        extra_psi_cols = extra_phi_cols

    steps = _read_columns(csv_path, ["global_step"])[:, 0]
    top_lambdas = _read_columns(csv_path, top_lambda_cols)
    extra_lambdas = _read_columns(csv_path, extra_lambda_cols)
    top_phi = _read_columns(csv_path, top_phi_cols)
    extra_phi = _read_columns(csv_path, extra_phi_cols)
    top_eos_rho = _read_columns(csv_path, top_eos_rho_cols)
    extra_eos_rho = _read_columns(csv_path, extra_eos_rho_cols)
    top_psi = _read_columns(csv_path, top_psi_cols)
    extra_psi = _read_columns(csv_path, extra_psi_cols)
    top_g_proj = _read_columns(csv_path, top_g_proj_cols)
    extra_g_proj = _read_columns(csv_path, extra_g_proj_cols)
    top_d_proj = _read_columns(csv_path, top_d_proj_cols)
    extra_d_proj = _read_columns(csv_path, extra_d_proj_cols)
    top_update_energy_frac = _read_columns(csv_path, top_update_energy_frac_cols)
    extra_update_energy_frac = _read_columns(csv_path, extra_update_energy_frac_cols)
    top_alpha_valid = _read_columns(csv_path, top_alpha_valid_cols)
    extra_alpha_valid = _read_columns(csv_path, extra_alpha_valid_cols)
    grad_norm = _read_columns(csv_path, ["grad_norm"])[:, 0]
    update_norm = _read_columns(csv_path, ["update_norm"])[:, 0]
    probe_loss_before = _read_columns(csv_path, ["probe_loss_before"])[:, 0]
    probe_loss_reduction = _read_columns(csv_path, ["probe_loss_reduction"])[:, 0]
    probe_acc_gain = _read_columns(csv_path, ["probe_acc_gain"])[:, 0]
    tracked_update_energy_frac = _read_columns(csv_path, ["tracked_update_energy_frac"])[:, 0]
    tracked_update_grad_cosine = _read_columns(csv_path, ["tracked_update_grad_cosine"])[:, 0]
    nu_reg              = _read_columns(csv_path, ["nu_reg"])[:, 0]
    nu_reg_r2           = _read_columns(csv_path, ["nu_reg_r2"])[:, 0]
    nu_reg_var_log_lambda = _read_columns(csv_path, ["nu_reg_var_log_lambda"])[:, 0]
    nu_reg_ew           = _read_columns(csv_path, ["nu_reg_ew"])[:, 0]
    nu_reg_ew_r2        = _read_columns(csv_path, ["nu_reg_ew_r2"])[:, 0]
    nu_reg_dao          = _read_columns(csv_path, ["nu_reg_dao"])[:, 0]
    nu_reg_dao_r2       = _read_columns(csv_path, ["nu_reg_dao_r2"])[:, 0]
    nu_reg_dao_ew       = _read_columns(csv_path, ["nu_reg_dao_ew"])[:, 0]
    nu_reg_dao_ew_r2    = _read_columns(csv_path, ["nu_reg_dao_ew_r2"])[:, 0]

    top_eff_curv_eig = _read_columns(csv_path, top_eff_curv_eig_cols) if top_eff_curv_eig_cols else np.empty((len(steps), 0))
    extra_eff_curv_eig = _read_columns(csv_path, extra_eff_curv_eig_cols) if extra_eff_curv_eig_cols else np.empty((len(steps), 0))
    top_precond_dir_gain = _read_columns(csv_path, top_precond_dir_gain_cols) if top_precond_dir_gain_cols else np.empty((len(steps), 0))
    extra_precond_dir_gain = _read_columns(csv_path, extra_precond_dir_gain_cols) if extra_precond_dir_gain_cols else np.empty((len(steps), 0))
    eff_cond = _read_columns(csv_path, ["eff_cond"])[:, 0] if "eff_cond" in fieldnames else np.full(len(steps), np.nan)
    effective_curvature_cond = _read_columns(csv_path, ["effective_curvature_cond"])[:, 0] if "effective_curvature_cond" in fieldnames else np.full(len(steps), np.nan)
    actual_update_rayleigh = _read_columns(csv_path, ["actual_update_rayleigh"])[:, 0] if "actual_update_rayleigh" in fieldnames else np.full(len(steps), np.nan)

    if not top_eos_rho_cols:
        lr_raw = _read_simple_yaml_value(run_dir / "config.yaml", "lr")
        try:
            lr = abs(float(lr_raw)) if lr_raw is not None else float("nan")
        except ValueError:
            lr = float("nan")
        top_eos_rho = np.abs(top_phi) * lr
        extra_eos_rho = np.abs(extra_phi) * lr

    top_nu = _compute_nu(top_psi, top_lambdas, eps, clip_nu=clip_nu)
    extra_nu = _compute_nu(extra_psi, extra_lambdas, eps, clip_nu=clip_nu)
    top_dao_phi = _compute_dao_phi(
        top_lambdas,
        top_g_proj,
        top_d_proj,
        grad_norm,
        update_norm,
        top_alpha_valid,
        eps,
    )
    extra_dao_phi = _compute_dao_phi(
        extra_lambdas,
        extra_g_proj,
        extra_d_proj,
        grad_norm,
        update_norm,
        extra_alpha_valid,
        eps,
    )
    top_dao_nu = _compute_nu(top_dao_phi, top_lambdas, eps, clip_nu=clip_nu)
    extra_dao_nu = _compute_nu(extra_dao_phi, extra_lambdas, eps, clip_nu=clip_nu)
    (
        top_lambdas,
        top_phi,
        top_eos_rho,
        top_nu,
        top_dao_nu,
        top_update_energy_frac,
        extra_lambdas,
        extra_phi,
        extra_eos_rho,
        extra_nu,
        extra_dao_nu,
        extra_update_energy_frac,
    ) = _move_bulk_top_modes_to_extra(
        top_lambdas=top_lambdas,
        top_phi=top_phi,
        top_eos_rho=top_eos_rho,
        top_nu=top_nu,
        top_dao_nu=top_dao_nu,
        top_update_energy_frac=top_update_energy_frac,
        extra_lambdas=extra_lambdas,
        extra_phi=extra_phi,
        extra_eos_rho=extra_eos_rho,
        extra_nu=extra_nu,
        extra_dao_nu=extra_dao_nu,
        extra_update_energy_frac=extra_update_energy_frac,
        eig_threshold=bulk_eig_threshold,
        relative_threshold=bulk_eig_relative_threshold,
        min_fraction=bulk_eig_min_fraction,
    )
    optimizer = _optimizer_from_path(run_dir)

    return Run(
        label=optimizer,
        optimizer=optimizer,
        path=run_dir,
        steps=steps,
        top_lambdas=top_lambdas,
        extra_lambdas=extra_lambdas,
        top_phi=top_phi,
        extra_phi=extra_phi,
        top_eos_rho=top_eos_rho,
        extra_eos_rho=extra_eos_rho,
        topk_eos_rho_max=_nanmax(top_eos_rho, axis=1),
        topk_eos_rho_mean=_nanmean(top_eos_rho, axis=1),
        topk_eos_rho_update_weighted=_weighted_nanmean(
            top_eos_rho,
            top_update_energy_frac,
            axis=1,
        ),
        topk_eos_rho_over_2_max=_nanmax(top_eos_rho, axis=1) / 2.0,
        topk_eos_rho_over_2_update_weighted=_weighted_nanmean(
            top_eos_rho,
            top_update_energy_frac,
            axis=1,
        )
        / 2.0,
        top_nu=top_nu,
        extra_nu=extra_nu,
        top_dao_nu=top_dao_nu,
        extra_dao_nu=extra_dao_nu,
        top_update_energy_frac=top_update_energy_frac,
        extra_update_energy_frac=extra_update_energy_frac,
        probe_loss_before=probe_loss_before,
        probe_loss_reduction=probe_loss_reduction,
        probe_acc_gain=probe_acc_gain,
        tracked_update_energy_frac=tracked_update_energy_frac,
        tracked_update_grad_cosine=tracked_update_grad_cosine,
        nu_reg=nu_reg,
        nu_reg_r2=nu_reg_r2,
        nu_reg_var_log_lambda=nu_reg_var_log_lambda,
        nu_reg_ew=nu_reg_ew,
        nu_reg_ew_r2=nu_reg_ew_r2,
        nu_reg_dao=nu_reg_dao,
        nu_reg_dao_r2=nu_reg_dao_r2,
        nu_reg_dao_ew=nu_reg_dao_ew,
        nu_reg_dao_ew_r2=nu_reg_dao_ew_r2,
        top_eff_curv_eig=top_eff_curv_eig,
        extra_eff_curv_eig=extra_eff_curv_eig,
        top_precond_dir_gain=top_precond_dir_gain,
        extra_precond_dir_gain=extra_precond_dir_gain,
        eff_cond=eff_cond,
        effective_curvature_cond=effective_curvature_cond,
        actual_update_rayleigh=actual_update_rayleigh,
    )


def _disambiguate_labels(runs: list[Run]) -> list[Run]:
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.optimizer] = counts.get(run.optimizer, 0) + 1

    labeled_runs: list[Run] = []
    for run in runs:
        if counts[run.optimizer] == 1:
            labeled_runs.append(run)
            continue
        label = f"{run.optimizer} ({run.path.name})"
        labeled_runs.append(Run(label=label, **{k: v for k, v in run.__dict__.items() if k != "label"}))
    return labeled_runs


def _setup_axes(title: str, ylabel: str, yscale: str | None = None) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.set_title(title)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.25)
    if yscale:
        ax.set_yscale(yscale)
    return fig, ax


def _plot_direction_family(
    steps: np.ndarray,
    values: np.ndarray,
    title: str,
    ylabel: str,
    output_path: Path,
    yscale: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = _setup_axes(title, ylabel, yscale=yscale)
    if values.size == 0 or values.shape[1] == 0:
        ax.text(0.5, 0.5, "No directions", transform=ax.transAxes, ha="center", va="center")
    for idx in range(values.shape[1]):
        ax.plot(steps, values[:, idx], linewidth=1.8, label=f"direction {idx}")
    if ylim:
        ax.set_ylim(*ylim)
    if values.size and values.shape[1] > 0:
        ax.legend(ncol=2, fontsize=9)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_eos_direction_family(
    steps: np.ndarray,
    values: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = _setup_axes(title, "EOS rho = |alpha lambda|")
    if values.size == 0 or values.shape[1] == 0:
        ax.text(0.5, 0.5, "No directions", transform=ax.transAxes, ha="center", va="center")
    for idx in range(values.shape[1]):
        ax.plot(steps, values[:, idx], linewidth=1.8, label=f"direction {idx}")
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1.2, alpha=0.65, label="GD edge rho=2")
    finite_positive = values[np.logical_and(np.isfinite(values), values > 0.0)]
    if finite_positive.size:
        ax.set_ylim(0.0, max(2.2, float(np.max(finite_positive)) * 1.08))
    if values.size and values.shape[1] > 0:
        ax.legend(ncol=2, fontsize=9)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    valid = np.isfinite(values)
    counts = valid.sum(axis=axis)
    sums = np.where(valid, values, 0.0).sum(axis=axis)
    return np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)


def _nanmax(values: np.ndarray, axis: int) -> np.ndarray:
    if values.size == 0:
        length = values.shape[0] if values.ndim > 0 else 0
        return np.full((length,), np.nan, dtype=float)
    valid = np.isfinite(values)
    has_valid = valid.any(axis=axis)
    max_values = np.max(np.where(valid, values, -np.inf), axis=axis)
    return np.where(has_valid, max_values, np.nan)


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    valid = np.logical_and(np.isfinite(values), np.isfinite(weights))
    valid = np.logical_and(valid, weights > 0.0)
    weighted_sums = np.where(valid, values * weights, 0.0).sum(axis=axis)
    weight_sums = np.where(valid, weights, 0.0).sum(axis=axis)
    return np.divide(
        weighted_sums,
        weight_sums,
        out=np.full_like(weighted_sums, np.nan, dtype=float),
        where=weight_sums > 0,
    )


def _relative_probe_loss_reduction(run: Run, eps: float = 1e-12) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = run.probe_loss_reduction / run.probe_loss_before
    valid = np.logical_and(
        np.isfinite(rel),
        np.logical_and(np.isfinite(run.probe_loss_before), np.abs(run.probe_loss_before) > eps),
    )
    return np.where(valid, rel, np.nan)


def _energy_weighted_topk_nu(run: Run, *, dao: bool = False) -> np.ndarray:
    values = run.top_dao_nu if dao else run.top_nu
    return _weighted_nanmean(values, run.top_update_energy_frac, axis=1)


def _energy_weighted_topk_eos_rho(run: Run) -> np.ndarray:
    return _weighted_nanmean(run.top_eos_rho, run.top_update_energy_frac, axis=1)


def _set_dynamic_ylim(ax: plt.Axes, values: np.ndarray, pad_fraction: float = 0.08) -> None:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        ax.set_ylim(0.0, 1.0)
        return

    ymin = float(np.min(finite_values))
    ymax = float(np.max(finite_values))
    if math.isclose(ymin, ymax):
        pad = max(abs(ymin) * pad_fraction, 1e-3)
    else:
        pad = (ymax - ymin) * pad_fraction

    ax.set_ylim(ymin - pad, ymax + pad)


def _plot_mean_nu(
    runs: list[Run],
    category: str,
    output_path: Path,
    title: str,
    *,
    dao: bool = False,
) -> None:
    ylabel = "Mean nu DAO" if dao else "Mean nu"
    fig, ax = _setup_axes(title, ylabel)
    mean_values: list[np.ndarray] = []
    for run in runs:
        if category == "top":
            values = run.top_dao_nu if dao else run.top_nu
        else:
            values = run.extra_dao_nu if dao else run.extra_nu
        if values.size == 0:
            continue
        mean_value = _nanmean(values, axis=1)
        mean_values.append(mean_value)
        ax.plot(run.steps, mean_value, linewidth=2.2, label=run.label)
    if mean_values:
        _set_dynamic_ylim(ax, np.concatenate(mean_values))
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No directions", transform=ax.transAxes, ha="center", va="center")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_mean_eos_rho(
    runs: list[Run],
    output_path: Path,
    title: str,
    *,
    value: str = "mean",
) -> None:
    fig, ax = _setup_axes(title, "EOS rho = |alpha lambda|")
    values_for_ylim: list[np.ndarray] = []
    for run in runs:
        if value == "weighted":
            series = _energy_weighted_topk_eos_rho(run)
        elif value == "max":
            series = _nanmax(run.top_eos_rho, axis=1)
        else:
            series = _nanmean(run.top_eos_rho, axis=1)
        if series.size == 0:
            continue
        values_for_ylim.append(series)
        ax.plot(run.steps, series, linewidth=2.2, label=run.label)
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1.2, alpha=0.65, label="GD edge rho=2")
    if values_for_ylim:
        finite = np.concatenate([v[np.isfinite(v)] for v in values_for_ylim])
        if finite.size:
            ax.set_ylim(0.0, max(2.2, float(np.max(finite)) * 1.08))
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No finite EOS rho points", transform=ax.transAxes, ha="center", va="center")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _final_abs_values(values: np.ndarray) -> np.ndarray:
    if values.size == 0 or values.shape[0] == 0 or values.shape[1] == 0:
        return np.asarray([], dtype=float)
    return np.abs(values[-1, :])


def _mean_abs_values(values: np.ndarray) -> np.ndarray:
    if values.size == 0 or values.shape[0] == 0 or values.shape[1] == 0:
        return np.asarray([], dtype=float)
    return np.nanmean(np.abs(values), axis=0)


def _use_log_scale_for_bars(ax: plt.Axes, values: np.ndarray) -> None:
    finite_positive = values[np.logical_and(np.isfinite(values), values > 0.0)]
    if finite_positive.size == 0:
        return
    ax.set_yscale("log")
    ax.set_ylim(float(np.min(finite_positive)) * 0.7, float(np.max(finite_positive)) * 1.35)


def _plot_eig_bar(
    values: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.set_title(title)
    ax.set_xlabel("Direction index")
    ax.set_ylabel("Final |eigenvalue|")
    ax.grid(True, axis="y", alpha=0.25)

    if values.size == 0:
        ax.text(0.5, 0.5, "No directions", transform=ax.transAxes, ha="center", va="center")
    else:
        x = np.arange(values.shape[0])
        ax.bar(x, values, width=0.72)
        ax.set_xticks(x)
        _use_log_scale_for_bars(ax, values)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_grouped_eig_bars(
    runs: list[Run],
    category: str,
    output_path: Path,
    title: str,
    value_fn,
) -> None:
    eig_values = [
        value_fn(run.top_lambdas if category == "top" else run.extra_lambdas)
        for run in runs
    ]
    max_modes = max((values.shape[0] for values in eig_values), default=0)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    ax.set_title(title)
    ax.set_xlabel("Direction index")
    ax.set_ylabel("Final |eigenvalue|")
    ax.grid(True, axis="y", alpha=0.25)

    if max_modes == 0:
        ax.text(0.5, 0.5, "No directions", transform=ax.transAxes, ha="center", va="center")
    else:
        x = np.arange(max_modes)
        width = min(0.8 / max(len(runs), 1), 0.18)
        offsets = (np.arange(len(runs)) - (len(runs) - 1) / 2.0) * width
        all_values: list[np.ndarray] = []
        for run_idx, (run, values) in enumerate(zip(runs, eig_values, strict=True)):
            padded = np.full(max_modes, np.nan, dtype=float)
            padded[: values.shape[0]] = values
            mask = np.isfinite(padded)
            if not np.any(mask):
                continue
            all_values.append(padded[mask])
            ax.bar(x[mask] + offsets[run_idx], padded[mask], width=width, label=run.label)
        ax.set_xticks(x)
        if all_values:
            _use_log_scale_for_bars(ax, np.concatenate(all_values))
            ax.legend(fontsize=9)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_optimizer_mean_eig_bars(
    runs: list[Run],
    output_path: Path,
    value_fn,
    title: str,
) -> None:
    labels = [run.label for run in runs]
    top_means = np.asarray(
        [
            np.nanmean(value_fn(run.top_lambdas))
            if value_fn(run.top_lambdas).size
            else np.nan
            for run in runs
        ],
        dtype=float,
    )
    extra_means = np.asarray(
        [
            np.nanmean(value_fn(run.extra_lambdas))
            if value_fn(run.extra_lambdas).size
            else np.nan
            for run in runs
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    ax.set_title(title)
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Mean final |eigenvalue|")
    ax.grid(True, axis="y", alpha=0.25)

    x = np.arange(len(runs))
    width = 0.34
    ax.bar(x - width / 2.0, top_means, width=width, label="top-k")
    ax.bar(x + width / 2.0, extra_means, width=width, label="extra")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    finite_values = np.concatenate([top_means[np.isfinite(top_means)], extra_means[np.isfinite(extra_means)]])
    _use_log_scale_for_bars(ax, finite_values)
    ax.legend(fontsize=9)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_loss_corr_weighted_nu(
    runs: list[Run],
    output_path: Path,
    *,
    dao: bool = False,
) -> None:
    x_label = "Energy-weighted mean top-k DAO nu" if dao else "Energy-weighted mean top-k nu"
    title_prefix = "DAO " if dao else ""
    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    ax.set_title(f"{title_prefix}nu vs relative probe loss reduction")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Probe loss reduction / probe loss before")
    ax.grid(True, alpha=0.25)

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for run in runs:
        x = _energy_weighted_topk_nu(run, dao=dao)
        y = _relative_probe_loss_reduction(run)
        mask = np.logical_and(np.isfinite(x), np.isfinite(y))
        if not np.any(mask):
            continue
        all_x.append(x[mask])
        all_y.append(y[mask])
        ax.scatter(x[mask], y[mask], s=42, alpha=0.78, label=run.label)

    if all_x:
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No finite probe-loss points", transform=ax.transAxes, ha="center", va="center")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_loss_corr_interaction(
    runs: list[Run],
    output_path: Path,
    *,
    dao: bool = False,
) -> None:
    x_label = "Energy-weighted mean top-k DAO nu" if dao else "Energy-weighted mean top-k nu"
    title_prefix = "DAO " if dao else ""
    color_values = []
    for run in runs:
        color_values.append(_relative_probe_loss_reduction(run))
    finite_colors = np.concatenate([x[np.isfinite(x)] for x in color_values if x.size])
    vmin = float(np.min(finite_colors)) if finite_colors.size else None
    vmax = float(np.max(finite_colors)) if finite_colors.size else None

    fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    ax.set_title(f"{title_prefix}nu, subspace descent alignment, and loss reduction")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Tracked update-gradient cosine")
    ax.grid(True, alpha=0.25)

    plotted = False
    last_scatter = None
    for run in runs:
        x = _energy_weighted_topk_nu(run, dao=dao)
        y = run.tracked_update_grad_cosine
        c = _relative_probe_loss_reduction(run)
        energy = np.clip(run.tracked_update_energy_frac, 0.0, 1.0)
        size = 35.0 + 260.0 * energy
        mask = np.logical_and.reduce(
            (
                np.isfinite(x),
                np.isfinite(y),
                np.isfinite(c),
                np.isfinite(size),
            )
        )
        if not np.any(mask):
            continue
        last_scatter = ax.scatter(
            x[mask],
            y[mask],
            c=c[mask],
            s=size[mask],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            alpha=0.78,
            edgecolors="black",
            linewidths=0.35,
            label=run.label,
        )
        plotted = True

    if plotted:
        ax.legend(fontsize=9)
        if last_scatter is not None:
            cbar = fig.colorbar(last_scatter, ax=ax)
            cbar.set_label("Probe loss reduction / probe loss before")
    else:
        ax.text(0.5, 0.5, "No finite probe-loss points", transform=ax.transAxes, ha="center", va="center")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_measured_vs_effective_curvature(
    runs: list[Run],
    category: str,
    output_path: Path,
    title: str,
) -> None:
    """Side-by-side bar chart: mean measured eigenvalue vs mean effective curvature eigenvalue."""
    labels = [run.label for run in runs]
    if category == "top":
        measured = [np.nanmean(np.abs(run.top_lambdas)) if run.top_lambdas.size else np.nan for run in runs]
        effective = [np.nanmean(np.abs(run.top_eff_curv_eig)) if run.top_eff_curv_eig.size else np.nan for run in runs]
    else:
        measured = [np.nanmean(np.abs(run.extra_lambdas)) if run.extra_lambdas.size else np.nan for run in runs]
        effective = [np.nanmean(np.abs(run.extra_eff_curv_eig)) if run.extra_eff_curv_eig.size else np.nan for run in runs]

    measured_arr = np.asarray(measured, dtype=float)
    effective_arr = np.asarray(effective, dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    ax.set_title(title)
    ax.set_xlabel("Optimizer")
    ax.set_ylabel("Mean |eigenvalue|")
    ax.grid(True, axis="y", alpha=0.25)

    x = np.arange(len(runs))
    width = 0.34
    ax.bar(x - width / 2.0, measured_arr, width=width, label="measured")
    ax.bar(x + width / 2.0, effective_arr, width=width, label="effective")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    finite_all = np.concatenate([
        measured_arr[np.isfinite(measured_arr)],
        effective_arr[np.isfinite(effective_arr)],
    ])
    _use_log_scale_for_bars(ax, finite_all)
    ax.legend(fontsize=9)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _set_log_ylim(ax: plt.Axes, values: np.ndarray) -> None:
    finite_pos = values[np.logical_and(np.isfinite(values), values > 0.0)]
    if finite_pos.size:
        ax.set_ylim(float(np.min(finite_pos)) * 0.5, float(np.max(finite_pos)) * 2.0)


def _plot_eff_curv_trajectory(
    runs: list[Run],
    category: str,
    output_path: Path,
    title: str,
) -> None:
    """Mean effective curvature eigenvalue trajectory per optimizer."""
    fig, ax = _setup_axes(title, "Mean |eff. curv. eigenvalue|", yscale="log")
    mean_values: list[np.ndarray] = []
    for run in runs:
        arr = run.top_eff_curv_eig if category == "top" else run.extra_eff_curv_eig
        if arr.size == 0:
            continue
        mean_val = _nanmean(np.abs(arr), axis=1)
        mean_values.append(mean_val)
        ax.plot(run.steps, mean_val, linewidth=2.2, label=run.label)
    if mean_values:
        _set_log_ylim(ax, np.concatenate(mean_values))
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No effective curvature data", transform=ax.transAxes, ha="center", va="center")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_measured_vs_effective_trajectory(
    run: Run,
    category: str,
    output_path: Path,
) -> None:
    """Side-by-side trajectory: mean measured eig (left) vs mean eff curvature eig (right)."""
    lambdas = run.top_lambdas if category == "top" else run.extra_lambdas
    eff_eig = run.top_eff_curv_eig if category == "top" else run.extra_eff_curv_eig
    kind = "top-k" if category == "top" else "extra"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle(f"{run.label}: measured vs effective curvature ({kind} directions)")

    def _apply_log_if_warranted(ax: plt.Axes, data: np.ndarray) -> None:
        pos = data[np.logical_and(np.isfinite(data), data > 0)]
        if pos.size >= 2 and float(np.max(pos)) / float(np.min(pos)) > 20.0:
            ax.set_yscale("log")
            _set_log_ylim(ax, pos)

    ax_m = axes[0]
    ax_m.set_title("Measured eigenvalues")
    ax_m.set_xlabel("Optimizer step")
    ax_m.set_ylabel("Eigenvalue")
    ax_m.grid(True, alpha=0.25)
    if lambdas.size and lambdas.shape[1] > 0:
        for idx in range(lambdas.shape[1]):
            ax_m.plot(run.steps, lambdas[:, idx], linewidth=1.8, label=f"dir {idx}")
        ax_m.legend(ncol=2, fontsize=9)
        _apply_log_if_warranted(ax_m, lambdas.ravel())
    else:
        ax_m.text(0.5, 0.5, "No data", transform=ax_m.transAxes, ha="center", va="center")

    ax_e = axes[1]
    ax_e.set_title("Effective curvature eigenvalues")
    ax_e.set_xlabel("Optimizer step")
    ax_e.set_ylabel("Eff. curv. eigenvalue")
    ax_e.grid(True, alpha=0.25)
    if eff_eig.size and eff_eig.shape[1] > 0:
        has_data = np.any(np.isfinite(eff_eig))
        if has_data:
            for idx in range(eff_eig.shape[1]):
                ax_e.plot(run.steps, eff_eig[:, idx], linewidth=1.8, label=f"dir {idx}")
            ax_e.legend(ncol=2, fontsize=9)
            _apply_log_if_warranted(ax_e, eff_eig.ravel())
        else:
            ax_e.text(0.5, 0.5, "No eff. curv. data", transform=ax_e.transAxes, ha="center", va="center")
    else:
        ax_e.text(0.5, 0.5, "No eff. curv. data", transform=ax_e.transAxes, ha="center", va="center")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_eff_cond_trajectory(
    runs: list[Run],
    output_path: Path,
) -> None:
    """Effective curvature condition number trajectory per optimizer."""
    fig, ax = _setup_axes("Effective curvature condition number", "Condition number", yscale="log")
    all_values: list[np.ndarray] = []
    for run in runs:
        val = run.effective_curvature_cond
        if not np.any(np.isfinite(val)):
            val = run.eff_cond
        if not np.any(np.isfinite(val)):
            continue
        all_values.append(val[np.isfinite(val)])
        ax.plot(run.steps, val, linewidth=2.2, label=run.label)
    if all_values:
        _set_log_ylim(ax, np.concatenate(all_values))
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No condition number data", transform=ax.transAxes, ha="center", va="center")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_actual_update_rayleigh(
    runs: list[Run],
    output_path: Path,
) -> None:
    """Actual update Rayleigh quotient (update^T C update / ||update||^2) trajectory."""
    fig, ax = _setup_axes("Actual update Rayleigh quotient", r"$\Delta^T C \Delta / \|\Delta\|^2$", yscale="log")
    all_values: list[np.ndarray] = []
    for run in runs:
        val = run.actual_update_rayleigh
        if not np.any(np.isfinite(val)):
            continue
        pos_val = np.where(val > 0, val, np.nan)
        if not np.any(np.isfinite(pos_val)):
            continue
        all_values.append(pos_val[np.isfinite(pos_val)])
        ax.plot(run.steps, pos_val, linewidth=2.2, label=run.label)
    if all_values:
        _set_log_ylim(ax, np.concatenate(all_values))
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No Rayleigh data", transform=ax.transAxes, ha="center", va="center")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_figures(runs: list[Run], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_optimizers_dir = output_dir / "all_optimizers"
    all_optimizers_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for run in runs:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", run.label).strip("_").lower()
        run_output_dir = output_dir / safe_label
        run_output_dir.mkdir(parents=True, exist_ok=True)

        top_phi_path = run_output_dir / "topk_phi.png"
        _plot_direction_family(
            run.steps,
            run.top_phi,
            f"{run.label}: top-k phi trajectories",
            "Phi",
            top_phi_path,
            yscale="symlog",
        )
        written.append(top_phi_path)

        top_nu_path = run_output_dir / "topk_nu.png"
        _plot_direction_family(
            run.steps,
            run.top_nu,
            f"{run.label}: top-k nu trajectories",
            "Nu",
            top_nu_path,
        )
        written.append(top_nu_path)

        top_dao_nu_path = run_output_dir / "DAO_topk_nu.png"
        _plot_direction_family(
            run.steps,
            run.top_dao_nu,
            f"{run.label}: DAO top-k nu trajectories",
            "Nu DAO",
            top_dao_nu_path,
        )
        written.append(top_dao_nu_path)

        top_eos_rho_path = run_output_dir / "topk_eos_rho.png"
        _plot_eos_direction_family(
            run.steps,
            run.top_eos_rho,
            f"{run.label}: top-k EOS rho trajectories",
            top_eos_rho_path,
        )
        written.append(top_eos_rho_path)

        if run.extra_phi.size:
            extra_phi_path = run_output_dir / "extra_phi.png"
            _plot_direction_family(
                run.steps,
                run.extra_phi,
                f"{run.label}: extra phi trajectories",
                "Phi",
                extra_phi_path,
                yscale="symlog",
            )
            written.append(extra_phi_path)

        if run.extra_nu.size:
            extra_nu_path = run_output_dir / "extra_nu.png"
            _plot_direction_family(
                run.steps,
                run.extra_nu,
                f"{run.label}: extra nu trajectories",
                "Nu",
                extra_nu_path,
            )
            written.append(extra_nu_path)

        if run.extra_dao_nu.size:
            extra_dao_nu_path = run_output_dir / "DAO_extra_nu.png"
            _plot_direction_family(
                run.steps,
                run.extra_dao_nu,
                f"{run.label}: DAO extra nu trajectories",
                "Nu DAO",
                extra_dao_nu_path,
            )
            written.append(extra_dao_nu_path)

        if run.extra_eos_rho.size:
            extra_eos_rho_path = run_output_dir / "extra_eos_rho.png"
            _plot_eos_direction_family(
                run.steps,
                run.extra_eos_rho,
                f"{run.label}: extra EOS rho trajectories",
                extra_eos_rho_path,
            )
            written.append(extra_eos_rho_path)

        top_eig_bar_path = run_output_dir / "final_topk_eig_magnitude.png"
        _plot_eig_bar(
            _final_abs_values(run.top_lambdas),
            f"{run.label}: final top-k eigenvalue magnitudes",
            top_eig_bar_path,
        )
        written.append(top_eig_bar_path)

        extra_eig_bar_path = run_output_dir / "final_extra_eig_magnitude.png"
        _plot_eig_bar(
            _final_abs_values(run.extra_lambdas),
            f"{run.label}: final extra eigenvalue magnitudes",
            extra_eig_bar_path,
        )
        written.append(extra_eig_bar_path)

        mean_top_eig_bar_path = run_output_dir / "mean_topk_eig_magnitude.png"
        _plot_eig_bar(
            _mean_abs_values(run.top_lambdas),
            f"{run.label}: mean top-k eigenvalue magnitudes",
            mean_top_eig_bar_path,
        )
        written.append(mean_top_eig_bar_path)

        mean_extra_eig_bar_path = run_output_dir / "mean_extra_eig_magnitude.png"
        _plot_eig_bar(
            _mean_abs_values(run.extra_lambdas),
            f"{run.label}: mean extra eigenvalue magnitudes",
            mean_extra_eig_bar_path,
        )
        written.append(mean_extra_eig_bar_path)

    top_mean_path = all_optimizers_dir / "mean_nu_topk_by_optimizer.png"
    _plot_mean_nu(
        runs,
        category="top",
        output_path=top_mean_path,
        title="Mean nu over top-k directions",
    )
    written.append(top_mean_path)

    extra_mean_path = all_optimizers_dir / "mean_nu_extra_by_optimizer.png"
    _plot_mean_nu(
        runs,
        category="extra",
        output_path=extra_mean_path,
        title="Mean nu over extra directions",
    )
    written.append(extra_mean_path)

    top_dao_mean_path = all_optimizers_dir / "mean_nu_DAO_topk_by_optimizer.png"
    _plot_mean_nu(
        runs,
        category="top",
        output_path=top_dao_mean_path,
        title="Mean DAO nu over top-k directions",
        dao=True,
    )
    written.append(top_dao_mean_path)

    extra_dao_mean_path = all_optimizers_dir / "mean_nu_DAO_extra_by_optimizer.png"
    _plot_mean_nu(
        runs,
        category="extra",
        output_path=extra_dao_mean_path,
        title="Mean DAO nu over extra directions",
        dao=True,
    )
    written.append(extra_dao_mean_path)

    mean_eos_rho_path = all_optimizers_dir / "mean_eos_rho_topk_by_optimizer.png"
    _plot_mean_eos_rho(
        runs,
        mean_eos_rho_path,
        title="Mean EOS rho over top-k directions",
        value="mean",
    )
    written.append(mean_eos_rho_path)

    weighted_eos_rho_path = all_optimizers_dir / "update_weighted_eos_rho_topk_by_optimizer.png"
    _plot_mean_eos_rho(
        runs,
        weighted_eos_rho_path,
        title="Update-energy-weighted EOS rho over top-k directions",
        value="weighted",
    )
    written.append(weighted_eos_rho_path)

    max_eos_rho_path = all_optimizers_dir / "max_eos_rho_topk_by_optimizer.png"
    _plot_mean_eos_rho(
        runs,
        max_eos_rho_path,
        title="Max EOS rho over top-k directions",
        value="max",
    )
    written.append(max_eos_rho_path)

    loss_corr_path = all_optimizers_dir / "loss_corr_energy_weighted_mean_nu_topk.png"
    _plot_loss_corr_weighted_nu(runs, loss_corr_path, dao=False)
    written.append(loss_corr_path)

    loss_interaction_path = all_optimizers_dir / "loss_corr_interaction_nu_topk.png"
    _plot_loss_corr_interaction(runs, loss_interaction_path, dao=False)
    written.append(loss_interaction_path)

    dao_loss_corr_path = all_optimizers_dir / "loss_corr_DAO_energy_weighted_mean_nu_topk.png"
    _plot_loss_corr_weighted_nu(runs, dao_loss_corr_path, dao=True)
    written.append(dao_loss_corr_path)

    dao_loss_interaction_path = all_optimizers_dir / "loss_corr_DAO_interaction_nu_topk.png"
    _plot_loss_corr_interaction(runs, dao_loss_interaction_path, dao=True)
    written.append(dao_loss_interaction_path)

    grouped_top_eig_path = all_optimizers_dir / "final_topk_eig_magnitude_by_direction.png"
    _plot_grouped_eig_bars(
        runs,
        category="top",
        output_path=grouped_top_eig_path,
        title="Final top-k eigenvalue magnitudes by optimizer",
        value_fn=_final_abs_values,
    )
    written.append(grouped_top_eig_path)

    grouped_extra_eig_path = all_optimizers_dir / "final_extra_eig_magnitude_by_direction.png"
    _plot_grouped_eig_bars(
        runs,
        category="extra",
        output_path=grouped_extra_eig_path,
        title="Final extra eigenvalue magnitudes by optimizer",
        value_fn=_final_abs_values,
    )
    written.append(grouped_extra_eig_path)

    optimizer_mean_eig_path = all_optimizers_dir / "final_mean_eig_magnitude_by_optimizer.png"
    _plot_optimizer_mean_eig_bars(
        runs,
        optimizer_mean_eig_path,
        value_fn=_final_abs_values,
        title="Final eigenvalue magnitude by optimizer",
    )
    written.append(optimizer_mean_eig_path)

    grouped_mean_top_eig_path = all_optimizers_dir / "mean_topk_eig_magnitude_by_direction.png"
    _plot_grouped_eig_bars(
        runs,
        category="top",
        output_path=grouped_mean_top_eig_path,
        title="Mean top-k eigenvalue magnitudes by optimizer",
        value_fn=_mean_abs_values,
    )
    written.append(grouped_mean_top_eig_path)

    grouped_mean_extra_eig_path = all_optimizers_dir / "mean_extra_eig_magnitude_by_direction.png"
    _plot_grouped_eig_bars(
        runs,
        category="extra",
        output_path=grouped_mean_extra_eig_path,
        title="Mean extra eigenvalue magnitudes by optimizer",
        value_fn=_mean_abs_values,
    )
    written.append(grouped_mean_extra_eig_path)

    optimizer_training_mean_eig_path = all_optimizers_dir / "training_mean_eig_magnitude_by_optimizer.png"
    _plot_optimizer_mean_eig_bars(
        runs,
        optimizer_training_mean_eig_path,
        value_fn=_mean_abs_values,
        title="Training-mean eigenvalue magnitude by optimizer",
    )
    written.append(optimizer_training_mean_eig_path)

    # Effective curvature plots (per-run side-by-side)
    runs_with_eff = [r for r in runs if r.top_eff_curv_eig.size > 0]
    if runs_with_eff:
        for run in runs_with_eff:
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", run.label).strip("_").lower()
            run_output_dir = output_dir / safe_label
            run_output_dir.mkdir(parents=True, exist_ok=True)

            topk_sb = run_output_dir / "measured_vs_eff_curv_topk.png"
            _plot_measured_vs_effective_trajectory(run, "top", topk_sb)
            written.append(topk_sb)

            if run.extra_eff_curv_eig.size:
                extra_sb = run_output_dir / "measured_vs_eff_curv_extra.png"
                _plot_measured_vs_effective_trajectory(run, "extra", extra_sb)
                written.append(extra_sb)

        # All-optimizer effective curvature trajectory
        eff_traj_top_path = all_optimizers_dir / "mean_eff_curv_eig_topk_by_optimizer.png"
        _plot_eff_curv_trajectory(
            runs_with_eff, "top", eff_traj_top_path,
            "Mean effective curvature eigenvalue over top-k directions",
        )
        written.append(eff_traj_top_path)

        eff_traj_extra_path = all_optimizers_dir / "mean_eff_curv_eig_extra_by_optimizer.png"
        _plot_eff_curv_trajectory(
            runs_with_eff, "extra", eff_traj_extra_path,
            "Mean effective curvature eigenvalue over extra directions",
        )
        written.append(eff_traj_extra_path)

        # Measured vs effective curvature bars
        meas_vs_eff_top_path = all_optimizers_dir / "measured_vs_eff_curv_topk_bars.png"
        _plot_measured_vs_effective_curvature(
            runs_with_eff, "top", meas_vs_eff_top_path,
            "Measured GGN vs effective curvature (top-k, training mean)",
        )
        written.append(meas_vs_eff_top_path)

        meas_vs_eff_extra_path = all_optimizers_dir / "measured_vs_eff_curv_extra_bars.png"
        _plot_measured_vs_effective_curvature(
            runs_with_eff, "extra", meas_vs_eff_extra_path,
            "Measured GGN vs effective curvature (extra, training mean)",
        )
        written.append(meas_vs_eff_extra_path)

        # Condition number and Rayleigh quotient
        eff_cond_path = all_optimizers_dir / "effective_curvature_cond_by_optimizer.png"
        _plot_eff_cond_trajectory(runs_with_eff, eff_cond_path)
        written.append(eff_cond_path)

        rayleigh_path = all_optimizers_dir / "actual_update_rayleigh_by_optimizer.png"
        _plot_actual_update_rayleigh(runs_with_eff, rayleigh_path)
        written.append(rayleigh_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create phi and nu plots from eigen_tracking.csv files."
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        default=DEFAULT_RUN_DIRS,
        help="Experiment folders containing eigen_tracking.csv. Defaults to the LLM comparison runs.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/eigen_tracking",
        help="Directory where PNG figures are written, relative to plainCV unless absolute.",
    )
    parser.add_argument(
        "--psi-prefix",
        default="phi",
        help="Column prefix to use as psi in the nu formula. Falls back to phi if absent.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Small denominator stabilizer in the nu formula.",
    )
    parser.add_argument(
        "--clip-nu",
        action="store_true",
        help="Clip nu to [0, 1], matching the previous spike-like diagnostic.",
    )
    parser.add_argument(
        "--bulk-eig-threshold",
        type=float,
        default=None,
        help=(
            "Move top-k directions whose |eigenvalue| is regularly below this "
            "threshold into the extra-direction plots."
        ),
    )
    parser.add_argument(
        "--bulk-eig-relative-threshold",
        type=float,
        default=None,
        help=(
            "Move top-k directions whose |eigenvalue_i| / |eigenvalue_0| is "
            "regularly below this relative threshold into the extra-direction "
            "plots. For EFISHER, 0.1 is usually a better bulk rule than an "
            "absolute eigenvalue threshold."
        ),
    )
    parser.add_argument(
        "--bulk-eig-min-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of finite checkpoints below the absolute or relative "
            "bulk threshold needed to count a top-k direction as bulk/extra."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = _repo_root() / output_dir

    run_dirs = _dedupe_paths(args.run_dirs)
    runs = [
        _load_run(
            run_dir,
            psi_prefix=args.psi_prefix,
            eps=args.eps,
            clip_nu=args.clip_nu,
            bulk_eig_threshold=args.bulk_eig_threshold,
            bulk_eig_relative_threshold=args.bulk_eig_relative_threshold,
            bulk_eig_min_fraction=args.bulk_eig_min_fraction,
        )
        for run_dir in run_dirs
    ]
    runs = _disambiguate_labels(runs)

    written = make_figures(runs, output_dir.resolve())
    print(f"Wrote {len(written)} figures to {output_dir.resolve()}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
