#!/usr/bin/env python3
"""C5 -- beta recheck on the converged modes only.

Refits the gradient-alignment exponent

    log(g_proj_i^2) = a + b * log(eig_i)

over the top-k tracked modes only (default k=5: the modes Lanczos actually
converges; the `extra_*` modes are the unconverged buffer that keeps the top-k
clean, and including them mixes fitted signal with Ritz noise).

Reports the top-k fit alongside the all-modes fit on the same checkpoints so the
size of the contamination is visible, plus a Ritz-residual-filtered variant.

Outputs (analysis/c5_beta_recheck/):
    c5_beta_per_checkpoint.csv   per (optimizer, seed, checkpoint) fits
    c5_beta_table.csv            seed mean +/- sd per optimizer
    c5_beta_recheck.png          slope vs step, top-k vs all modes

Usage:
    python3 analysis/c5_beta_recheck.py [--topk 5] [--resid-max 0.1]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allocation_common as ac  # noqa: E402

LOG_EPS = 1e-30


def _ols(x: np.ndarray, y: np.ndarray) -> tuple:
    """(slope, intercept, r2, n) of y on x; NaN when under-determined."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 3 or np.ptp(x) <= 0:
        return float("nan"), float("nan"), float("nan"), n
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2), n


def _fit_row(row: pd.Series, eig_cols: list, g_cols: list, resid_cols: list,
             resid_max: float) -> tuple:
    eig = row[eig_cols].to_numpy(dtype=np.float64)
    g = row[g_cols].to_numpy(dtype=np.float64)
    keep = np.isfinite(eig) & (eig > 0) & np.isfinite(g) & (np.abs(g) > 0)
    x = np.log(np.where(keep, eig, np.nan))
    y = np.log(np.where(keep, np.square(g), np.nan) + LOG_EPS)
    unfiltered = _ols(x, y)

    if resid_cols and resid_max is not None:
        resid = row[resid_cols].to_numpy(dtype=np.float64)
        # Ritz residual is an absolute quantity; scale it by the eigenvalue so
        # the cut means "relative accuracy", not "small eigenvalue".
        rel = np.abs(resid) / np.maximum(np.abs(eig), LOG_EPS)
        conv = keep & np.isfinite(rel) & (rel <= resid_max)
        filtered = _ols(
            np.where(conv, x, np.nan), np.where(conv, y, np.nan)
        )
    else:
        filtered = (float("nan"),) * 3 + (0,)
    return unfiltered, filtered


def build(runs: list, topk: int, resid_max: float) -> pd.DataFrame:
    rows = []
    for run in runs:
        df = pd.read_csv(os.path.join(run.path, "eigen_tracking.csv"))
        n_top = len([c for c in df.columns if re.fullmatch(r"eig_\d+", c)])
        n_extra = len([c for c in df.columns if re.fullmatch(r"extra_eig_\d+", c)])
        k = min(topk, n_top)

        top_eig = [f"eig_{i}" for i in range(k)]
        top_g = [f"g_proj_{i}" for i in range(k)]
        top_r = [f"ritz_resid_{i}" for i in range(k)]
        all_eig = [f"eig_{i}" for i in range(n_top)] + [
            f"extra_eig_{i}" for i in range(n_extra)
        ]
        all_g = [f"g_proj_{i}" for i in range(n_top)] + [
            f"extra_g_proj_{i}" for i in range(n_extra)
        ]

        for _, row in df.iterrows():
            (b_top, a_top, r2_top, n_t), (b_cv, a_cv, r2_cv, n_c) = _fit_row(
                row, top_eig, top_g, top_r, resid_max
            )
            (b_all, a_all, r2_all, n_a), _ = _fit_row(row, all_eig, all_g, [], None)
            rows.append(
                {
                    "family": run.family,
                    "optimizer": run.optimizer,
                    "seed": run.seed,
                    "global_step": int(row["global_step"]),
                    "slope_topk": b_top,
                    "r2_topk": r2_top,
                    "n_topk": n_t,
                    "slope_topk_conv": b_cv,
                    "r2_topk_conv": r2_cv,
                    "n_topk_conv": n_c,
                    "slope_all": b_all,
                    "r2_all": r2_all,
                    "n_all": n_a,
                    "nu_reg_logged": float(row.get("nu_reg", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def plot(per_ck: pd.DataFrame, path: str, topk: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    optimizers = sorted(per_ck["optimizer"].unique())
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, col, title in zip(
        axes,
        ["slope_topk", "slope_all"],
        [f"top-{topk} (converged) modes", "all tracked modes"],
    ):
        for i, opt in enumerate(optimizers):
            g = per_ck[per_ck["optimizer"] == opt].groupby("global_step")[col].mean()
            ax.plot(g.index, g.values, marker="o", ms=3, lw=1.4,
                    color=cmap(i % 10), label=opt)
        ax.axhline(0.0, color="k", ls=":", lw=1)
        ax.set_xlabel("step")
        ax.set_ylabel(r"slope of $\log g_i^2$ on $\log \lambda_i$")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("C5 beta recheck", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--resid-max", type=float, default=0.1,
                   help="max relative Ritz residual for the converged variant")
    p.add_argument("--family", default="lm", choices=["lm", "vit"])
    p.add_argument("--root", default=None, help="override the run-tree root")
    args = p.parse_args()

    runs = [
        r for r in ac.discover_runs(args.family, args.root)
        if os.path.isfile(os.path.join(r.path, "eigen_tracking.csv"))
    ]
    if not runs:
        print(f"No {args.family} runs with eigen_tracking.csv.", file=sys.stderr)
        return 1
    print(f"[C5] {len(runs)} runs, top-{args.topk} modes, "
          f"relative Ritz residual <= {args.resid_max}")

    outdir = ac.out_dir("c5_beta_recheck")
    per_ck = build(runs, args.topk, args.resid_max)
    per_ck.to_csv(os.path.join(outdir, "c5_beta_per_checkpoint.csv"), index=False)

    cols = ["slope_topk", "r2_topk", "slope_topk_conv", "slope_all", "r2_all"]
    table = ac.seed_summary(per_ck, ["family", "optimizer"], cols)
    table.to_csv(os.path.join(outdir, "c5_beta_table.csv"), index=False)
    plot(per_ck, os.path.join(outdir, "c5_beta_recheck.png"), args.topk)

    show = table[["optimizer"] + [f"{c}_mean" for c in cols] + ["slope_topk_sd"]]
    show.columns = ["optimizer"] + cols + ["slope_topk_sd"]
    print("\n[C5] seed mean of per-checkpoint fits:\n")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\n[C5] wrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
