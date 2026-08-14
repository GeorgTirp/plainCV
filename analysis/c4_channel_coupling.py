#!/usr/bin/env python3
"""C4 -- measured-beta placement and channel coupling.

Requires the A1/A2 columns (``block_gnorm2_*``, ``block_cos_*``) that
``section1_measures`` writes once the A patch is in; runs produced before it
carry only the proxy channel and are skipped with a clear message.

Two parts:

1. Measured-beta C1/C2. ``beta_b^2 = block_gnorm2_b * block_cos_b^2`` replaces
   the proxy ``size_b * AM_b``. Run with ``--rerun-c1c2`` (default) to produce
   the ``*_measured.csv`` companions next to the proxy outputs.

2. Channel coupling. Per optimizer and checkpoint:
     Spearman(c_b, A_b)                       cosine vs block exposure
     Spearman(c_b, AM_b)                      cosine vs block mean curvature
     Spearman(block_gnorm2_b / size_b, AM_b)  gradient-curvature alignment (GCA)
                                              at block level
   A method that steers energy into blocks where it also aligns well shows a
   positive c-vs-AM coupling; GCA asks the separate question of whether gradient
   density itself tracks curvature.

Outputs (analysis/c4_channel_coupling/):
    c4_coupling_per_checkpoint.csv
    c4_coupling_table.csv
    c4_channel_coupling.png

Usage:
    python3 analysis/c4_channel_coupling.py [--root DIR] [--no-rerun-c1c2]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allocation_common as ac  # noqa: E402

COUPLING_COLS = ["rho_c_A", "rho_c_AM", "rho_gca", "mean_c", "mean_abs_c"]


def build(runs: list) -> pd.DataFrame:
    rows = []
    for run in runs:
        n = run.n_blocks
        for _, row in run.measures.iterrows():
            w = ac._block_array(row, "block_w", n)
            A = ac._block_array(row, "block_A", n)
            AM = ac._block_array(row, "block_AM", n)
            cos = ac._block_array(row, "block_cos", n)
            gn2 = ac._block_array(row, "block_gnorm2", n)
            size = run.sizes

            ok = (
                np.isfinite(w) & (w > ac.W_FLOOR)
                & np.isfinite(A) & (A > 0)
                & np.isfinite(AM) & (AM > 0)
            )
            ok_c = ok & np.isfinite(cos)
            ok_g = ok & np.isfinite(gn2) & (size > 0)
            gdens = np.where(ok_g, gn2 / np.maximum(size, 1.0), np.nan)

            rows.append(
                {
                    "family": run.family,
                    "optimizer": run.optimizer,
                    "seed": run.seed,
                    "global_step": int(row["global_step"]),
                    "rho_c_A": ac.spearman(cos[ok_c], A[ok_c]),
                    "rho_c_AM": ac.spearman(cos[ok_c], AM[ok_c]),
                    "rho_gca": ac.spearman(gdens[ok_g], AM[ok_g]),
                    "mean_c": float(np.nanmean(cos[ok_c])) if ok_c.any() else np.nan,
                    "mean_abs_c": (
                        float(np.nanmean(np.abs(cos[ok_c]))) if ok_c.any() else np.nan
                    ),
                    "n_blocks_valid": int(ok.sum()),
                }
            )
    return pd.DataFrame(rows)


def plot(per_ck: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    optimizers = sorted(per_ck["optimizer"].unique())
    cmap = plt.get_cmap("tab10")
    titles = {
        "rho_c_A": r"Spearman($c_b$, $A_b$)",
        "rho_c_AM": r"Spearman($c_b$, $AM_b$)",
        "rho_gca": r"Spearman($\|g_b\|^2/r_b$, $AM_b$)  (GCA)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, col in zip(axes, ["rho_c_A", "rho_c_AM", "rho_gca"]):
        for i, opt in enumerate(optimizers):
            g = per_ck[per_ck["optimizer"] == opt].groupby("global_step")[col].mean()
            ax.plot(g.index, g.values, marker="o", ms=3, lw=1.4,
                    color=cmap(i % 10), label=opt)
        ax.axhline(0.0, color="k", ls=":", lw=1)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("step")
        ax.set_title(titles[col], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("C4 channel coupling (block level)", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="lm", choices=["lm", "vit"])
    p.add_argument("--root", default=None, help="override the run-tree root")
    p.add_argument("--rerun-c1c2", dest="rerun", action="store_true", default=True)
    p.add_argument("--no-rerun-c1c2", dest="rerun", action="store_false")
    args = p.parse_args()

    runs = ac.discover_runs(args.family, args.root)
    if not runs:
        print(f"No {args.family} runs with section1_measures.csv.", file=sys.stderr)
        return 1

    ready = [r for r in runs if ac.has_columns(r, "block_gnorm2")
             and ac.has_columns(r, "block_cos")]
    if not ready:
        print(
            "[C4] no run carries the A1/A2 columns yet (block_gnorm2_*, "
            "block_cos_*). Re-run training with the A patch, then re-run C4.",
            file=sys.stderr,
        )
        return 3
    if len(ready) < len(runs):
        skipped = sorted(set(r.label for r in runs) - set(r.label for r in ready))
        print(f"[C4] skipping {len(skipped)} pre-A runs: " + ", ".join(skipped))
    print(f"[C4] {len(ready)} runs with the measured gradient channel")

    outdir = ac.out_dir("c4_channel_coupling")
    per_ck = build(ready)
    per_ck.to_csv(os.path.join(outdir, "c4_coupling_per_checkpoint.csv"), index=False)
    table = ac.seed_summary(per_ck, ["family", "optimizer"], COUPLING_COLS)
    table.to_csv(os.path.join(outdir, "c4_coupling_table.csv"), index=False)
    plot(per_ck, os.path.join(outdir, "c4_channel_coupling.png"))

    show = table[["optimizer"] + [f"{c}_mean" for c in COUPLING_COLS]]
    show.columns = ["optimizer"] + COUPLING_COLS
    print("\n[C4] channel coupling, seed mean:\n")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\n[C4] wrote {outdir}/")

    if args.rerun:
        import subprocess

        here = os.path.dirname(os.path.abspath(__file__))
        for script in ("c1_placement_table.py", "c2_fallback_ratio.py"):
            cmd = [sys.executable, os.path.join(here, script),
                   "--channel", "measured", "--family", args.family]
            print(f"\n[C4] $ {' '.join(cmd)}")
            subprocess.run(cmd, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
