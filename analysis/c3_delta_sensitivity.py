#!/usr/bin/env python3
"""C3 -- damping sensitivity of the C1 placement table.

Recomputes C1 at delta = {1e-2, 1e-3, 1e-4} x (block-diagonal exposure
sum_b w_b A_b). delta only enters through (A_b + delta), so it matters exactly
where some block has A_b comparable to or below the damping: if the ordering of
optimizers by R survives all three, the C1 conclusion is not a damping artefact.

Outputs (analysis/c3_delta_sensitivity/):
    c3_delta_per_checkpoint.csv   one row per (optimizer, seed, checkpoint, delta_frac)
    c3_delta_table.csv            seed mean +/- sd per (optimizer, delta_frac)
    c3_delta_sensitivity.png      R and cert vs delta_frac, per optimizer

Usage:
    python3 analysis/c3_delta_sensitivity.py [--channel proxy|measured]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allocation_common as ac  # noqa: E402
from c1_placement_table import VALUE_COLS, build  # noqa: E402

DELTA_FRACS = (1e-2, 1e-3, 1e-4)


def plot(table: pd.DataFrame, path: str, channel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    optimizers = sorted(table["optimizer"].unique())
    cmap = plt.get_cmap("tab10")
    for ax, col, title in zip(
        axes, ["R", "spearman_w_wstar"], ["gain ratio R", r"Spearman($w$, $w^*$)"]
    ):
        for i, opt in enumerate(optimizers):
            sub = table[table["optimizer"] == opt].sort_values("delta_frac")
            ax.errorbar(
                sub["delta_frac"],
                sub[f"{col}_mean"],
                yerr=sub[f"{col}_sd"].fillna(0),
                marker="o",
                capsize=3,
                lw=1.6,
                color=cmap(i % 10),
                label=opt,
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\delta$ as a fraction of $\sum_b w_b A_b$")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"C3 damping sensitivity  (beta channel: {channel})", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--channel", default="proxy", choices=["proxy", "measured"])
    p.add_argument("--family", default="lm", choices=["lm", "vit"])
    p.add_argument("--root", default=None, help="override the run-tree root")
    p.add_argument("--step-tol", type=float, default=ac.STEP_TOL_DEFAULT,
                   help="max step offset treated as the same checkpoint")
    args = p.parse_args()

    runs = ac.discover_runs(args.family, args.root)
    if not runs:
        print(f"No {args.family} runs with section1_measures.csv.", file=sys.stderr)
        return 1
    if args.channel == "measured":
        missing = [r.label for r in runs if not ac.has_columns(r, "block_gnorm2")]
        if missing:
            print(
                "[C3] measured channel needs A1/A2 columns, absent from: "
                + ", ".join(missing),
                file=sys.stderr,
            )
            return 2

    outdir = ac.out_dir("c3_delta_sensitivity")
    suffix = "" if args.channel == "proxy" else f"_{args.channel}"

    per_ck = pd.concat(
        [build(runs, args.channel, d, args.step_tol) for d in DELTA_FRACS],
        ignore_index=True,
    )
    per_ck.to_csv(os.path.join(outdir, f"c3_delta_per_checkpoint{suffix}.csv"), index=False)

    late = ac.late_half(per_ck, group_cols=["optimizer", "seed", "delta_frac"])
    table = ac.seed_summary(per_ck, ["family", "optimizer", "delta_frac"], VALUE_COLS)
    table.to_csv(os.path.join(outdir, f"c3_delta_table{suffix}.csv"), index=False)
    table_late = ac.seed_summary(late, ["family", "optimizer", "delta_frac"], VALUE_COLS)
    table_late.to_csv(
        os.path.join(outdir, f"c3_delta_table_late_half{suffix}.csv"), index=False
    )
    plot(table, os.path.join(outdir, f"c3_delta_sensitivity{suffix}.png"), args.channel)

    print(f"[C3] {len(runs)} runs x {len(DELTA_FRACS)} damping levels\n")
    for window, tbl in (("all checkpoints", table), ("late half", table_late)):
        piv = tbl.pivot(index="optimizer", columns="delta_frac", values="R_mean")
        print(f"[C3] R (seed mean), {window}, by damping fraction:\n")
        print(piv.to_string(float_format=lambda v: f"{v:.4g}"))
        print()
    pivot = table.pivot(index="optimizer", columns="delta_frac", values="R_mean")

    # Does the optimizer ranking by R survive the damping sweep?
    ranks = pivot.rank(ascending=False)
    stable = bool((ranks.nunique(axis=1) == 1).all())
    print(
        f"\n[C3] optimizer ranking by R is {'STABLE' if stable else 'NOT stable'} "
        f"across delta in {DELTA_FRACS}"
    )
    if not stable:
        moved = ranks[ranks.nunique(axis=1) > 1]
        print("[C3] ranking changes for: " + ", ".join(moved.index))
    print(f"\n[C3] wrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
