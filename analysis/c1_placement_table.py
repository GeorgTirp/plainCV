#!/usr/bin/env python3
"""C1 -- placement table.

For every optimizer, checkpoint and seed, score the realised block allocation w
against the gain-optimal allocation w*: the achieved gain ratio R = G(w)/G*, the
block certificate cert(w) = sum_b w_b AM_b / AM, the namesake energy share, and
the Spearman rank correlation between w and w*.

Outputs (analysis/c1_placement/):
    c1_placement_per_checkpoint.csv   one row per (optimizer, seed, checkpoint)
    c1_placement_table.csv            seed mean +/- sd per (optimizer, checkpoint)
    c1_placement.png                  R and cert vs step, per optimizer

Usage:
    python3 analysis/c1_placement_table.py [--channel proxy|measured] [--family lm]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allocation_common as ac  # noqa: E402

VALUE_COLS = ["R", "cert", "namesake_share", "spearman_w_wstar"]


def build(runs: list, channel: str, delta_frac: float,
          step_tol: float = ac.STEP_TOL_DEFAULT) -> pd.DataFrame:
    # SOAP's grid is offset by +34, so cross-arm views must key on the shared
    # checkpoint label, not on the raw step.
    step_map = ac.checkpoint_map(runs, step_tol)
    rows = []
    for run, ck in ac.iter_checkpoints(runs, channel=channel):
        stats = ac.score_allocation(ck, delta_frac=delta_frac)
        rows.append(
            {
                "family": run.family,
                "optimizer": run.optimizer,
                "seed": run.seed,
                "global_step": ck.global_step,
                "matched_step": ac.matched_step_of(step_map, ck.global_step),
                "channel": channel,
                "delta_frac": delta_frac,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def plot(per_ck: pd.DataFrame, path: str, channel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    metrics = [
        ("R", "gain ratio  R = G(w)/G*"),
        ("cert", r"block certificate  $\sum_b w_b AM_b / AM$"),
        ("namesake_share", "namesake energy share"),
    ]
    optimizers = sorted(per_ck["optimizer"].unique())
    cmap = plt.get_cmap("tab10")
    for ax, (col, title) in zip(axes, metrics):
        for i, opt in enumerate(optimizers):
            sub = per_ck[per_ck["optimizer"] == opt]
            g = sub.groupby("matched_step")[col].agg(["mean", "std"])
            ax.plot(g.index, g["mean"], label=opt, color=cmap(i % 10), lw=1.6)
            if g["std"].notna().any():
                ax.fill_between(
                    g.index,
                    g["mean"] - g["std"].fillna(0),
                    g["mean"] + g["std"].fillna(0),
                    color=cmap(i % 10),
                    alpha=0.15,
                    lw=0,
                )
        ax.set_xlabel("step")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"C1 placement  (beta channel: {channel})", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--channel", default="proxy", choices=["proxy", "measured"])
    p.add_argument("--family", default="lm", choices=["lm", "vit"])
    p.add_argument("--delta-frac", type=float, default=ac.DELTA_FRAC_DEFAULT)
    p.add_argument("--outdir", default=None)
    p.add_argument("--root", default=None, help="override the run-tree root")
    p.add_argument("--step-tol", type=float, default=ac.STEP_TOL_DEFAULT,
                   help="max step offset treated as the same checkpoint "
                        "(SOAP's grid is offset by +34)")
    args = p.parse_args()

    runs = ac.discover_runs(args.family, args.root)
    if not runs:
        print(
            f"No {args.family} runs with section1_measures.csv found under "
            f"{ac.LM_RUNS if args.family == 'lm' else ac.VIT_RUNS}.",
            file=sys.stderr,
        )
        return 1
    print(f"[C1] {len(runs)} runs: " + ", ".join(sorted(r.label for r in runs)))

    if args.channel == "measured":
        missing = [r.label for r in runs if not ac.has_columns(r, "block_gnorm2")]
        if missing:
            print(
                "[C1] measured channel needs the A1/A2 columns (block_gnorm2_*, "
                "block_cos_*), absent from: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 2

    outdir = args.outdir or ac.out_dir("c1_placement")
    per_ck = build(runs, args.channel, args.delta_frac, args.step_tol)
    suffix = "" if args.channel == "proxy" else f"_{args.channel}"
    per_ck_path = os.path.join(
        outdir, f"c1_placement_per_checkpoint{suffix}.csv"
    )
    per_ck.to_csv(per_ck_path, index=False)

    # Report how many arms each shared checkpoint actually carries: with exact
    # step matching SOAP would sit alone on its own labels.
    per_label = per_ck.groupby("matched_step")["optimizer"].nunique()
    n_arms = per_ck["optimizer"].nunique()
    full = int((per_label == n_arms).sum())
    print(f"[C1] step tolerance {args.step_tol:g}: {full}/{len(per_label)} shared "
          f"checkpoints carry all {n_arms} arms")
    offset = (per_ck["global_step"] - per_ck["matched_step"]).abs()
    if (offset > 0).any():
        moved = per_ck.loc[offset > 0, "optimizer"].unique()
        print(f"[C1] realigned by nearest-step matching: {', '.join(sorted(moved))} "
              f"(max offset {int(offset.max())} steps)")

    table = ac.seed_summary(
        per_ck, ["family", "optimizer", "matched_step"], VALUE_COLS
    )
    table_path = os.path.join(outdir, f"c1_placement_table{suffix}.csv")
    table.to_csv(table_path, index=False)

    plot(per_ck, os.path.join(outdir, f"c1_placement{suffix}.png"), args.channel)

    for window, frame in (("all checkpoints", per_ck),
                          ("late half", ac.late_half(per_ck))):
        overall = ac.seed_summary(frame, ["family", "optimizer"], VALUE_COLS)
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            print(f"\n[C1] {window}, seed mean +/- sd:\n")
            print(
                overall[
                    ["optimizer"]
                    + [f"{c}_{s}" for c in VALUE_COLS for s in ("mean", "sd")]
                ].to_string(index=False, float_format=lambda v: f"{v:.4g}")
            )
    print(f"\n[C1] wrote {per_ck_path}\n[C1] wrote {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
