#!/usr/bin/env python3
"""C2 -- fallback-ratio counterfactual.

INSTANTANEOUS (same-checkpoint) counterfactual: at each measured checkpoint the
energy of every fallback block is rescaled by rho^2 and the allocation is
renormalized, holding the *directions* fixed (beta_b and A_b are the measured
ones). It answers "how would this checkpoint's placement have scored under a
different namesake/fallback energy split", NOT "how would training have gone" --
the trajectory would have differed, and nothing here models that.

Reports R(rho) and cert(rho) curves, their values at the rho that matches a
namesake share of 0.5, and the namesake-only scores R_nsake / cert_nsake
obtained by renormalizing w over namesake blocks alone.

Outputs (analysis/c2_fallback_ratio/):
    c2_fallback_curves.csv    per (optimizer, seed, step, rho)
    c2_fallback_matched.csv   per (optimizer, seed, step): baseline, rho@0.5, namesake-only
    c2_fallback_ratio.png     R(rho) and cert(rho), seed- and checkpoint-averaged

Usage:
    python3 analysis/c2_fallback_ratio.py [--channel proxy|measured]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allocation_common as ac  # noqa: E402

RHO_GRID = np.geomspace(1e-2, 1e2, 41)
TARGET_SHARE = 0.5


def build(runs: list, channel: str, delta_frac: float,
          step_tol: float = ac.STEP_TOL_DEFAULT):
    # SOAP's grid is offset by +34; key cross-arm views on the shared label.
    step_map = ac.checkpoint_map(runs, step_tol)
    curves, matched = [], []
    for run, ck in ac.iter_checkpoints(runs, channel=channel):
        base = ac.score_allocation(ck, delta_frac=delta_frac)
        meta = {
            "family": run.family,
            "optimizer": run.optimizer,
            "seed": run.seed,
            "global_step": ck.global_step,
            "matched_step": ac.matched_step_of(step_map, ck.global_step),
            "channel": channel,
        }

        for rho in RHO_GRID:
            w = ac.rescale_fallback(ck, rho)
            s = ac.score_allocation(ck, w=w, delta_frac=delta_frac)
            curves.append({**meta, "rho": float(rho), **s})

        rho_half = ac.rho_for_namesake_share(ck, TARGET_SHARE)
        if np.isfinite(rho_half):
            s_half = ac.score_allocation(
                ck, w=ac.rescale_fallback(ck, rho_half), delta_frac=delta_frac
            )
        else:
            s_half = {k: float("nan") for k in base}

        sub = ac.namesake_only(ck)
        if sub is not None:
            s_ns = ac.score_allocation(sub[0], delta_frac=delta_frac)
        else:
            s_ns = {k: float("nan") for k in base}

        matched.append(
            {
                **meta,
                "R_base": base["R"],
                "cert_base": base["cert"],
                "namesake_share_base": base["namesake_share"],
                "rho_half": rho_half,
                "R_half": s_half["R"],
                "cert_half": s_half["cert"],
                "R_nsake": s_ns["R"],
                "cert_nsake": s_ns["cert"],
                "spearman_w_wstar_nsake": s_ns["spearman_w_wstar"],
                "n_namesake_blocks": int(ck.namesake.sum()),
            }
        )
    return pd.DataFrame(curves), pd.DataFrame(matched)


def plot(curves: pd.DataFrame, matched: pd.DataFrame, path: str, channel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    optimizers = sorted(curves["optimizer"].unique())
    cmap = plt.get_cmap("tab10")
    panels = [
        ("R", "gain ratio  R(rho)"),
        ("cert", r"block certificate  cert(rho)"),
        ("namesake_share", "namesake energy share(rho)"),
    ]
    for ax, (col, title) in zip(axes, panels):
        for i, opt in enumerate(optimizers):
            g = curves[curves["optimizer"] == opt].groupby("rho")[col].mean()
            ax.plot(g.index, g.values, label=opt, color=cmap(i % 10), lw=1.6)
        ax.set_xscale("log")
        ax.set_xlabel(r"fallback amplitude ratio $\rho$")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.axvline(1.0, color="k", ls=":", lw=1)
    axes[2].axhline(TARGET_SHARE, color="k", ls="--", lw=1)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        f"C2 fallback-ratio counterfactual, instantaneous (same-checkpoint)  "
        f"[beta channel: {channel}]",
        fontsize=11,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--channel", default="proxy", choices=["proxy", "measured"])
    p.add_argument("--family", default="lm", choices=["lm", "vit"])
    p.add_argument("--delta-frac", type=float, default=ac.DELTA_FRAC_DEFAULT)
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
                "[C2] measured channel needs A1/A2 columns, absent from: "
                + ", ".join(missing),
                file=sys.stderr,
            )
            return 2
    print(f"[C2] {len(runs)} runs, {len(RHO_GRID)} rho values")

    outdir = ac.out_dir("c2_fallback_ratio")
    suffix = "" if args.channel == "proxy" else f"_{args.channel}"
    curves, matched = build(runs, args.channel, args.delta_frac, args.step_tol)
    curves.to_csv(os.path.join(outdir, f"c2_fallback_curves{suffix}.csv"), index=False)
    matched.to_csv(os.path.join(outdir, f"c2_fallback_matched{suffix}.csv"), index=False)
    plot(curves, matched, os.path.join(outdir, f"c2_fallback_ratio{suffix}.png"), args.channel)

    cols = ["R_base", "R_half", "R_nsake", "cert_base", "cert_half", "cert_nsake", "rho_half"]
    for window, frame in (("all checkpoints", matched),
                          ("late half", ac.late_half(matched))):
        summary = ac.seed_summary(frame, ["family", "optimizer"], cols)
        print(f"\n[C2] {window} -- baseline vs matched namesake share 0.5 vs "
              f"namesake-only (seed mean):\n")
        show = summary[["optimizer"] + [f"{c}_mean" for c in cols]]
        show.columns = ["optimizer"] + cols
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\n[C2] wrote {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
