#!/usr/bin/env python3
"""F -- the channels of the one-step reduction.

Eq. (6) factors the model's attainable one-step loss reduction into three
channels, computed per (arm, seed, checkpoint) from the raw eigen-tracking
columns:

    scale     = ||g||^2 / lambda_max      gradient scale against the curvature top
    align     = cos^2                     alignment of the update with -g
    exposure  = A / lambda_max            curvature the update actually meets
    Phi_model = scale * align / exposure  the attainable reduction

``scale`` is reported without the 1/2 of Eq. (6): a constant factor moves no log
spread and no rank correlation. The performance target is
``probe_loss_reduction``.

F fills the alignment column of Table 1 and produces the spreads the Section-3
argument rests on. It does *not* assume the answer: whether alignment's spread
and correlation are comparable to exposure's is what F1-F3 measure, and the
printed sentence is generated from the numbers.

Outputs (analysis/f_channels/):
    f_channels_per_checkpoint.csv  one row per (arch, optimizer, seed, checkpoint)
    f_channel_table.csv            F1: per-arm channel means + the log spreads
    f_channel_correlations.csv     F2: seed-wise correlations and log covariances
    f_probe_vs_val.csv             F4: probe-reduction vs eval-loss orderings

Usage:
    python3 analysis/f_channels.py [--step-tol 64] [--window both]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allocation_common as ac  # noqa: E402

CHANNELS = ["scale", "align", "exposure"]
# Reported alongside the channels so the table stands on its own.
TABLE_COLS = CHANNELS + ["Phi_model", "A", "lmax", "probe_red"]
# Spread and covariance are taken over these, in log space.
SPREAD_COLS = CHANNELS + ["A", "lmax", "Phi_model"]


def build(step_tol: float) -> tuple:
    """Per-checkpoint channel frame for both architectures, plus loader notes."""
    frames, notes = [], []
    for arch in ("lm", "vit"):
        runs = ac.discover_eigen_runs(arch)
        if not runs:
            notes.append(f"[F] no {arch} runs with eigen_tracking.csv found")
            continue
        panel, info = ac.eigen_panel(runs, tol=step_tol)
        notes.append(
            f"[F] {arch}: {info['n_runs']} runs, {info['n_rows']} rows, "
            f"{info['n_labels']} shared checkpoints, "
            f"max cluster diameter {info['max_cluster_diameter']} "
            f"(tolerance 2*{step_tol:g} = {2 * step_tol:g})"
        )
        if info["dropped_lone_arm"]:
            notes.append(
                f"[F] {arch}: dropped lone-arm checkpoint(s) "
                f"{info['dropped_lone_arm']} -- SOAP's transient first probe"
            )
        if info["dropped_seed_align"]:
            notes.append(
                f"[F] {arch}: dropped {info['dropped_seed_align']} row(s) so every "
                f"seed of an arm spans the same steps"
            )
        df = ac.eos_frame(panel)
        df["probe_red"] = ac.probe_reduction(df)
        if "probe_loss_reduction" not in panel.columns:
            notes.append(
                f"[F] {arch}: probe_loss_reduction absent, using the relative "
                f"(before-after)/before drop instead"
            )
        frames.append(df)
    if not frames:
        return pd.DataFrame(), notes
    return pd.concat(frames, ignore_index=True), notes


def arm_means(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Per-seed arm means: average the raw rows inside each (arm, seed).

    Ratios are formed per row and only then averaged -- never a ratio of already
    averaged columns.
    """
    return df.groupby(
        ["architecture", "family", "optimizer", "seed"], as_index=False
    )[cols].mean()


def f1_table(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """F1 -- channel means per arm, then the log spread across main-body arms."""
    out = []
    for window in windows:
        sub = ac.window_frame(df, window)
        sub = sub[sub["guard_ok"]]
        if sub.empty:
            continue
        for arch, per_arch in sub.groupby("architecture", sort=True):
            agg = ac.seed_summary(
                per_arch, ["family", "architecture", "optimizer"], TABLE_COLS
            )
            n_ck = (
                per_arch.groupby(
                    ["family", "architecture", "optimizer"], as_index=False
                ).agg(n_checkpoints=("matched_step", "nunique"))
            )
            agg = agg.merge(
                n_ck, on=["family", "architecture", "optimizer"], how="left"
            )
            agg = agg.rename(columns={f"{TABLE_COLS[0]}_n": "n_seeds"})
            agg = agg.drop(columns=[f"{c}_n" for c in TABLE_COLS[1:]])
            agg.insert(0, "row_kind", "optimizer")
            agg.insert(3, "window", window)
            out.append(agg)

            # The spread row: sd of log across the five main-body arm means.
            per_seed = arm_means(per_arch, SPREAD_COLS)
            main = per_seed[per_seed["family"] == "main"]
            arm = main.groupby("optimizer", as_index=False)[SPREAD_COLS].mean()
            spread = {
                "row_kind": "spread",
                "family": "main",
                "architecture": arch,
                "optimizer": "SPREAD_over_main_body",
                "window": window,
                "n_seeds": int(main["seed"].nunique()),
                "n_checkpoints": int(per_arch["matched_step"].nunique()),
            }
            for col in SPREAD_COLS:
                spread[f"sigma_log_{col}"] = ac.log_sd(arm[col])
                spread[f"max_over_min_{col}"] = ac.max_over_min(arm[col])
            out.append(pd.DataFrame([spread]))
    if not out:
        return pd.DataFrame()
    table = pd.concat(out, ignore_index=True)
    lead = ["row_kind", "family", "architecture", "optimizer", "window",
            "n_seeds", "n_checkpoints"]
    rest = [c for c in table.columns if c not in lead]
    return table[lead + rest]


def _pearson_log(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 3:
        return float("nan")
    lx, ly = np.log(x[ok]), np.log(y[ok])
    if np.std(lx) == 0 or np.std(ly) == 0:
        return float("nan")
    return float(np.corrcoef(lx, ly)[0, 1])


def _log_cov(x: np.ndarray, y: np.ndarray) -> float:
    """Population covariance of the logs (matches log_sd's convention)."""
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 2:
        return float("nan")
    lx, ly = np.log(x[ok]), np.log(y[ok])
    return float(np.mean((lx - lx.mean()) * (ly - ly.mean())))


def f2_correlations(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """F2 -- correlations against the probe reduction, computed seed by seed.

    With five main-body arms a single correlation is fragile, so every quantity
    is computed within each seed and the sd across seeds is reported next to the
    mean.
    """
    predictors = ["scale", "align", "inv_exposure", "Phi_model"]
    pairs = list(itertools.combinations(CHANNELS, 2))
    rows = []
    for window in windows:
        sub = ac.window_frame(df, window)
        sub = sub[sub["guard_ok"] & (sub["family"] == "main")]
        if sub.empty:
            continue
        for arch, per_arch in sub.groupby("architecture", sort=True):
            per_seed = arm_means(per_arch, TABLE_COLS)
            for seed, g in per_seed.groupby("seed", sort=True):
                g = g.sort_values("optimizer")
                target = g["probe_red"].to_numpy(dtype=float)
                vals = {
                    "scale": g["scale"].to_numpy(dtype=float),
                    "align": g["align"].to_numpy(dtype=float),
                    "exposure": g["exposure"].to_numpy(dtype=float),
                    "Phi_model": g["Phi_model"].to_numpy(dtype=float),
                }
                vals["inv_exposure"] = 1.0 / vals["exposure"]
                rec = {
                    "architecture": arch,
                    "window": window,
                    "scope": "seed",
                    "seed": int(seed),
                    "n_optimizers": int(len(g)),
                }
                for name in predictors:
                    rho = ac.spearman(target, vals[name])
                    rec[f"spearman_abs_{name}"] = abs(rho) if np.isfinite(rho) else np.nan
                    rec[f"spearman_{name}"] = rho
                    rec[f"pearson_log_{name}"] = _pearson_log(target, vals[name])
                for a, b in pairs:
                    rec[f"c_{a}_{b}"] = _log_cov(vals[a], vals[b])
                rows.append(rec)
    if not rows:
        return pd.DataFrame()
    seedwise = pd.DataFrame(rows)

    metric_cols = [c for c in seedwise.columns
                   if c not in ("architecture", "window", "scope", "seed",
                                "n_optimizers")]
    # sd across seeds uses the sample convention, to match ac.seed_summary.
    def _sd(v: np.ndarray) -> float:
        return float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan

    summ = []
    for (arch, window), g in seedwise.groupby(["architecture", "window"], sort=True):
        for scope, fn in (("mean_over_seeds", np.nanmean), ("sd_over_seeds", _sd)):
            rec = {"architecture": arch, "window": window, "scope": scope,
                   "seed": np.nan, "n_optimizers": int(g["n_optimizers"].max())}
            for c in metric_cols:
                rec[c] = float(fn(g[c].to_numpy(dtype=float)))
            summ.append(rec)
    return pd.concat(
        [seedwise, pd.DataFrame(summ)], ignore_index=True
    ).sort_values(["architecture", "window", "scope", "seed"]).reset_index(drop=True)


def f4_probe_vs_val(df: pd.DataFrame) -> tuple:
    """F4 -- do the probe-reduction and eval-loss orderings coincide?

    Section 2 asserts they do. A larger probe reduction should pair with a
    smaller eval loss, so perfect agreement is Spearman ``-1``; the audit counts
    the checkpoints where |rho| is not exactly 1.
    """
    sub = df[df["guard_ok"] & (df["family"] == "main")]
    rows = []
    for (arch, seed, step), g in sub.groupby(
        ["architecture", "seed", "matched_step"], sort=True
    ):
        if g["optimizer"].nunique() < 3 or "eval_loss" not in g.columns:
            continue
        g = g.sort_values("optimizer")
        rho = ac.spearman(
            g["probe_red"].to_numpy(dtype=float),
            g["eval_loss"].to_numpy(dtype=float),
        )
        rows.append({
            "architecture": arch,
            "seed": int(seed),
            "matched_step": int(step),
            "n_optimizers": int(g["optimizer"].nunique()),
            "spearman_probe_vs_eval": rho,
            "abs_is_one": bool(np.isfinite(rho) and abs(abs(rho) - 1.0) < 1e-12),
            # The orderings coincide when more probe reduction pairs with less
            # eval loss, i.e. at rho = -1 exactly.
            "orderings_coincide": bool(np.isfinite(rho) and abs(rho + 1.0) < 1e-12),
        })
    per_ck = pd.DataFrame(rows)
    if per_ck.empty:
        return per_ck, pd.DataFrame()

    def _summarise(g: pd.DataFrame, arch: str, seed: int) -> dict:
        rho = g["spearman_probe_vs_eval"]
        return {
            "architecture": arch,
            "seed": seed,          # -1 pools the seeds
            "n_checkpoints": len(g),
            "spearman_mean": float(np.nanmean(rho)),
            "spearman_sd": float(np.nanstd(rho, ddof=1)) if len(g) > 1 else np.nan,
            "spearman_min": float(np.nanmin(rho)),
            "spearman_max": float(np.nanmax(rho)),
            "n_coincide": int(g["orderings_coincide"].sum()),
            "frac_coincide": float(g["orderings_coincide"].mean()),
            "n_not_pm1": int((~g["abs_is_one"]).sum()),
            "frac_not_pm1": float((~g["abs_is_one"]).mean()),
        }

    summ = [
        _summarise(g, arch, int(seed))
        for (arch, seed), g in per_ck.groupby(["architecture", "seed"], sort=True)
    ] + [
        _summarise(g, arch, -1)
        for arch, g in per_ck.groupby("architecture", sort=True)
    ]
    return per_ck, pd.DataFrame(summ)


def _fmt(frame: pd.DataFrame) -> str:
    with pd.option_context("display.width", 260, "display.max_columns", 80):
        return frame.to_string(index=False, float_format=lambda v: f"{v:.4g}")


def spearman_quantum(n: int) -> float:
    """Smallest non-zero change in Spearman rho on ``n`` points.

    Swapping one adjacent pair of ranks moves sum d^2 by 2, so rho moves by
    ``12 / (n(n^2-1))`` -- 0.1 at n = 5. Two correlations closer together than
    this differ by less than one rank swap and cannot be ordered, however
    tightly the seeds happen to agree: with five points the seed sd is often
    exactly zero because all three seeds land on the same rank ordering, and
    that is quantization, not precision.
    """
    if n < 3:
        return float("nan")
    return 12.0 / (n * (n * n - 1))


def f3_sentence(table: pd.DataFrame, corr: pd.DataFrame,
                arch: str, window: str) -> str:
    """F3 -- one sentence per (architecture, window), read off the numbers."""
    sp = table[(table["row_kind"] == "spread")
               & (table["architecture"] == arch)
               & (table["window"] == window)]
    cr = corr[(corr["architecture"] == arch)
              & (corr["window"] == window)
              & (corr["scope"] == "mean_over_seeds")]
    cs = corr[(corr["architecture"] == arch)
              & (corr["window"] == window)
              & (corr["scope"] == "sd_over_seeds")]
    if sp.empty or cr.empty:
        return f"  {arch}/{window}: not enough data."

    s_align = float(sp["sigma_log_align"].iloc[0])
    s_exp = float(sp["sigma_log_exposure"].iloc[0])
    r_align = float(cr["spearman_abs_align"].iloc[0])
    r_exp = float(cr["spearman_abs_inv_exposure"].iloc[0])
    d_align = float(cs["spearman_abs_align"].iloc[0]) if not cs.empty else np.nan
    d_exp = float(cs["spearman_abs_inv_exposure"].iloc[0]) if not cs.empty else np.nan

    # "comparable" is a stated band, not a test: within a factor of two either way.
    ratio = s_align / s_exp if s_exp > 0 else np.nan
    if np.isfinite(ratio) and 0.5 <= ratio <= 2.0:
        spread_verdict = f"within a factor of two of (x{ratio:.2f}) exposure's"
    elif np.isfinite(ratio) and ratio < 0.5:
        spread_verdict = f"smaller than exposure's by more than 2x (x{ratio:.2f})"
    else:
        spread_verdict = f"larger than exposure's by more than 2x (x{ratio:.2f})"

    n_opt = int(cr["n_optimizers"].iloc[0])
    quantum = spearman_quantum(n_opt)
    # The seed sd alone understates the uncertainty: on n_opt points Spearman is
    # quantized, so nothing below one rank swap is resolvable.
    noise = float(np.nanmax([d_align, d_exp, quantum]))
    gap = r_align - r_exp
    if abs(gap) <= noise:
        corr_verdict = (
            f"and its |Spearman| against the probe reduction ({r_align:.2f}) sits "
            f"within the resolution ({noise:.2f}; seed sd {np.nanmax([d_align, d_exp]):.2f}, "
            f"one rank swap on {n_opt} points = {quantum:.2f}) of exposure's "
            f"({r_exp:.2f}), so on this evidence the two cannot be ordered"
        )
    else:
        direction = "more" if gap > 0 else "less"
        corr_verdict = (
            f"and it correlates {direction} strongly with the probe reduction "
            f"(|rho| {r_align:.2f} against exposure's {r_exp:.2f}, a gap of "
            f"{abs(gap):.2f} against a resolution of {noise:.2f})"
        )
    return (f"  {arch}/{window}: the alignment channel's log spread is "
            f"{spread_verdict} (sigma_align {s_align:.3g} vs sigma_exposure "
            f"{s_exp:.3g}), {corr_verdict}.")


def main() -> int:
    p = argparse.ArgumentParser(description="F -- channels of the one-step reduction")
    p.add_argument("--step-tol", type=float, default=ac.STEP_TOL_DEFAULT,
                   help="max step offset treated as the same checkpoint")
    p.add_argument("--window", default="both", choices=["all", "late", "both"])
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    windows = ["all", "late"] if args.window == "both" else [args.window]
    outdir = args.outdir or ac.out_dir("f_channels")

    df, notes = build(args.step_tol)
    for line in notes:
        print(line)
    if df.empty:
        print("[F] no eigen-tracking data found", file=sys.stderr)
        return 1

    n_excl = int((~df["guard_ok"]).sum())
    print(f"\n[F] guards (gnorm>0, unorm>0, cos>0): {n_excl} of {len(df)} rows "
          f"excluded, {int(df['ascent_step'].sum())} of them ascent steps")

    keep = ["arm", "architecture", "family", "optimizer", "seed", "global_step",
            "matched_step", "guard_ok", "scale", "align", "exposure", "Phi_model",
            "A", "lmax", "gnorm", "cos", "probe_red", "eval_loss"]
    per_ck_path = os.path.join(outdir, "f_channels_per_checkpoint.csv")
    df[[c for c in keep if c in df.columns]].to_csv(per_ck_path, index=False)

    table = f1_table(df, windows)
    table.to_csv(os.path.join(outdir, "f_channel_table.csv"), index=False)

    corr = f2_correlations(df, windows)
    corr.to_csv(os.path.join(outdir, "f_channel_correlations.csv"), index=False)

    per_ck_f4, summ_f4 = f4_probe_vs_val(df)
    if not per_ck_f4.empty:
        per_ck_f4.to_csv(os.path.join(outdir, "f_probe_vs_val.csv"), index=False)
        summ_f4.to_csv(os.path.join(outdir, "f_probe_vs_val_summary.csv"), index=False)

    # ---- printed report ---------------------------------------------------
    chan_cols = [f"{c}_{s}" for c in ["scale", "align", "exposure", "A", "lmax"]
                 for s in ("mean", "sd")]
    for arch in sorted(table["architecture"].unique()):
        for window in windows:
            sel = table[(table["architecture"] == arch)
                        & (table["window"] == window)
                        & (table["row_kind"] == "optimizer")]
            if sel.empty:
                continue
            print(f"\n[F1] {arch}, window={window} -- channel means "
                  f"(seed mean +/- sd)\n")
            print(_fmt(sel[["family", "optimizer", "n_seeds", "n_checkpoints"]
                           + chan_cols]))
            sp = table[(table["row_kind"] == "spread")
                       & (table["architecture"] == arch)
                       & (table["window"] == window)]
            if not sp.empty:
                cols = ([f"sigma_log_{c}" for c in SPREAD_COLS]
                        + [f"max_over_min_{c}" for c in SPREAD_COLS])
                print(f"\n[F1] {arch}, window={window} -- spread across the five "
                      f"main-body arms (sd of log, population)\n")
                print(_fmt(sp[[c for c in cols if c in sp.columns]]))

    print("\n[F2] correlations against the probe reduction, across the five "
          "main-body arm means\n")
    show = ["architecture", "window", "scope", "seed"] + [
        c for c in corr.columns
        if c.startswith(("spearman_abs_", "pearson_log_", "c_"))
    ]
    print(_fmt(corr[show]))

    print("\n[F3] alignment against exposure -- read off the numbers above:")
    for arch in sorted(table["architecture"].unique()):
        for window in windows:
            print(f3_sentence(table, corr, arch, window))

    if not summ_f4.empty:
        print("\n[F4] probe reduction vs eval loss, across the main-body arms")
        print("     (the orderings coincide exactly at Spearman -1)\n")
        print(_fmt(summ_f4))
        print("\n[F4] Section 2 asserts these two orderings coincide. Measured:")
        for arch, g in summ_f4[summ_f4["seed"] == -1].groupby("architecture"):
            r = g.iloc[0]
            print(f"    {arch}: they coincide at {int(r['n_coincide'])}/"
                  f"{int(r['n_checkpoints'])} checkpoints "
                  f"({100 * r['frac_coincide']:.1f}%); Spearman mean "
                  f"{r['spearman_mean']:+.3f} (sd {r['spearman_sd']:.3f}, range "
                  f"[{r['spearman_min']:+.1f}, {r['spearman_max']:+.1f}])")

    print(f"\n[F] wrote {per_ck_path} and the F1/F2/F4 tables under {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
