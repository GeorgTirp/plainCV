#!/usr/bin/env python3
"""G -- instrument note: how well converged are the tracked directions?

The eigen tracker stores ten modes per checkpoint: five "tracked" (``eig_i``,
``ritz_resid_i``) and five "extra" (``extra_eig_i``, ``extra_ritz_resid_i``).
This script audits what that window is actually worth.

    G1  the relative Ritz residual as a ladder in mode index
    G2  how much of the exposure A the ten stored modes account for
    G3  the per-direction cutoff the paper should quote, rather than assert

The residual convention, which is the whole point of the audit
---------------------------------------------------------------
``ritz_resid_i`` and ``extra_ritz_resid_i`` are logged **already relative** --
the tracker stores ``residual / (|eigenvalue| + eps)``. They are read here
exactly as logged. Nothing in this file divides a residual by an eigenvalue; a
previous hand analysis did, which made the tracked modes look an order of
magnitude better converged than they are.

Outputs (analysis/g_ritz_audit/):
    g_ritz_residuals.csv       G1: per (arm, seed, checkpoint, mode), long format
    g_ritz_late_means.csv      G1: the same averaged over the late window
    g_truncation_coverage.csv  G2: A_topk / A per arm and checkpoint
    g_cutoff.csv               G3: largest usable k at each residual threshold

Usage:
    python3 analysis/g_ritz_audit.py [--step-tol 64] [--window both]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allocation_common as ac  # noqa: E402

N_MODES = 5                       # per band; ten stored modes in total
CUTOFF_THRESHOLDS = (0.01, 0.05, 0.10)


def build(step_tol: float) -> tuple:
    """Eigen panels for both architectures, with the Section-1 A where it exists."""
    frames, notes = [], []
    for arch in ("lm", "vit"):
        runs = ac.discover_eigen_runs(arch)
        if not runs:
            notes.append(f"[G] no {arch} runs with eigen_tracking.csv found")
            continue
        panel, info = ac.eigen_panel(runs, tol=step_tol)
        notes.append(
            f"[G] {arch}: {info['n_runs']} runs, {info['n_rows']} rows, "
            f"{info['n_labels']} shared checkpoints, "
            f"max cluster diameter {info['max_cluster_diameter']} "
            f"(tolerance 2*{step_tol:g} = {2 * step_tol:g})"
        )
        if info["dropped_lone_arm"]:
            notes.append(f"[G] {arch}: dropped lone-arm checkpoint(s) "
                         f"{info['dropped_lone_arm']} -- SOAP's transient first probe")
        if info["dropped_seed_align"]:
            notes.append(f"[G] {arch}: dropped {info['dropped_seed_align']} row(s) "
                         f"so every seed of an arm spans the same steps")

        # G2's denominator. Section-1 carries the independently measured A_full on
        # LM; ViT has no Section-1 wiring, so the tracker's own Rayleigh is used.
        panel["A_reference"] = panel["actual_update_rayleigh"].astype(float)
        panel["A_source"] = "eigen_tracking:actual_update_rayleigh"
        n_joined = 0
        for run in runs:
            path = os.path.join(run.path, "section1_measures.csv")
            if not os.path.isfile(path):
                continue
            s1 = pd.read_csv(path)
            if "A_full" not in s1.columns:
                continue
            lut = dict(zip(s1["global_step"].astype(int), s1["A_full"].astype(float)))
            sel = ((panel["architecture"] == run.family)
                   & (panel["optimizer"] == run.optimizer)
                   & (panel["seed"] == run.seed))
            hit = sel & panel["global_step"].astype(int).isin(lut)
            panel.loc[hit, "A_reference"] = (
                panel.loc[hit, "global_step"].astype(int).map(lut)
            )
            panel.loc[hit, "A_source"] = "section1_measures:A_full"
            n_joined += int(hit.sum())
        if n_joined:
            notes.append(f"[G] {arch}: A taken from section1_measures.A_full on "
                         f"{n_joined}/{len(panel)} rows, from the tracker elsewhere")
        else:
            notes.append(f"[G] {arch}: no section1_measures.csv, A taken from the "
                         f"tracker's actual_update_rayleigh")
        frames.append(panel)
    if not frames:
        return pd.DataFrame(), notes
    return pd.concat(frames, ignore_index=True), notes


def g1_residuals(panel: pd.DataFrame) -> pd.DataFrame:
    """G1 -- one row per (arm, seed, checkpoint, stored mode).

    ``ritz_resid_rel`` is the logged column verbatim: it is already
    residual/eigenvalue and is NOT divided by ``eig`` here.
    """
    idx = ["arm", "architecture", "family", "optimizer", "seed",
           "global_step", "matched_step"]
    rows = []
    for band, eig_pre, res_pre, uef_pre in (
        ("tracked", "eig", "ritz_resid", "update_energy_frac"),
        ("extra", "extra_eig", "extra_ritz_resid", "extra_update_energy_frac"),
    ):
        for i in range(N_MODES):
            cols = {f"{eig_pre}_{i}": "eig",
                    f"{res_pre}_{i}": "ritz_resid_rel"}
            uef = f"{uef_pre}_{i}"
            if uef in panel.columns:
                cols[uef] = "update_energy_frac"
            missing = [c for c in cols if c not in panel.columns]
            if missing:
                continue
            sub = panel[idx + list(cols)].rename(columns=cols).copy()
            sub["band"] = band
            sub["band_index"] = i
            # A single ladder index across both bands: tracked 0-4, extra 5-9.
            sub["mode"] = i + (0 if band == "tracked" else N_MODES)
            rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(idx + ["mode"]).reset_index(drop=True)


def g1_late_means(long: pd.DataFrame, windows: list) -> pd.DataFrame:
    """G1 -- mean eig and mean relative residual per arm, mode and window."""
    out = []
    for window in windows:
        sub = ac.window_frame(long, window)
        if sub.empty:
            continue
        per_seed = sub.groupby(
            ["architecture", "family", "optimizer", "seed", "band", "band_index",
             "mode"], as_index=False
        )[["eig", "ritz_resid_rel"]].mean()
        agg = per_seed.groupby(
            ["architecture", "family", "optimizer", "band", "band_index", "mode"],
            as_index=False,
        ).agg(
            eig_mean=("eig", "mean"),
            eig_sd=("eig", "std"),
            ritz_resid_rel_mean=("ritz_resid_rel", "mean"),
            ritz_resid_rel_sd=("ritz_resid_rel", "std"),
            n_seeds=("seed", "nunique"),
        )
        agg.insert(1, "window", window)
        out.append(agg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def g2_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """G2 -- exposure carried by the stored modes: sum_i eig_i * energy_frac_i."""
    idx = ["arm", "architecture", "family", "optimizer", "seed",
           "global_step", "matched_step", "A_reference", "A_source"]
    df = panel[idx].copy()

    def band_sum(eig_pre: str, uef_pre: str) -> tuple:
        a = np.zeros(len(panel))
        r = np.zeros(len(panel))
        for i in range(N_MODES):
            e, u = f"{eig_pre}_{i}", f"{uef_pre}_{i}"
            if e not in panel.columns or u not in panel.columns:
                continue
            ev = panel[e].to_numpy(dtype=float)
            uv = panel[u].to_numpy(dtype=float)
            a += np.nan_to_num(ev * uv)
            r += np.nan_to_num(uv)
        return a, r

    a_tr, r_tr = band_sum("eig", "update_energy_frac")
    a_ex, r_ex = band_sum("extra_eig", "extra_update_energy_frac")

    A = df["A_reference"].to_numpy(dtype=float)
    df["A_topk"] = a_tr                      # i = 0..4
    df["A_top10"] = a_tr + a_ex              # all ten stored modes
    df["r_k"] = r_tr                         # tracked update energy fraction
    df["r_10"] = r_tr + r_ex
    with np.errstate(divide="ignore", invalid="ignore"):
        df["A_topk_over_A"] = np.where(A != 0, df["A_topk"] / A, np.nan)
        df["A_top10_over_A"] = np.where(A != 0, df["A_top10"] / A, np.nan)
    return df


def g3_cutoff(late: pd.DataFrame) -> pd.DataFrame:
    """G3 -- the largest k whose mean relative residual stays under a threshold.

    ``k`` is the last mode index for which every mode up to and including it has
    a late-window mean relative residual below the threshold, so the answer is a
    usable prefix of the ladder rather than an isolated well-converged mode.
    ``k = -1`` means not even mode 0 clears the threshold.
    """
    rows = []
    for (arch, fam, opt), g in late.groupby(
        ["architecture", "family", "optimizer"], sort=True
    ):
        g = g.sort_values("mode")
        res = g.set_index("mode")["ritz_resid_rel_mean"]
        for t in CUTOFF_THRESHOLDS:
            k = -1
            for m in sorted(res.index):
                if np.isfinite(res[m]) and res[m] < t:
                    k = int(m)
                else:
                    break
            rows.append({
                "architecture": arch,
                "family": fam,
                "optimizer": opt,
                "threshold": t,
                "largest_k": k,
                "n_directions": k + 1,
                "resid_at_k": float(res[k]) if k >= 0 else np.nan,
                "resid_at_k_plus_1": (
                    float(res[k + 1]) if (k + 1) in res.index else np.nan
                ),
            })
    per_arm = pd.DataFrame(rows)
    if per_arm.empty:
        return per_arm
    # The cutoff the paper can quote has to hold on every main-body arm.
    main = per_arm[per_arm["family"] == "main"]
    agg = main.groupby(["architecture", "threshold"], as_index=False).agg(
        largest_k=("largest_k", "min"),
        worst_arm=("largest_k", lambda s: s.idxmin()),
    )
    agg["optimizer"] = "MIN_over_main_body"
    agg["family"] = "main"
    agg["worst_arm"] = [main.loc[i, "optimizer"] for i in agg["worst_arm"]]
    agg["n_directions"] = agg["largest_k"] + 1
    return pd.concat([per_arm, agg], ignore_index=True)


def _fmt(frame: pd.DataFrame) -> str:
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        return frame.to_string(index=False, float_format=lambda v: f"{v:.4g}")


def main() -> int:
    p = argparse.ArgumentParser(description="G -- Ritz residual / truncation audit")
    p.add_argument("--step-tol", type=float, default=ac.STEP_TOL_DEFAULT,
                   help="max step offset treated as the same checkpoint")
    p.add_argument("--window", default="both", choices=["all", "late", "both"])
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    windows = ["all", "late"] if args.window == "both" else [args.window]
    outdir = args.outdir or ac.out_dir("g_ritz_audit")

    panel, notes = build(args.step_tol)
    for line in notes:
        print(line)
    if panel.empty:
        print("[G] no eigen-tracking data found", file=sys.stderr)
        return 1

    long = g1_residuals(panel)
    long.to_csv(os.path.join(outdir, "g_ritz_residuals.csv"), index=False)
    late_means = g1_late_means(long, windows)
    late_means.to_csv(os.path.join(outdir, "g_ritz_late_means.csv"), index=False)

    cover = g2_coverage(panel)
    cover.to_csv(os.path.join(outdir, "g_truncation_coverage.csv"), index=False)

    window_for_cutoff = "late" if "late" in windows else windows[0]
    late = late_means[late_means["window"] == window_for_cutoff]
    cutoff = g3_cutoff(late)
    cutoff.to_csv(os.path.join(outdir, "g_cutoff.csv"), index=False)

    # ---- G1: the ladder ---------------------------------------------------
    print("\n[G1] relative Ritz residual against mode index "
          f"({window_for_cutoff} window, seed mean; logged already relative, "
          "not renormalized here)\n")
    ladder = late.pivot_table(
        index=["architecture", "family", "optimizer"],
        columns="mode", values="ritz_resid_rel_mean",
    )
    ladder.columns = [f"m{int(c)}" + ("" if c < N_MODES else "(x)")
                      for c in ladder.columns]
    print(_fmt(ladder.reset_index()))
    print("\n     m0-m4 are the tracked modes, m5-m9(x) the extra ones.")
    xtra = late[late["band"] == "extra"]
    over_one = xtra[xtra["ritz_resid_rel_mean"] >= 1.0]
    print(f"     {len(over_one)}/{len(xtra)} extra-mode entries have a relative "
          f"residual >= 1, i.e. a residual larger than their own eigenvalue.")

    mono = []
    for (arch, opt), g in late.groupby(["architecture", "optimizer"]):
        v = g.sort_values("mode")["ritz_resid_rel_mean"].to_numpy(dtype=float)
        mono.append(bool(np.all(np.diff(v) >= 0)))
    print(f"     the ladder is monotone in mode index for {sum(mono)}/{len(mono)} "
          f"arms.")

    # ---- G2: coverage -----------------------------------------------------
    print("\n[G2] exposure carried by the stored modes "
          f"({window_for_cutoff} window, seed mean)\n")
    cover_w = ac.window_frame(cover, window_for_cutoff)
    per_seed = cover_w.groupby(
        ["architecture", "family", "optimizer", "seed"], as_index=False
    )[["A_topk_over_A", "A_top10_over_A", "r_k", "r_10", "A_reference"]].mean()
    g2 = per_seed.groupby(
        ["architecture", "family", "optimizer"], as_index=False
    ).agg(
        A_topk_over_A_mean=("A_topk_over_A", "mean"),
        A_topk_over_A_sd=("A_topk_over_A", "std"),
        A_top10_over_A_mean=("A_top10_over_A", "mean"),
        A_top10_over_A_sd=("A_top10_over_A", "std"),
        r_k_mean=("r_k", "mean"),
        r_10_mean=("r_10", "mean"),
        A_mean=("A_reference", "mean"),
    )
    print(_fmt(g2))
    print(f"\n     A source: "
          f"{', '.join(sorted(cover['A_source'].unique()))}")
    print(f"     The tracked window carries a median "
          f"{100 * float(np.nanmedian(g2['A_topk_over_A_mean'])):.3g}% of A "
          f"(all ten stored modes: "
          f"{100 * float(np.nanmedian(g2['A_top10_over_A_mean'])):.3g}%), on a "
          f"median update-energy fraction of "
          f"{float(np.nanmedian(g2['r_k_mean'])):.3g} (ten modes: "
          f"{float(np.nanmedian(g2['r_10_mean'])):.3g}).")

    # ---- G3: the cutoff ---------------------------------------------------
    print("\n[G3] recommended per-direction cutoff -- largest k whose late-window "
          "mean relative residual, and every mode below it, stays under the "
          "threshold\n")
    print(_fmt(cutoff.sort_values(
        ["architecture", "threshold", "family", "optimizer"]
    )[["architecture", "family", "optimizer", "threshold", "largest_k",
       "n_directions", "resid_at_k", "resid_at_k_plus_1"]]))
    print("\n[G3] the cutoff to quote (holds on every main-body arm):")
    for _, r in cutoff[cutoff["optimizer"] == "MIN_over_main_body"].sort_values(
        ["architecture", "threshold"]
    ).iterrows():
        k = int(r["largest_k"])
        if k < 0:
            print(f"    {r['architecture']}, residual < {r['threshold']:.2f}: "
                  f"no direction qualifies (worst arm {r['worst_arm']})")
        else:
            print(f"    {r['architecture']}, residual < {r['threshold']:.2f}: "
                  f"k = {k} ({k + 1} direction(s)); binding arm {r['worst_arm']}")

    print(f"\n[G] wrote the G1/G2/G3 tables under {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
