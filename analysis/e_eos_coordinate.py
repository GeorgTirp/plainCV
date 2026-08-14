#!/usr/bin/env python3
"""E -- the EoS coordinate.

The dimensionless step size that the trajectory actually takes along its own
update direction:

    gdotd   = ||g|| ||d|| cos            = -g'Delta   (> 0 on a descent step)
    eta_eff = ||d||^2 / gdotd            effective step size
    rho_dir = eta_eff * A                A = Delta'C Delta / ||Delta||^2
    rho_max = eta_eff * lambda_max       counterfactual: NOT a trajectory property

``rho_dir`` is the EoS coordinate read along the direction the optimizer chose;
``rho_max`` is what it would be if the update lay entirely in the top
eigendirection, which it does not. ``efficiency = 1 - rho_dir/2`` is the
fraction of the first-order gain the quadratic model says survives the curvature
correction.

Everything is full-space and exact -- no truncation to the tracked modes.

Read before interpreting the numbers
------------------------------------
``rho_dir < 2`` is *equivalent* to the quadratic model predicting a loss
decrease. Observing it during training that does reduce the loss is therefore
near-tautological, and this script prints no "within limits" verdict. The
content is the *distribution* of rho_dir and the spread comparison in E3: A and
eta_eff each vary over orders of magnitude across optimizers while their product
does not.

Outputs (analysis/e_eos_coordinate/):
    e_eos_per_checkpoint.csv   one row per (arch, optimizer, seed, checkpoint)
    e_eos_table.csv            E1: seed mean +/- sd per arm and window
    e_eos_trajectory.csv       E2: rho_dir and A/lmax against matched_step
    e_eos_trajectory.png       E2: the same, two rows x one panel per architecture
    e_invariance.csv           E3: spread of A, eta_eff, rho_dir across arms
    e_identity_check.csv       E4: the two algebraic identities, max rel. error
    e_threshold_audit.csv      E5: rows at or above rho_dir = 1 and = 2

Usage:
    python3 analysis/e_eos_coordinate.py [--step-tol 64] [--window both]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allocation_common as ac  # noqa: E402

# E1's metrics, in the order the printed table reads them.
TABLE_COLS = [
    "A", "A_over_lmax", "eta_eff", "rho_dir", "rho_max", "lmax", "cos", "efficiency",
]
# E3 compares the spread of these three: the first two move over orders of
# magnitude across optimizers, the third is the claim under test.
INVARIANCE_COLS = ["A", "eta_eff", "rho_dir"]


def build(step_tol: float) -> tuple:
    """The per-checkpoint EoS frame for both architectures, plus loader notes."""
    frames, notes = [], []
    for arch in ("lm", "vit"):
        runs = ac.discover_eigen_runs(arch)
        if not runs:
            notes.append(f"[E] no {arch} runs with eigen_tracking.csv found")
            continue
        panel, info = ac.eigen_panel(runs, tol=step_tol)
        notes.append(
            f"[E] {arch}: {info['n_runs']} runs, {info['n_rows']} rows, "
            f"{info['n_labels']} shared checkpoints, "
            f"max cluster diameter {info['max_cluster_diameter']} "
            f"(tolerance 2*{step_tol:g} = {2 * step_tol:g})"
        )
        if info["dropped_lone_arm"]:
            notes.append(
                f"[E] {arch}: dropped lone-arm checkpoint(s) "
                f"{info['dropped_lone_arm']} -- SOAP's transient first probe has "
                f"no comparator in any other arm"
            )
        if info["dropped_seed_align"]:
            notes.append(
                f"[E] {arch}: dropped {info['dropped_seed_align']} row(s) so every "
                f"seed of an arm spans the same steps"
            )
        frames.append(ac.eos_frame(panel))
    if not frames:
        return pd.DataFrame(), notes
    return pd.concat(frames, ignore_index=True), notes


def guard_report(df: pd.DataFrame) -> pd.DataFrame:
    """Rows excluded by the gnorm/unorm/cos guards, and how many were ascents."""
    rows = []
    for (arch, fam, opt), sub in df.groupby(
        ["architecture", "family", "optimizer"], sort=True
    ):
        n = len(sub)
        n_ok = int(sub["guard_ok"].sum())
        rows.append({
            "architecture": arch,
            "family": fam,
            "optimizer": opt,
            "n_rows": n,
            "n_excluded": n - n_ok,
            "n_ascent_steps": int(sub["ascent_step"].sum()),
            "frac_excluded": (n - n_ok) / n if n else np.nan,
        })
    return pd.DataFrame(rows)


def e1_table(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """E1 -- mean and sd over seeds, after averaging over each seed's window."""
    out = []
    for window in windows:
        sub = ac.window_frame(df, window)
        sub = sub[sub["guard_ok"]]
        if sub.empty:
            continue
        agg = ac.seed_summary(
            sub, ["family", "architecture", "optimizer"], TABLE_COLS
        )
        n_ck = (
            sub.groupby(["family", "architecture", "optimizer"], as_index=False)
            .agg(n_checkpoints=("matched_step", "nunique"))
        )
        agg = agg.merge(n_ck, on=["family", "architecture", "optimizer"], how="left")
        agg.insert(3, "window", window)
        # seed_summary emits one _n per metric; they are all the seed count.
        agg = agg.rename(columns={f"{TABLE_COLS[0]}_n": "n_seeds"})
        agg = agg.drop(columns=[f"{c}_n" for c in TABLE_COLS[1:]])
        out.append(agg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def e2_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """E2 -- seed mean and sd of rho_dir and A/lmax at every shared checkpoint."""
    sub = df[df["guard_ok"]]
    agg = sub.groupby(
        ["family", "architecture", "optimizer", "matched_step"], as_index=False
    ).agg(
        rho_dir_mean=("rho_dir", "mean"),
        rho_dir_sd=("rho_dir", "std"),
        A_over_lmax_mean=("A_over_lmax", "mean"),
        A_over_lmax_sd=("A_over_lmax", "std"),
        eta_eff_mean=("eta_eff", "mean"),
        n_seeds=("seed", "nunique"),
    )
    return agg.sort_values(
        ["architecture", "family", "optimizer", "matched_step"]
    ).reset_index(drop=True)


def e2_plot(traj: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    archs = [a for a in ("lm", "vit") if a in set(traj["architecture"])]
    if not archs:
        return
    fig, axes = plt.subplots(
        2, len(archs), figsize=(6.4 * len(archs), 8.0), squeeze=False,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("tab10")
    optimizers = sorted(traj["optimizer"].unique())
    colour = {opt: cmap(i % 10) for i, opt in enumerate(optimizers)}

    for j, arch in enumerate(archs):
        per_arch = traj[traj["architecture"] == arch]
        for row, (mean_col, sd_col, title, logy) in enumerate([
            ("rho_dir_mean", "rho_dir_sd",
             r"EoS coordinate  $\rho_{\rm dir}=\eta_{\rm eff}A$", False),
            ("A_over_lmax_mean", "A_over_lmax_sd",
             r"exposure  $A/\lambda_{\max}$", True),
        ]):
            ax = axes[row][j]
            # On a log axis mean - sd can go non-positive (ViT SOAP does, early).
            # Clip the band to half the smallest plotted mean rather than to some
            # absolute epsilon, which would drag the axis down by decades.
            pos = per_arch[mean_col]
            pos = pos[np.isfinite(pos) & (pos > 0)]
            floor = 0.5 * float(pos.min()) if logy and len(pos) else None
            for opt in optimizers:
                s = per_arch[per_arch["optimizer"] == opt].sort_values("matched_step")
                if s.empty:
                    continue
                ls = "-" if ac.optimizer_family(opt) == "main" else "--"
                ax.plot(s["matched_step"], s[mean_col], label=opt,
                        color=colour[opt], lw=1.6, ls=ls)
                sd = s[sd_col].fillna(0.0)
                lo = s[mean_col] - sd
                if floor is not None:
                    lo = lo.clip(lower=floor)
                ax.fill_between(s["matched_step"], lo, s[mean_col] + sd,
                                color=colour[opt], alpha=0.15, lw=0)
            if floor is not None:
                ax.set_ylim(bottom=floor)
            ax.set_xlabel("matched step")
            ax.set_title(f"{arch}: {title}", fontsize=10)
            ax.grid(alpha=0.3)
            if logy:
                ax.set_yscale("log")
            else:
                # The threshold has to be on screen: the point of the panel is
                # how far below 2 everything sits.
                ax.axhline(2.0, color="0.35", ls="--", lw=1.2)
                ax.annotate(r"$\rho_{\rm dir}=2$", xy=(0.99, 2.0),
                            xycoords=("axes fraction", "data"),
                            ha="right", va="bottom", fontsize=8, color="0.35")
                top = max(2.0, float(np.nanmax(per_arch["rho_dir_mean"])))
                ax.set_ylim(0.0, top * 1.15)
        axes[0][j].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "E2  EoS coordinate along the realised update direction "
        "(seed mean, +/-1 sd band, clipped at the panel floor on the log axes; "
        "dashed = control arm)",
        fontsize=11,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def e3_invariance(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """E3 -- how much each quantity varies across the main-body optimizers.

    The comparison is between arms, so it runs on the per-arm seed means; the
    pooled IQR of rho_dir is added as context on the row-level spread.
    """
    rows = []
    main = df[(df["family"] == "main") & df["guard_ok"]]
    for window in windows:
        sub = ac.window_frame(main, window)
        if sub.empty:
            continue
        for arch, per_arch in sub.groupby("architecture", sort=True):
            per_seed = per_arch.groupby(
                ["optimizer", "seed"], as_index=False
            )[INVARIANCE_COLS].mean()
            arm_mean = per_seed.groupby("optimizer", as_index=False)[
                INVARIANCE_COLS
            ].mean()
            q25, q75 = np.nanpercentile(per_arch["rho_dir"], [25, 75])
            for col in INVARIANCE_COLS:
                vals = arm_mean[col].to_numpy(dtype=float)
                rows.append({
                    "architecture": arch,
                    "window": window,
                    "quantity": col,
                    "n_optimizers": int(np.isfinite(vals).sum()),
                    "min_arm_mean": float(np.nanmin(vals)),
                    "max_arm_mean": float(np.nanmax(vals)),
                    "max_over_min": ac.max_over_min(vals),
                    "sd_log": ac.log_sd(vals),
                    "decades": (
                        float(np.log10(ac.max_over_min(vals)))
                        if np.isfinite(ac.max_over_min(vals)) else np.nan
                    ),
                    "pooled_rho_dir_q25": float(q25),
                    "pooled_rho_dir_q75": float(q75),
                    "pooled_rho_dir_iqr": float(q75 - q25),
                })
    return pd.DataFrame(rows)


def e4_identities(df: pd.DataFrame) -> pd.DataFrame:
    """E4 -- both identities per row, reported as the max relative error.

    bridge:      A / lmax          ==  rho_dir / rho_max
    two routes:  eta_eff * A       ==  A * unorm^2 / gdotd

    Both are exact algebra, so anything above float round-off means a guard or a
    unit convention is wrong.
    """
    sub = df[df["guard_ok"]].copy()
    bridge_lhs = sub["A"] / sub["lmax"]
    bridge_rhs = sub["rho_dir"] / sub["rho_max"]
    route_lhs = sub["eta_eff"] * sub["A"]
    route_rhs = sub["A"] * sub["unorm"] ** 2 / sub["gdotd"]

    def rel(lhs, rhs):
        denom = np.maximum(np.abs(lhs), np.abs(rhs))
        return np.where(denom > 0, np.abs(lhs - rhs) / denom, 0.0)

    sub["rel_err_bridge"] = rel(bridge_lhs, bridge_rhs)
    sub["rel_err_two_routes"] = rel(route_lhs, route_rhs)

    rows = []
    for (arch, opt), g in sub.groupby(["architecture", "optimizer"], sort=True):
        for name in ("bridge", "two_routes"):
            rows.append({
                "architecture": arch,
                "optimizer": opt,
                "identity": name,
                "n_rows": len(g),
                "max_abs_rel_error": float(np.nanmax(g[f"rel_err_{name}"])),
            })
    for name in ("bridge", "two_routes"):
        rows.append({
            "architecture": "ALL",
            "optimizer": "ALL",
            "identity": name,
            "n_rows": len(sub),
            "max_abs_rel_error": float(np.nanmax(sub[f"rel_err_{name}"])),
        })
    return pd.DataFrame(rows)


def e5_threshold_audit(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """E5 -- how many rows reach rho_dir = 1 and rho_dir = 2, per arm."""
    rows = []
    for window in windows:
        sub = ac.window_frame(df, window)
        for (arch, fam, opt), g in sub.groupby(
            ["architecture", "family", "optimizer"], sort=True
        ):
            ok = g[g["guard_ok"]]
            n = len(ok)
            rho = ok["rho_dir"].to_numpy(dtype=float)
            rows.append({
                "architecture": arch,
                "family": fam,
                "optimizer": opt,
                "window": window,
                "n_rows_kept": n,
                "n_excluded_by_guards": int(len(g) - n),
                "n_ascent_steps": int(g["ascent_step"].sum()),
                "n_rho_ge_2": int(np.sum(rho >= 2.0)),
                "frac_rho_ge_2": float(np.mean(rho >= 2.0)) if n else np.nan,
                "n_rho_ge_1": int(np.sum(rho >= 1.0)),
                "frac_rho_ge_1": float(np.mean(rho >= 1.0)) if n else np.nan,
                "rho_dir_max": float(np.nanmax(rho)) if n else np.nan,
                "rho_dir_median": float(np.nanmedian(rho)) if n else np.nan,
            })
    return pd.DataFrame(rows)


def _fmt(frame: pd.DataFrame) -> str:
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        return frame.to_string(index=False, float_format=lambda v: f"{v:.4g}")


def main() -> int:
    p = argparse.ArgumentParser(description="E -- the EoS coordinate")
    p.add_argument("--step-tol", type=float, default=ac.STEP_TOL_DEFAULT,
                   help="max step offset treated as the same checkpoint")
    p.add_argument("--window", default="both", choices=["all", "late", "both"])
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    windows = ["all", "late"] if args.window == "both" else [args.window]
    outdir = args.outdir or ac.out_dir("e_eos_coordinate")

    df, notes = build(args.step_tol)
    for line in notes:
        print(line)
    if df.empty:
        print("[E] no eigen-tracking data found", file=sys.stderr)
        return 1

    guards = guard_report(df)
    n_excl = int(guards["n_excluded"].sum())
    print(f"\n[E] guards (gnorm>0, unorm>0, cos>0): {n_excl} of {len(df)} rows "
          f"excluded, {int(guards['n_ascent_steps'].sum())} of them ascent steps "
          f"(cos <= 0)")
    if n_excl:
        print(_fmt(guards[guards["n_excluded"] > 0]))

    keep = [c for c in [
        "arm", "architecture", "family", "optimizer", "seed", "global_step",
        "matched_step", "guard_ok", "ascent_step", "A", "gnorm", "unorm", "cos",
        "lmax", "gdotd", "eta_eff", "rho_dir", "rho_max", "A_over_lmax",
        "efficiency", "probe_loss_reduction", "eval_loss",
    ] if c in df.columns]
    per_ck_path = os.path.join(outdir, "e_eos_per_checkpoint.csv")
    df[keep].to_csv(per_ck_path, index=False)
    guards.to_csv(os.path.join(outdir, "e_guard_report.csv"), index=False)

    table = e1_table(df, windows)
    table.to_csv(os.path.join(outdir, "e_eos_table.csv"), index=False)

    traj = e2_trajectory(df)
    traj.to_csv(os.path.join(outdir, "e_eos_trajectory.csv"), index=False)
    e2_plot(traj, os.path.join(outdir, "e_eos_trajectory.png"))

    inv = e3_invariance(df, windows)
    inv.to_csv(os.path.join(outdir, "e_invariance.csv"), index=False)

    ident = e4_identities(df)
    ident.to_csv(os.path.join(outdir, "e_identity_check.csv"), index=False)

    audit = e5_threshold_audit(df, windows)
    audit.to_csv(os.path.join(outdir, "e_threshold_audit.csv"), index=False)

    # ---- printed report ---------------------------------------------------
    show = ["family", "architecture", "optimizer", "window", "n_seeds",
            "n_checkpoints"] + [f"{c}_{s}" for c in TABLE_COLS for s in ("mean", "sd")]
    for arch in sorted(table["architecture"].unique()):
        for window in windows:
            sel = table[(table["architecture"] == arch) & (table["window"] == window)]
            if sel.empty:
                continue
            print(f"\n[E1] {arch}, window={window} -- seed mean +/- sd\n")
            print(_fmt(sel[[c for c in show if c in sel.columns]]))

    print("\n[E3] spread across the five main-body optimizers "
          "(arm means; max/min and sd of log)\n")
    print(_fmt(inv))

    print("\n[E4] identity check -- max absolute relative error\n")
    print(_fmt(ident))
    worst = float(ident["max_abs_rel_error"].max())
    tol = 1e-9
    print(f"\n[E4] worst identity error {worst:.3g} against the 1e-9 acceptance "
          f"bound: {'within round-off' if worst < tol else 'FAILS -- a guard or a unit convention is wrong'}")

    print("\n[E5] threshold audit -- rows at or above rho_dir = 1 and = 2\n")
    print(_fmt(audit))

    print("\n[E] Reading these numbers")
    print("  rho_dir < 2 is *equivalent* to the quadratic model predicting a loss")
    print("  decrease, so observing it along a trajectory that does reduce the loss")
    print("  is near-tautological and is not evidence for anything. No PASS verdict")
    print("  is issued here. What the run does say is the spread comparison:")
    for arch in sorted(inv["architecture"].unique()):
        for window in windows:
            sel = inv[(inv["architecture"] == arch) & (inv["window"] == window)]
            if sel.empty:
                continue
            bits = []
            for col in INVARIANCE_COLS:
                r = sel[sel["quantity"] == col]
                if r.empty:
                    continue
                bits.append(f"{col} x{float(r['max_over_min'].iloc[0]):.3g} "
                            f"({float(r['decades'].iloc[0]):.2f} decades)")
            rho_row = sel[sel["quantity"] == "rho_dir"]
            iqr = (f", pooled rho_dir IQR "
                   f"[{float(rho_row['pooled_rho_dir_q25'].iloc[0]):.3g}, "
                   f"{float(rho_row['pooled_rho_dir_q75'].iloc[0]):.3g}]"
                   if not rho_row.empty else "")
            print(f"    {arch}/{window}: " + "; ".join(bits) + iqr)

    print(f"\n[E] wrote {per_ck_path} and 5 tables + 1 figure under {outdir}")
    return 0 if worst < tol else 3


if __name__ == "__main__":
    raise SystemExit(main())
