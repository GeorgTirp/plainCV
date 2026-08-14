#!/usr/bin/env python3
"""W1-W5 -- width-sweep outputs (experiment D).

W1  direction-level normalized certificate  M*Cov(phat, a) / AM   vs width (soap, adam)
W2  block-level certificate  sum_b w_b AM_b / AM                  vs width (all arms)
W3  precond_basis_sin2                                            vs width (soap, adam)
W4  eval-loss gaps (soap-muon, adam-muon) at a matched step        vs width
W5  gain ratio R on the measured beta channel                     vs width (all arms)

Each produces one CSV and one figure under analysis/width/.

Pre-registered hypotheses (H1: W1 -> 0 with width while W2 stays flat -- a
two-scale picture; H2: W4 shrinks with width) are *reported against*, never
tuned toward. The script prints the observed trend and its seed spread and
leaves the verdict to the reader.

Acceptance: the gap being interpreted must exceed the seed sd of eval loss. The
width finals carry one seed each, so W4 borrows that sd from the 3-seed 768
paper runs and labels every number derived from it as borrowed.

Usage:
    python3 analysis/w_width_outputs.py [--root exp/width] [--which W1 W4]
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
from c1_placement_table import build as build_c1  # noqa: E402

DEFAULT_ROOT = os.path.join(ac.REPO, "exp", "width")
ARMS_WITH_PRECOND = ("soap", "adam")


def discover_width_runs(root: str) -> list:
    """Find exp/width/WIDTH_<arm>_d<width>/job_idx_<k>/ runs."""
    runs = []
    if not os.path.isdir(root):
        return runs
    for name in sorted(os.listdir(root)):
        m = re.match(r"WIDTH_(.+?)_d(\d+)$", name)
        if not m:
            continue
        arm, width = m.group(1), int(m.group(2))
        arm_dir = os.path.join(root, name)
        for job in sorted(os.listdir(arm_dir)):
            jm = re.match(r"job_idx_(\d+)$", job)
            if not jm:
                continue
            d = os.path.join(arm_dir, job)
            meas = os.path.join(d, "section1_measures.csv")
            tags = os.path.join(d, "section1_block_tags.csv")
            if not (os.path.isfile(meas) and os.path.isfile(tags)):
                continue
            job_idx = int(jm.group(1))
            run = ac.Run(
                family="lm", optimizer=arm,
                seed=ac.JOB_IDX_TO_SEED.get(job_idx, job_idx),
                job_idx=job_idx, path=d,
                measures=pd.read_csv(meas), tags=pd.read_csv(tags),
            )
            run.width = width  # type: ignore[attr-defined]
            runs.append(run)
    return runs


def _eig(run) -> pd.DataFrame:
    p = os.path.join(run.path, "eigen_tracking.csv")
    return pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()


def w1_direction_certificate(runs: list) -> pd.DataFrame:
    rows = []
    for run in runs:
        if run.optimizer not in ARMS_WITH_PRECOND:
            continue
        eig = _eig(run)
        if "precond_cov_phat_a" not in eig.columns:
            continue
        a_cols = [c for c in eig.columns if re.fullmatch(r"a_per_dir_\d+", c)]
        am = run.measures[["global_step", "AM"]]
        j = eig[["global_step", "precond_cov_phat_a", "A_P_believed",
                 "A_Muon_block"] + a_cols].merge(am, on="global_step", how="inner")
        if j.empty:
            continue
        M = np.sum(np.isfinite(j[a_cols].to_numpy(dtype=float)), axis=1)
        cov = j["precond_cov_phat_a"].to_numpy(dtype=float)
        AM = j["AM"].to_numpy(dtype=float)
        # Identity cross-check: M*Cov(phat, a) == A_P - A_Muon.
        direct = (j["A_P_believed"] - j["A_Muon_block"]).to_numpy(dtype=float)
        for i in range(len(j)):
            rows.append({
                "arm": run.optimizer, "width": run.width, "seed": run.seed,
                "global_step": int(j["global_step"].iloc[i]),
                "M": int(M[i]),
                "cert_dir": float(M[i] * cov[i] / AM[i]) if AM[i] > 0 else np.nan,
                "cert_dir_via_AP": float(direct[i] / AM[i]) if AM[i] > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def w2_block_certificate(runs: list) -> pd.DataFrame:
    rows = []
    for run in runs:
        m = run.measures
        if not {"sum_wb_AMb", "AM"} <= set(m.columns):
            continue
        cert = m["sum_wb_AMb"].to_numpy(dtype=float) / m["AM"].to_numpy(dtype=float)
        for step, c in zip(m["global_step"], cert):
            rows.append({"arm": run.optimizer, "width": run.width,
                         "seed": run.seed, "global_step": int(step),
                         "cert_block": float(c)})
    return pd.DataFrame(rows)


def w3_basis_sin2(runs: list) -> pd.DataFrame:
    rows = []
    for run in runs:
        if run.optimizer not in ARMS_WITH_PRECOND:
            continue
        eig = _eig(run)
        if "precond_basis_sin2" not in eig.columns:
            continue
        for step, v in zip(eig["global_step"], eig["precond_basis_sin2"]):
            rows.append({"arm": run.optimizer, "width": run.width,
                         "seed": run.seed, "global_step": int(step),
                         "precond_basis_sin2": float(v)})
    return pd.DataFrame(rows)


def borrowed_seed_sd() -> dict:
    """Per-arm eval-loss seed sd, borrowed from the 3-seed 768 paper runs.

    The width finals carry one seed each, so they cannot estimate their own seed
    noise. The existing ``exp/paper_runs/llm/GGN_<arm>_seeds`` runs are three
    seeds at width 768 on the same budget and schedule, so their spread is the
    best available estimate of the scale of seed noise. It is an estimate from a
    *different* set of runs at a *single* width -- every number derived from it
    is labelled 'borrowed' and should not be reported as this sweep's own error
    bar.
    """
    out: dict = {}
    for run in ac.discover_runs("lm"):
        p = os.path.join(run.path, "eigen_tracking.csv")
        if not os.path.isfile(p):
            continue
        df = pd.read_csv(p)
        if "eval_loss" not in df.columns:
            continue
        v = df[["global_step", "eval_loss"]].dropna()
        out.setdefault(run.optimizer, []).append(
            v.set_index("global_step")["eval_loss"]
        )
    sd = {}
    for arm, series in out.items():
        if len(series) < 2:
            continue
        wide = pd.concat(series, axis=1)
        # sd across seeds at each step, then the typical value over steps.
        sd[arm] = float(np.nanmedian(wide.std(axis=1, ddof=1)))
    return sd


def w4_eval_gaps(runs: list) -> tuple:
    rows = []
    for run in runs:
        eig = _eig(run)
        if "eval_loss" not in eig.columns:
            continue
        for step, v in zip(eig["global_step"], eig["eval_loss"]):
            if np.isfinite(v):
                rows.append({"arm": run.optimizer, "width": run.width,
                             "seed": run.seed, "global_step": int(step),
                             "eval_loss": float(v)})
    per = pd.DataFrame(rows)
    if per.empty:
        return per, per

    # SOAP's grid is offset by +34 (post-refresh tracking), so intersecting raw
    # global_step values across arms would leave no common step at all and drop
    # SOAP from the comparison. Cluster nearby steps onto a shared label first.
    steps = np.unique(per["global_step"].to_numpy(dtype=np.int64))
    breaks = np.where(np.diff(steps) > ac.STEP_TOL_DEFAULT)[0]
    label_of = {}
    for group in np.split(steps, breaks + 1):
        for s in group:
            label_of[int(s)] = int(group.min())
    per["matched_step"] = per["global_step"].map(label_of)

    # Matched step: the last shared checkpoint every (arm, width) reached.
    steps_per_group = per.groupby(["arm", "width"])["matched_step"].apply(set)
    common = set.intersection(*steps_per_group) if len(steps_per_group) else set()
    if not common:
        return per, pd.DataFrame()
    step = max(common)

    at = per[per["matched_step"] == step]
    stats = at.groupby(["arm", "width"])["eval_loss"].agg(["mean", "std", "count"])
    borrowed = borrowed_seed_sd()
    gaps = []
    for width in sorted(at["width"].unique()):
        if ("muon", width) not in stats.index:
            continue
        base = stats.loc[("muon", width)]
        for arm in ("soap", "adam"):
            if (arm, width) not in stats.index:
                continue
            row = stats.loc[(arm, width)]
            gap = float(row["mean"] - base["mean"])
            own_sd = float(np.nanmax([row["std"], base["std"]]))
            # With one seed per width the within-sweep sd is undefined; fall back
            # to the sd borrowed from the 3-seed 768 paper runs so the gap can
            # still be compared against a seed-noise scale.
            borrowed_sd = float(
                np.nanmax([borrowed.get(arm, np.nan), borrowed.get("muon", np.nan)])
            )
            sd = own_sd if np.isfinite(own_sd) else borrowed_sd
            gaps.append({
                "comparison": f"{arm}-muon", "width": width,
                "matched_step": step, "gap": gap,
                "seed_sd_used": sd,
                "seed_sd_source": "within-sweep" if np.isfinite(own_sd)
                                  else "borrowed (3-seed 768 paper runs)",
                "n_seeds_arm": int(row["count"]), "n_seeds_muon": int(base["count"]),
                # Acceptance: the gap being interpreted must exceed the seed
                # noise scale, else it is not resolvable at this seed count.
                "resolvable": bool(np.isfinite(sd) and abs(gap) > sd),
            })
    return per, pd.DataFrame(gaps)


def w5_measured_R(runs: list) -> pd.DataFrame:
    ready = [r for r in runs if ac.has_columns(r, "block_gnorm2")]
    if not ready:
        return pd.DataFrame()
    # Score one run at a time so each row can carry its own width; build_c1
    # itself has no notion of width.
    frames = []
    for r in ready:
        sub = build_c1([r], "measured", ac.DELTA_FRAC_DEFAULT)
        sub["width"] = r.width
        sub["arm"] = r.optimizer
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def _trend_plot(df, value, title, path, group="arm", logy=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for i, key in enumerate(sorted(df[group].unique())):
        sub = df[df[group] == key]
        g = sub.groupby("width")[value].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), marker="o",
                    capsize=3, lw=1.7, color=cmap(i % 10), label=str(key))
    ax.set_xlabel(r"$d_{\mathrm{model}}$")
    ax.set_ylabel(value)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df["width"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if logy:
        ax.set_yscale("log")
    ax.axhline(0.0, color="k", ls=":", lw=1)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _trend_note(df, value, group="arm") -> None:
    for key in sorted(df[group].unique()):
        g = df[df[group] == key].groupby("width")[value].mean().sort_index()
        if len(g) < 2:
            continue
        first, last = g.iloc[0], g.iloc[-1]
        direction = "decreasing" if last < first else "increasing"
        print(f"    {key:6s}: {g.index[0]} -> {g.index[-1]}  "
              f"{first:.4g} -> {last:.4g}  ({direction})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--which", nargs="+", default=["W1", "W2", "W3", "W4", "W5"])
    args = p.parse_args()

    runs = discover_width_runs(args.root)
    if not runs:
        print(f"No width runs under {args.root}. Launch them with "
              f"'./run_width_sweep.sh finals'.", file=sys.stderr)
        return 1
    widths = sorted({r.width for r in runs})
    arms = sorted({r.optimizer for r in runs})
    print(f"[W] {len(runs)} runs; widths {widths}; arms {arms}")

    outdir = ac.out_dir("width")
    made = 0

    if "W1" in args.which:
        df = w1_direction_certificate(runs)
        if df.empty:
            print("[W1] SKIP -- no run carries the certificate columns")
        else:
            df.to_csv(os.path.join(outdir, "w1_direction_certificate.csv"), index=False)
            _trend_plot(df, "cert_dir",
                        r"W1  direction certificate  $M\,\mathrm{Cov}(\hat p, a)/AM$",
                        os.path.join(outdir, "w1_direction_certificate.png"))
            resid = np.abs(df["cert_dir"] - df["cert_dir_via_AP"])
            print(f"[W1] identity check max |M*Cov/AM - (A_P-A_Muon)/AM| = "
                  f"{np.nanmax(resid):.3e}")
            _trend_note(df, "cert_dir")
            made += 1

    if "W2" in args.which:
        df = w2_block_certificate(runs)
        if df.empty:
            print("[W2] SKIP -- no sum_wb_AMb column")
        else:
            df.to_csv(os.path.join(outdir, "w2_block_certificate.csv"), index=False)
            _trend_plot(df, "cert_block",
                        r"W2  block certificate  $\sum_b w_b AM_b / AM$",
                        os.path.join(outdir, "w2_block_certificate.png"))
            _trend_note(df, "cert_block")
            made += 1

    if "W3" in args.which:
        df = w3_basis_sin2(runs)
        if df.empty:
            print("[W3] SKIP -- no precond_basis_sin2 column")
        else:
            df.to_csv(os.path.join(outdir, "w3_basis_sin2.csv"), index=False)
            _trend_plot(df, "precond_basis_sin2",
                        r"W3  $\sin^2$ between believed and measured bases",
                        os.path.join(outdir, "w3_basis_sin2.png"))
            _trend_note(df, "precond_basis_sin2")
            made += 1

    if "W4" in args.which:
        per, gaps = w4_eval_gaps(runs)
        if gaps.empty:
            print("[W4] SKIP -- no matched step across all (arm, width) groups")
        else:
            gaps.to_csv(os.path.join(outdir, "w4_eval_gaps.csv"), index=False)
            _trend_plot(gaps.rename(columns={"comparison": "arm"}), "gap",
                        f"W4  eval-loss gap at step {int(gaps['matched_step'].iloc[0])}",
                        os.path.join(outdir, "w4_eval_gaps.png"))
            print(f"[W4] matched step {int(gaps['matched_step'].iloc[0])}")
            print(gaps.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
            unres = gaps[~gaps["resolvable"]]
            if len(unres):
                print("[W4] ACCEPTANCE: these gaps are inside the seed spread and "
                      "must not be interpreted:")
                for _, r in unres.iterrows():
                    print(f"    {r['comparison']} at width {int(r['width'])}: "
                          f"gap {r['gap']:.4g} vs seed sd {r['seed_sd_used']:.4g} "
                          f"[{r['seed_sd_source']}]")
            else:
                print("[W4] ACCEPTANCE: every reported gap exceeds its seed sd")
            made += 1

    if "W5" in args.which:
        df = w5_measured_R(runs)
        if df.empty:
            print("[W5] SKIP -- no run carries the A1/A2 columns (measured beta)")
        else:
            df.to_csv(os.path.join(outdir, "w5_measured_R.csv"), index=False)
            _trend_plot(df, "R", "W5  gain ratio R (measured beta)",
                        os.path.join(outdir, "w5_measured_R.png"), logy=True)
            _trend_note(df, "R")
            made += 1

    print(f"\n[W] wrote {made} output pairs to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
