#!/usr/bin/env python3
"""Acceptance checks for the A (per-block logging) and B (certificate) patches.

Run this on the smoke-test output before trusting any downstream analysis.

A1  sum_b block_gnorm2_b == grad_norm^2                    (relative 1e-3)
A2  block_cos_b in [-1, 1]
A3  CH >= AM at every checkpoint                           (contraharmonic >= arithmetic)
A4  namesake norms present exactly on namesake blocks,
    ||G_b||_nuc >= ||G_b||_F, and ||G_b||_F^2 == block_gnorm2_b
B   crit_resid ~ 0 relative to (A_P - A_Muon), M directions present,
    both strata populated, and the certificate's wall-clock overhead

Usage:
    python3 analysis/check_ab_acceptance.py exp/smoke/SMOKE_soap/job_idx_0
    python3 analysis/check_ab_acceptance.py RUNDIR --total-wall-seconds 5400
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

RTOL_A1 = 1e-3
OVERHEAD_LIMIT = 0.15


class Checker:
    def __init__(self) -> None:
        self.results: list = []

    def check(self, tag: str, ok, detail: str) -> None:
        self.results.append((tag, ok, detail))
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
        print(f"  [{mark}] {tag}: {detail}")

    @property
    def failed(self) -> int:
        return sum(1 for _, ok, _ in self.results if ok is False)


def _cols(df: pd.DataFrame, prefix: str) -> list:
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    hits = [(int(pat.match(c).group(1)), c) for c in df.columns if pat.match(c)]
    return [c for _, c in sorted(hits)]


def check_a(ck: Checker, meas: pd.DataFrame, eig: pd.DataFrame, tags: pd.DataFrame) -> None:
    print("\nA -- per-block logging")

    gn2_cols = _cols(meas, "block_gnorm2")
    cos_cols = _cols(meas, "block_cos")
    if not gn2_cols:
        ck.check("A1 sum_b block_gnorm2 == grad_norm^2", None,
                 "block_gnorm2_* absent -- A patch not in this run")
        return

    # A1. grad_norm2_full is the same quantity computed in the full space from
    # the same clipped gradients, so it is the tightest available reference;
    # eigen_tracking's grad_norm is the independent cross-check.
    sum_gn2 = meas[gn2_cols].to_numpy(dtype=float).sum(axis=1)
    if "grad_norm2_full" in meas.columns:
        ref = meas["grad_norm2_full"].to_numpy(dtype=float)
        rel = np.abs(sum_gn2 - ref) / np.maximum(np.abs(ref), 1e-30)
        ck.check("A1 sum_b block_gnorm2 == grad_norm2_full",
                 bool(np.nanmax(rel) <= RTOL_A1),
                 f"max rel err {np.nanmax(rel):.3e} over {len(rel)} checkpoints "
                 f"(tol {RTOL_A1:.0e})")

    if eig is not None and "grad_norm" in eig.columns:
        j = meas[["global_step"]].assign(sum_gn2=sum_gn2).merge(
            eig[["global_step", "grad_norm"]], on="global_step", how="inner"
        )
        if len(j):
            ref = np.square(j["grad_norm"].to_numpy(dtype=float))
            rel = np.abs(j["sum_gn2"].to_numpy() - ref) / np.maximum(np.abs(ref), 1e-30)
            ck.check("A1 sum_b block_gnorm2 == grad_norm^2 (eigen_tracking)",
                     bool(np.nanmax(rel) <= RTOL_A1),
                     f"max rel err {np.nanmax(rel):.3e} over {len(j)} checkpoints")
        else:
            ck.check("A1 vs eigen_tracking grad_norm", None,
                     "no shared global_step between the two CSVs")

    # A2
    cos = meas[cos_cols].to_numpy(dtype=float)
    fin = np.isfinite(cos)
    if fin.any():
        lo, hi = float(np.min(cos[fin])), float(np.max(cos[fin]))
        ck.check("A2 block_cos in [-1, 1]", bool(lo >= -1.0 and hi <= 1.0),
                 f"range [{lo:.6f}, {hi:.6f}], {fin.sum()} finite of {cos.size}")
    else:
        ck.check("A2 block_cos in [-1, 1]", False, "no finite block_cos values")

    # A3
    if {"CH", "AM"} <= set(meas.columns):
        ch = meas["CH"].to_numpy(dtype=float)
        am = meas["AM"].to_numpy(dtype=float)
        ok = np.isfinite(ch) & np.isfinite(am)
        viol = int(np.sum(ch[ok] < am[ok]))
        ratio = ch[ok] / np.maximum(am[ok], 1e-30)
        ck.check("A3 CH >= AM", viol == 0,
                 f"{viol} violations of {ok.sum()}; CH/AM in "
                 f"[{np.min(ratio):.4g}, {np.max(ratio):.4g}]")
    else:
        ck.check("A3 CH >= AM", None, "CH/AM columns absent")

    # A4
    fro_cols = _cols(meas, "block_g_fro")
    if not fro_cols:
        ck.check("A4 namesake matrix norms", None,
                 "block_g_fro_* absent (namesake_norms_enabled off)")
        return
    nsake = tags["namesake"].to_numpy().astype(bool)
    fro = meas[fro_cols].to_numpy(dtype=float)
    nuc = meas[_cols(meas, "block_g_nuc")].to_numpy(dtype=float)
    sig = meas[_cols(meas, "block_d_sigma")].to_numpy(dtype=float)

    present = np.isfinite(fro)
    expect = np.broadcast_to(nsake, fro.shape)
    ck.check("A4 norms present exactly on namesake blocks",
             bool(np.array_equal(present, expect)),
             f"{int(present.sum())} finite entries, {int(expect.sum())} namesake slots")

    m = np.isfinite(fro) & np.isfinite(nuc)
    ck.check("A4 ||G_b||_nuc >= ||G_b||_F", bool(np.all(nuc[m] >= fro[m] * (1 - 1e-5))),
             f"min nuc/fro = {np.min(nuc[m] / np.maximum(fro[m], 1e-30)):.6f}")

    gn2 = meas[gn2_cols].to_numpy(dtype=float)
    m2 = np.isfinite(fro) & np.isfinite(gn2) & (gn2 > 0)
    rel = np.abs(np.square(fro[m2]) - gn2[m2]) / np.maximum(gn2[m2], 1e-30)
    ck.check("A4 ||G_b||_F^2 == block_gnorm2_b", bool(np.nanmax(rel) <= 1e-3),
             f"max rel err {np.nanmax(rel):.3e}")
    ck.check("A4 ||Delta_b||_sigma finite and > 0",
             bool(np.all(np.isfinite(sig[expect])) and np.all(sig[expect] > 0)),
             f"{int(np.isfinite(sig).sum())} finite spectral norms")


def check_b(ck: Checker, eig: pd.DataFrame, total_wall: float) -> None:
    print("\nB -- preconditioner certificate")
    if eig is None or "crit_resid" not in eig.columns:
        ck.check("B crit_resid ~ 0", None,
                 "crit_resid absent -- precond_criterion_enabled off for this run")
        return

    resid = eig["crit_resid"].to_numpy(dtype=float)
    ap = eig["A_P_believed"].to_numpy(dtype=float)
    am = eig["A_Muon_block"].to_numpy(dtype=float)
    ok = np.isfinite(resid) & np.isfinite(ap) & np.isfinite(am)
    if not ok.any():
        ck.check("B crit_resid ~ 0", False, "no finite crit_resid rows")
        return
    scale = np.maximum(np.abs(ap[ok] - am[ok]), 1e-30)
    rel = np.abs(resid[ok]) / scale
    ck.check("B crit_resid ~ 0 (identity check)", bool(np.max(rel) <= 1e-4),
             f"max |crit_resid| / |A_P - A_Muon| = {np.max(rel):.3e} over {ok.sum()} rows")

    a_cols = _cols(eig, "a_per_dir")
    st_cols = _cols(eig, "stratum_per_dir")
    lam_cols = _cols(eig, "lambda_hat_per_dir")
    if a_cols:
        a = eig[a_cols].to_numpy(dtype=float)
        n_dirs = int(np.max(np.sum(np.isfinite(a), axis=1)))
        ck.check("B num_precond_dirs >= 32", n_dirs >= 32,
                 f"{n_dirs} finite directions per row (of {len(a_cols)} slots)")
    if lam_cols:
        lam = eig[lam_cols].to_numpy(dtype=float)
        ck.check("B lambda_hat logged per direction",
                 bool(np.isfinite(lam).any()),
                 f"{int(np.isfinite(lam).sum())} finite lambda_hat entries")
    if st_cols:
        st = eig[st_cols].to_numpy(dtype=float)
        fin = st[np.isfinite(st)]
        n_top, n_bot = int(np.sum(fin == 0)), int(np.sum(fin == 1))
        ck.check("B stratified selection populates both strata",
                 n_top > 0 and n_bot > 0,
                 f"{n_top} top-stratum, {n_bot} bottom-stratum entries")

    if "steps_since_precond_refresh" in eig.columns:
        s = eig["steps_since_precond_refresh"].to_numpy(dtype=float)
        fin = s[np.isfinite(s)]
        ck.check("B3 steps_since_precond_refresh logged", len(fin) > 0,
                 f"{len(fin)} finite values, range "
                 f"[{fin.min() if len(fin) else float('nan'):.0f}, "
                 f"{fin.max() if len(fin) else float('nan'):.0f}]")

    # Wall-clock overhead of the certificate.
    if "precond_criterion_seconds" in eig.columns:
        cert_s = float(np.nansum(eig["precond_criterion_seconds"].to_numpy(dtype=float)))
        parts = {
            k: float(np.nansum(eig[k].to_numpy(dtype=float)))
            for k in ("tracking_seconds", "precond_criterion_seconds")
            if k in eig.columns
        }
        detail = ", ".join(f"{k}={v:.1f}s" for k, v in parts.items())
        if total_wall and total_wall > cert_s:
            ov = cert_s / (total_wall - cert_s)
            ck.check(f"B certificate overhead < {OVERHEAD_LIMIT:.0%}",
                     ov < OVERHEAD_LIMIT,
                     f"{ov:.2%} of the certificate-free run ({detail})")
        else:
            ck.check("B certificate overhead", None,
                     f"pass --total-wall-seconds to score it ({detail})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("rundir", help="run directory holding section1_measures.csv")
    p.add_argument("--total-wall-seconds", type=float, default=None,
                   help="total run wall clock, for the certificate overhead check")
    args = p.parse_args()

    d = args.rundir
    meas_p = os.path.join(d, "section1_measures.csv")
    tags_p = os.path.join(d, "section1_block_tags.csv")
    eig_p = os.path.join(d, "eigen_tracking.csv")
    if not os.path.isfile(meas_p):
        cand = glob.glob(os.path.join(d, "**", "section1_measures.csv"), recursive=True)
        print(f"No section1_measures.csv in {d}."
              + (f" Did you mean:\n  " + "\n  ".join(os.path.dirname(c) for c in cand)
                 if cand else ""), file=sys.stderr)
        return 1

    meas = pd.read_csv(meas_p)
    tags = pd.read_csv(tags_p)
    eig = pd.read_csv(eig_p) if os.path.isfile(eig_p) else None
    print(f"Checking {d}\n  {len(meas)} section-1 checkpoints, {len(tags)} blocks, "
          f"{0 if eig is None else len(eig)} eigen-tracking rows")

    ck = Checker()
    check_a(ck, meas, eig, tags)
    check_b(ck, eig, args.total_wall_seconds)

    n_fail = ck.failed
    n_skip = sum(1 for _, ok, _ in ck.results if ok is None)
    print(f"\n{len(ck.results) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
