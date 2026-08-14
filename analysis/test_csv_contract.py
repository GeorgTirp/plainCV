#!/usr/bin/env python3
"""Width contract test for the Section-1 and eigen-tracking CSV writers.

Every column added by the A and B patches has to be added in two places: the
header builder (``init_*_csv``) and the row builder (``append_*_row``). A
mismatch silently shifts every downstream column, so this asserts
``len(header) == len(row)`` for each on/off combination of the optional blocks.

JAX cannot start on the login node, so ``jax`` is stubbed with the two entry
points these writers actually use (``device_get`` and ``tree_util``); nothing
else in the writers touches it.

Usage:  python3 analysis/test_csv_contract.py
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _stub_jax() -> None:
    jax = types.ModuleType("jax")
    jax.device_get = lambda x: x
    jax.tree_util = types.SimpleNamespace(
        tree_leaves=lambda t, **k: list(t) if isinstance(t, (list, tuple)) else [t]
    )
    jax.numpy = types.ModuleType("jax.numpy")
    sys.modules.setdefault("jax", jax)
    sys.modules.setdefault("jax.numpy", jax.numpy)

    # utils pulls LAYER_TYPES out of optim.eigentools, whose package __init__
    # imports optax (and so real jax). Read the literal straight out of the
    # source instead, so this stub cannot drift from the real taxonomy.
    import ast

    src = open(os.path.join(REPO, "optim", "eigentools.py")).read()
    layer_types = None
    for node in ast.parse(src).body:
        targets = getattr(node, "target", None) or getattr(node, "targets", [None])[0]
        if getattr(targets, "id", None) == "LAYER_TYPES":
            layer_types = ast.literal_eval(node.value)
            break
    if layer_types is None:
        raise RuntimeError("LAYER_TYPES not found in optim/eigentools.py")

    pkg = types.ModuleType("optim")
    pkg.__path__ = []
    eig = types.ModuleType("optim.eigentools")
    eig.LAYER_TYPES = layer_types
    pkg.eigentools = eig
    sys.modules.setdefault("optim", pkg)
    sys.modules.setdefault("optim.eigentools", eig)
    return layer_types


class Cfg:
    out_dir = None
    exp_name = "csv_contract"
    over_write = True
    optim = "soap"


def _header(path: str) -> list:
    with open(path) as f:
        return next(csv.reader(f))


def _last_row(path: str) -> list:
    with open(path) as f:
        return list(csv.reader(f))[-1]


def _block_meta(n_blocks: int) -> dict:
    namesake = np.array([b % 3 == 0 for b in range(n_blocks)])
    return {
        "names": [f"layer{b}/kernel" for b in range(n_blocks)],
        "types": ["attn_qkv"] * n_blocks,
        "is_matrix": [True] * n_blocks,
        "namesake": namesake,
        "rows": np.full(n_blocks, 64),
        "cols": np.full(n_blocks, 32),
        "sizes": np.full(n_blocks, 2048),
        "starts": np.arange(n_blocks) * 2048,
        "n_blocks": n_blocks,
    }


def main() -> int:
    layer_types = _stub_jax()
    # get_exp_dir_path reads FLAGS.job_idx, so absl needs parsed flags.
    from absl import flags

    flags.FLAGS(["test_csv_contract"])
    import utils

    n_blocks, total_keep = 87, 10
    n_types = len(layer_types)
    failures = []
    tmp = tempfile.mkdtemp(prefix="csv_contract_")
    Cfg.out_dir = tmp

    # ---- Section-1 measures, A4 off and on ----
    for namesake_norms in (False, True):
        cfg = Cfg()
        cfg.exp_name = f"s1_{int(namesake_norms)}"
        path = utils.init_section1_csv(
            cfg,
            total_keep=total_keep,
            n_blocks=n_blocks,
            n_types=n_types,
            block_meta=_block_meta(n_blocks),
            namesake_norms=namesake_norms,
        )
        out = {
            "hutch_m": 32,
            "pr": np.zeros(total_keep),
            "pr_norm": np.zeros(total_keep),
            "evec_mass_type": np.zeros(total_keep * n_types),
            "type_energy_frac": np.zeros(n_types),
            "block_w": np.zeros(n_blocks),
            "block_A": np.zeros(n_blocks),
            "block_AM": np.zeros(n_blocks),
            "block_gnorm2": np.zeros(n_blocks),
            "block_cos": np.zeros(n_blocks),
        }
        norms = (
            {k: np.full(n_blocks, np.nan) for k in
             ("block_g_fro", "block_g_nuc", "block_g_nuc_k", "block_d_sigma", "block_r")}
            if namesake_norms else None
        )
        utils.append_section1_row(
            path, 100, out, total_keep=total_keep, n_blocks=n_blocks,
            n_types=n_types, namesake_norms=norms, section1_seconds=1.25,
        )
        h, r = _header(path), _last_row(path)
        tag = f"section1 (namesake_norms={namesake_norms})"
        if len(h) != len(r):
            failures.append(f"{tag}: header {len(h)} != row {len(r)}")
        print(f"  [{'ok ' if len(h) == len(r) else 'BAD'}] {tag}: "
              f"{len(h)} header cols, {len(r)} row cols")

    # ---- eigen tracking, precond criterion off and on ----
    top_k, extra, M = 5, 5, 32
    for precond in (False, True):
        cfg = Cfg()
        cfg.exp_name = f"eig_{int(precond)}"
        path = utils.init_eigen_tracking_csv(
            cfg, top_k, extra_modes=extra, measurement_momentum=True,
            precond_criterion=precond, num_precond_dirs=M if precond else 0,
            soap_perlayer_sin2=False, muon_topk_energy=False, num_2d_layers=0,
        )
        state = types.SimpleNamespace(
            step=7, rotation_diff=0.0, eff_cond=1.0,
            **{k: np.zeros(top_k) for k in
               ("eigenvalues", "ritz_residual", "g_proj", "d_proj",
                "update_energy_frac", "alpha_valid", "alpha", "phi", "eos_rho",
                "effective_curvature_eigenvalues",
                "damped_effective_curvature_eigenvalues",
                "preconditioned_dir_gain", "m_proj")},
            **{k: np.zeros(extra) for k in
               ("extra_eigenvalues", "extra_ritz_residual", "extra_g_proj",
                "extra_d_proj", "extra_update_energy_frac", "extra_alpha_valid",
                "extra_alpha", "extra_phi", "extra_eos_rho",
                "extra_effective_curvature_eigenvalues",
                "extra_damped_effective_curvature_eigenvalues",
                "extra_preconditioned_dir_gain", "extra_m_proj")},
            grad_norm=1.0, update_norm=1.0, m_norm=1.0,
            tracked_update_energy_frac=0.0, tracked_grad_energy_frac=0.0,
            tracked_update_grad_cosine=0.0, full_update_grad_cosine=0.0,
            full_update_grad_cos2=0.0, pos_update_energy_frac=0.0,
            neg_update_energy_frac=0.0, pos_grad_energy_frac=0.0,
            neg_grad_energy_frac=0.0, pos_update_grad_cosine=0.0,
            neg_update_grad_cosine=0.0, effective_curvature_cond=0.0,
            actual_update_rayleigh=0.0, topk_eos_rho_max=0.0,
            topk_eos_rho_mean=0.0, topk_eos_rho_update_weighted=0.0,
            topk_eos_rho_over_2_max=0.0,
            topk_eos_rho_over_2_update_weighted=0.0,
        )
        pm = (
            {
                "A_P_believed": 1.0, "A_Muon_block": 1.0, "crit_resid": 0.0,
                "precond_basis_sin2": 0.5, "precond_cov_phat_a": 0.0,
                "steps_since_precond_refresh": 3.0,
                "a_per_dir": np.zeros(M), "phat_per_dir": np.zeros(M),
                "lambda_hat_per_dir": np.zeros(M), "stratum_per_dir": np.zeros(M),
            }
            if precond else None
        )
        utils.append_eigen_tracking_row(
            path, state, measurement_metrics={"tracking_seconds": 1.0},
            measurement_momentum=True, precond_criterion=precond,
            num_precond_dirs=M if precond else 0, precond_metrics=pm,
        )
        h, r = _header(path), _last_row(path)
        tag = f"eigen_tracking (precond_criterion={precond})"
        if len(h) != len(r):
            failures.append(f"{tag}: header {len(h)} != row {len(r)}")
        print(f"  [{'ok ' if len(h) == len(r) else 'BAD'}] {tag}: "
              f"{len(h)} header cols, {len(r)} row cols")

        if precond:
            # Short arrays must be padded, not silently truncate the row.
            pm_short = dict(pm, a_per_dir=np.zeros(M - 4))
            utils.append_eigen_tracking_row(
                path, state, measurement_metrics={}, measurement_momentum=True,
                precond_criterion=True, num_precond_dirs=M,
                precond_metrics=pm_short,
            )
            r2 = _last_row(path)
            ok = len(r2) == len(h)
            if not ok:
                failures.append(f"{tag}: short a_per_dir gave {len(r2)} cols")
            print(f"  [{'ok ' if ok else 'BAD'}] {tag} with a short per-dir array: "
                  f"{len(r2)} row cols")

    print()
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        return 1
    print("all CSV width contracts hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
