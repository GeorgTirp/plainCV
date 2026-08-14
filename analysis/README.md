# Revision analyses: new measures (A, B) and the width sweep (C, D)

## What is where

| Piece | Entry point | Runs on |
|---|---|---|
| A — per-block gradient channel, `tr_C2`/`CH`, namesake norms | `optim/eigentools.py:section1_measures`, `namesake_matrix_norms` | new training runs |
| B — preconditioner certificate | `optim/eigentools.py:extract_precond_directions`, `precond_criterion_from_directions` | new training runs (soap, adam) |
| A/B acceptance checks | `analysis/check_ab_acceptance.py RUNDIR` | a finished run |
| CSV width contract | `analysis/test_csv_contract.py` | anywhere (no GPU, no JAX) |
| C1 placement table | `analysis/c1_placement_table.py` | existing CSVs |
| C2 fallback-ratio counterfactual | `analysis/c2_fallback_ratio.py` | existing CSVs |
| C3 damping sensitivity | `analysis/c3_delta_sensitivity.py` | existing CSVs |
| C4 measured-beta C1/C2 + channel coupling | `analysis/c4_channel_coupling.py` | runs with the A columns |
| C5 beta recheck on converged modes | `analysis/c5_beta_recheck.py` | existing CSVs |
| D config generation | `gen_width_configs.py` | anywhere |
| D launch (tune / collect / finals) | `run_width_sweep.sh` | cluster |
| W1–W5 width outputs | `analysis/w_width_outputs.py` | width finals |

Shared math lives in `analysis/allocation_common.py`: gain `G(w)`, optimal gain
`G*`, gain ratio `R`, optimal allocation `w*`, block certificate, the
fallback-ratio counterfactual and the namesake restriction.

## Order of operations

1. **A + B patches** — in the tree. Verify with `analysis/test_csv_contract.py`
   (passes now), then with a smoke run:
   ```
   condor_submit_bid 30 config/smoke/SMOKE_muon.sub   # A only
   condor_submit_bid 30 config/smoke/SMOKE_soap.sub   # A + B, Kronecker branch
   condor_submit_bid 30 config/smoke/SMOKE_adam.sub   # A + B, one-hot branch
   python3 analysis/check_ab_acceptance.py exp/smoke/SMOKE_soap/job_idx_0
   ```
2. **C1–C3** — done, outputs under `analysis/c1_placement/`,
   `analysis/c2_fallback_ratio/`, `analysis/c3_delta_sensitivity/`.
   `analysis/c5_beta_recheck/` too; C5 did not need the A patch.
3. **C4** — `python3 analysis/c4_channel_coupling.py` once the smoke run (or any
   re-run) carries `block_gnorm2_*`. It also re-runs C1/C2 on the measured
   channel, writing `*_measured.csv` beside the proxy outputs.
4. **D** — `./run_width_sweep.sh tune --widths 256` first (cheapest), then 768,
   then 384/512; `./run_width_sweep.sh collect`; `./run_width_sweep.sh finals`;
   `python3 analysis/w_width_outputs.py`.

## Things worth knowing before reading the numbers

**The proxy beta channel is not the measured one.** Until a run carries the A
columns, `beta_b^2 = size_b * AM_b` — a curvature-only stand-in that knows
nothing about the gradient. Every C1–C3 number currently in `analysis/` is on
that proxy. C4 produces the measured-beta versions, and they can move.

**C2 is instantaneous.** Rescaling fallback energy at a fixed checkpoint says
what that checkpoint's placement would have scored under a different split. It
does not say how training would have gone; the trajectory would have differed
and nothing here models that.

**Adam's believed eigenvalue is `nu`, not `sqrt(nu)`.** That is the pre-existing
convention in `_precond_criterion_metrics`, kept unchanged. It matters because
`phat_i ∝ 1/(nu_i + damping)` with `damping = 1e-6` while `nu` can be far
smaller — in that regime `phat` is nearly uniform and the certificate mostly
measures the damping. `lambda_hat_per_dir` is now logged per direction precisely
so this can be checked rather than assumed; look at the spread of
`lambda_hat_per_dir_*` against `precond_criterion_damping` before interpreting
Adam's W1.

**The width sweep has one seed per point.** So it cannot estimate its own seed
noise. W4 compares each gap against a seed sd borrowed from the 3-seed 768 paper
runs (`adam` 0.0037, `muon` 0.0014, `soap` 0.0016 nats) and labels it as
borrowed. Gaps below that scale are flagged unresolvable and should not be
interpreted.

**768 is not free.** The existing 768 paper runs predate A and B, so they carry
neither `block_gnorm2_*` nor the certificate columns. They can serve W4 (eval
loss only); W1, W2, W3 and W5 need the 768 finals re-run with the new configs.

**ViT has no Section-1 data.** `section1_measures` is only wired into
`train_lm.py`, so C1–C4 are LM-only. The ViT arm of C1 needs Section-1 logging
added to `train.py` first.
