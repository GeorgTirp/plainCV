#!/usr/bin/env bash
set -euo pipefail
#
# Submit GGN seed-sweep jobs for all optimizers via condor_submit_bid.
#
# Launches 7 optimizers × 3 seeds = 21 HTCondor jobs, all submitted at once
# with condor_submit_bid 30. (5 existing arms + hadamard_signum + kaon.)
#
# Usage:
#   ./run_all_seeds.sh              # submit all 15 jobs
#   ./run_all_seeds.sh --dry-run    # print what would be submitted
#
# Override defaults:
#   NUM_SEEDS=5 ./run_all_seeds.sh
#   SEEDS="0 1 2" ./run_all_seeds.sh
#   BID=50 ./run_all_seeds.sh

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_DIR}"

NUM_SEEDS="${NUM_SEEDS:-3}"
ALL_SEEDS=(0 42 137 256 999)
BID="${BID:-30}"
CURV_BACKEND="ggn"

if [[ -n "${SEEDS:-}" ]]; then
    read -ra SEED_LIST <<< "${SEEDS}"
else
    SEED_LIST=("${ALL_SEEDS[@]:0:${NUM_SEEDS}}")
fi

OPTIMIZERS=(adam muon sgd signum soap hadamard_signum kaon)

CONFIG_FILES=()
for opt in "${OPTIMIZERS[@]}"; do
    CONFIG_FILES+=("config/lm_${opt}.yaml")
done

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
    esac
done

ACTUAL_NUM_SEEDS="${#SEED_LIST[@]}"
TOTAL_JOBS=$(( ${#OPTIMIZERS[@]} * ACTUAL_NUM_SEEDS ))

echo "=== GGN seed sweep ==="
echo "    Optimizers: ${OPTIMIZERS[*]}"
echo "    Seeds:      ${SEED_LIST[*]} (${ACTUAL_NUM_SEEDS} seeds)"
echo "    Backend:    ${CURV_BACKEND}"
echo "    Bid:        ${BID}"
echo "    Total jobs: ${TOTAL_JOBS}"
echo ""

mkdir -p job_outputs
mkdir -p config/seed_sweep

JOB_NUM=0

for config_path in "${CONFIG_FILES[@]}"; do
    if [[ ! -f "${config_path}" ]]; then
        echo "WARNING: ${config_path} not found, skipping."
        continue
    fi

    optim_name="$(basename "${config_path}" .yaml)"  # e.g. lm_adam

    SEED_CONFIG="config/seed_sweep/${optim_name}_seeds.yaml"

    python3 - "${config_path}" "${SEED_CONFIG}" "${SEED_LIST[*]}" <<'PY'
import sys, yaml
src, dst, seeds_str = sys.argv[1:4]
seeds = [int(s) for s in seeds_str.split()]
with open(src) as f:
    cfg = yaml.safe_load(f) or {}
cfg["seed"] = seeds
cfg["sweep_keys"] = ["seed"]
base_name = cfg.get("exp_name", "run")
if not base_name.endswith("_seeds"):
    cfg["exp_name"] = base_name + "_seeds"
cfg["over_write"] = True
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    for idx in $(seq 0 $((ACTUAL_NUM_SEEDS - 1))); do
        seed="${SEED_LIST[$idx]}"
        JOB_NUM=$((JOB_NUM + 1))

        # Short name for log files, e.g. adam_s42
        short_name="${optim_name#lm_}"  # strip lm_ prefix
        job_tag="GGN_${short_name}_s${seed}"

        SUB_FILE="config/seed_sweep/${job_tag}.sub"

        cat > "${SUB_FILE}" <<SUB
universe   = vanilla
initialdir = ${REPO_DIR}

executable = nc_train_lm.sh
arguments  = ${SEED_CONFIG} _ ${CURV_BACKEND}

environment = "JOB_IDX=${idx}"

output = job_outputs/${job_tag}.\$(Cluster).\$(Process).out
error  = job_outputs/${job_tag}.\$(Cluster).\$(Process).err
log    = job_outputs/${job_tag}.\$(Cluster).log

request_cpus   = 4
request_memory = 16384
request_gpus   = 1
min_gpu_mem_mb = 80000
requirements   = (TARGET.CUDAGlobalMemoryMb >= \$(min_gpu_mem_mb)) \\
                 && !regexp("^(g146|g193|g194|g195)\\\\.", Machine)

getenv = True

+JobBatchName = "ggn-seed-sweep"

MaxTime = 129600

periodic_hold = (JobStatus == 2) && ((CurrentTime - JobCurrentStartDate) >= \$(MaxTime))
periodic_hold_reason = "Job runtime exceeded MaxTime"

queue 1
SUB

        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "[${JOB_NUM}/${TOTAL_JOBS}] condor_submit_bid ${BID} ${SUB_FILE}  # ${short_name} seed=${seed}"
        else
            echo "[${JOB_NUM}/${TOTAL_JOBS}] Submitting ${short_name} seed=${seed} (job_idx=${idx})"
            condor_submit_bid "${BID}" "${SUB_FILE}"
        fi
    done
done

echo ""
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run — pass without --dry-run to actually submit."
else
    echo "=== All ${TOTAL_JOBS} jobs submitted with bid ${BID}. ==="
fi
