#!/usr/bin/env bash
set -euo pipefail
#
# Submit GGN seed-sweep jobs for all optimizers on the ViT / tiny-imagenet run,
# the image-model counterpart of run_all_seeds.sh (which does the LLM run).
#
# Launches 7 optimizers x 3 seeds = 21 HTCondor jobs, all submitted at once
# with condor_submit_bid 30. Same 7 arms as the LLM sweep
# (adam, muon, sgd, signum, soap, hadamard_signum, kaon).
#
# Config mapping (adam is the un-suffixed base config, matching the repo layout):
#   adam            -> config/vit_tinyimagenet_ggn.yaml
#   <opt>           -> config/vit_tinyimagenet_ggn_<opt>.yaml
#
# The seed sweep works exactly like the LLM one: a per-optimizer seed config is
# generated with `seed: [<list>]` + `sweep_keys: [seed]`, and each job selects its
# seed via JOB_IDX (utils.load_config maps JOB_IDX -> --job_idx -> combos[job_idx]).
# This is entrypoint-agnostic; train.py (ViT) and train_lm.py (LLM) share load_config.
#
# NOTE: `nc_train_ggn.sh` is the cluster-side wrapper (not in the repo, same as
# `nc_train_lm.sh` for the LLM). It must exist on the cluster and forward the
# JOB_IDX environment variable to `--job_idx`, matching its nc_train_lm.sh sibling.
# If your nc_train_ggn.sh uses a different argument convention, adjust the
# `arguments` line in the generated .sub below.
#
# Usage:
#   ./run_all_seeds_vit.sh              # submit all 21 jobs
#   ./run_all_seeds_vit.sh --dry-run    # print what would be submitted
#
# Override defaults:
#   NUM_SEEDS=5 ./run_all_seeds_vit.sh
#   SEEDS="0 1 2" ./run_all_seeds_vit.sh
#   BID=50 ./run_all_seeds_vit.sh
#   GPUS=1 REQUEST_MEMORY=16384 MIN_GPU_MEM_MB=40000 ./run_all_seeds_vit.sh

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_DIR}"

NUM_SEEDS="${NUM_SEEDS:-3}"
ALL_SEEDS=(0 42 137 256 999)
BID="${BID:-30}"
CURV_BACKEND="ggn"

# ViT-small on tiny-imagenet is far lighter than the LLM run; override as needed.
GPUS="${GPUS:-1}"
REQUEST_MEMORY="${REQUEST_MEMORY:-16384}"
MIN_GPU_MEM_MB="${MIN_GPU_MEM_MB:-40000}"

if [[ -n "${SEEDS:-}" ]]; then
    read -ra SEED_LIST <<< "${SEEDS}"
else
    SEED_LIST=("${ALL_SEEDS[@]:0:${NUM_SEEDS}}")
fi

OPTIMIZERS=(adam muon sgd signum soap hadamard_signum kaon)

# Map optimizer -> config path (adam is the un-suffixed base config).
config_for_opt() {
    local opt="$1"
    if [[ "${opt}" == "adam" ]]; then
        echo "config/vit_tinyimagenet_ggn.yaml"
    else
        echo "config/vit_tinyimagenet_ggn_${opt}.yaml"
    fi
}

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
    esac
done

ACTUAL_NUM_SEEDS="${#SEED_LIST[@]}"
TOTAL_JOBS=$(( ${#OPTIMIZERS[@]} * ACTUAL_NUM_SEEDS ))

echo "=== GGN ViT (tiny-imagenet) seed sweep ==="
echo "    Optimizers: ${OPTIMIZERS[*]}"
echo "    Seeds:      ${SEED_LIST[*]} (${ACTUAL_NUM_SEEDS} seeds)"
echo "    Backend:    ${CURV_BACKEND}"
echo "    Bid:        ${BID}"
echo "    GPUs/mem:   ${GPUS} gpu, ${REQUEST_MEMORY} MB ram, >=${MIN_GPU_MEM_MB} MB gpu"
echo "    Total jobs: ${TOTAL_JOBS}"
echo ""

mkdir -p job_outputs
mkdir -p config/seed_sweep

JOB_NUM=0

for opt in "${OPTIMIZERS[@]}"; do
    config_path="$(config_for_opt "${opt}")"
    if [[ ! -f "${config_path}" ]]; then
        echo "WARNING: ${config_path} not found, skipping."
        continue
    fi

    SEED_CONFIG="config/seed_sweep/vit_tinyimagenet_${opt}_seeds.yaml"

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

        job_tag="GGN_vit_${opt}_s${seed}"

        SUB_FILE="config/seed_sweep/${job_tag}.sub"

        cat > "${SUB_FILE}" <<SUB
universe   = vanilla
initialdir = ${REPO_DIR}

executable = nc_train_ggn.sh
arguments  = ${SEED_CONFIG} _ ${CURV_BACKEND}

environment = "JOB_IDX=${idx}"

output = job_outputs/${job_tag}.\$(Cluster).\$(Process).out
error  = job_outputs/${job_tag}.\$(Cluster).\$(Process).err
log    = job_outputs/${job_tag}.\$(Cluster).log

request_cpus   = 4
request_memory = ${REQUEST_MEMORY}
request_gpus   = ${GPUS}
min_gpu_mem_mb = ${MIN_GPU_MEM_MB}
requirements   = (TARGET.CUDAGlobalMemoryMb >= \$(min_gpu_mem_mb)) \\
                 && !regexp("^(g146|g193|g194|g195)\\\\.", Machine)

getenv = True

+JobBatchName = "ggn-vit-seed-sweep"

MaxTime = 129600

periodic_hold = (JobStatus == 2) && ((CurrentTime - JobCurrentStartDate) >= \$(MaxTime))
periodic_hold_reason = "Job runtime exceeded MaxTime"

queue 1
SUB

        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "[${JOB_NUM}/${TOTAL_JOBS}] condor_submit_bid ${BID} ${SUB_FILE}  # ${opt} seed=${seed}"
        else
            echo "[${JOB_NUM}/${TOTAL_JOBS}] Submitting ${opt} seed=${seed} (job_idx=${idx})"
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
