#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/lustre/home/gtirpitz/plainCV"
CONFIG_PATH="${1:?need config path like config/vit_cifar10_ggn.yaml}"
EXP_NAME="${2:-}"

cd "${REPO_DIR}"

mkdir -p job_outputs

source /etc/profile.d/modules.sh
module purge
module load cuda/12.9
module load cudnn/9.10.2

source .venv/bin/activate

export WANDB_DIR="${PWD}/wandb"

prepend_ld_path() {
  if [[ -d "$1" ]]; then
    export LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH:-}"
  fi
}

prepend_ld_path "${CUDA_HOME:-/usr/local/cuda}/lib64"
prepend_ld_path "${CUDA_HOME:-/usr/local/cuda}/extras/CUPTI/lib64"

while IFS= read -r lib_dir; do
  prepend_ld_path "${lib_dir}"
done < <(
  python3 - <<'PY'
import pathlib, site, sysconfig
roots = set(site.getsitepackages())
purelib = sysconfig.get_paths().get("purelib")
if purelib:
    roots.add(purelib)
for root in sorted(roots):
    nvidia_root = pathlib.Path(root) / "nvidia"
    if nvidia_root.is_dir():
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            print(lib_dir)
PY
)

module list

echo "HOSTNAME=$(hostname)"
echo "PWD=${PWD}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "EXP_NAME=${EXP_NAME:-<config default>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<not set>}"
nvidia-smi -L || true

EXTRA_FLAGS=()
if [[ -n "${EXP_NAME}" ]]; then
  EXTRA_FLAGS+=(--exp_name="${EXP_NAME}")
fi
# JOB_IDX (set by the seed-sweep .sub files) selects one seed from the
# sweep_keys=["seed"] list in the config; output lands in <exp>/job_idx_<JOB_IDX>.
if [[ -n "${JOB_IDX:-}" ]]; then
  EXTRA_FLAGS+=(--job_idx="${JOB_IDX}")
fi

python3 train.py --config="${CONFIG_PATH}" "${EXTRA_FLAGS[@]}"
