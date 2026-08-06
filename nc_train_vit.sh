#!/usr/bin/env bash
set -euo pipefail
#
# Cluster wrapper for the ViT (image) experiments -> train.py
#
# This is the IMAGE counterpart of nc_train_lm.sh. The environment preamble below
# (modules, venv activation, LD_LIBRARY_PATH incl. CUPTI + nvidia wheel dirs) is
# deliberately kept IDENTICAL to nc_train_lm.sh -- that setup is what makes JAX
# find the GPU. Only the final entrypoint differs:
#
#     train.py      <- ViT / ResNet / MLP  (this wrapper)
#     train_lm.py   <- transformer / pythia LM  (nc_train_lm.sh)
#
# Only the MEASUREMENT protocol is shared (optim/eigentools.py + the eigen-tracking
# CSV writers in utils.py). Never point a ViT config at train_lm.py: it hard-fails
# with "LM training expects model='transformer' or 'pythia*'".
#
# Usage (HTCondor):
#   executable  = nc_train_vit.sh
#   arguments   = <config.yaml> [exp_name]
#   environment = "JOB_IDX=<idx>"
#
# "_" is accepted as the empty placeholder for exp_name (HTCondor rejects "").

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="${1:?need config path like config/vit_tinyimagenet_ggn.yaml}"
EXP_NAME="${2:-}"
[[ "${EXP_NAME}" == "_" ]] && EXP_NAME=""

cd "${REPO_DIR}"
mkdir -p job_outputs

# --- Environment: mirrors nc_train_lm.sh -------------------------------------
source /etc/profile.d/modules.sh
module purge
module load cuda/12.9
module load cudnn/9.10.2
if [[ -n "${NCCL_MODULE:-}" ]]; then
  module load "${NCCL_MODULE}"
fi

source .venv/bin/activate

export WANDB_DIR="${PWD}/wandb"

# Keep CUDA runtime/tooling libs visible to JAX. Without the CUPTI dir, jaxlib
# fails with "Unable to load cuPTI" and silently falls back to CPU. Also add the
# NVIDIA wheel library dirs, since the venv is installed with jax[cuda12].
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
import pathlib
import site
import sysconfig

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

# --- Guard: refuse to launch an LM config through the image entrypoint. -------
MODEL="$(python3 - "${CONFIG_PATH}" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
print(str(cfg.get("model", "")).strip())
PY
)"
case "${MODEL}" in
    transformer|pythia*)
        echo "ERROR: ${CONFIG_PATH} has model='${MODEL}', which is an LM config." >&2
        echo "       Use nc_train_lm.sh / train_lm.py for language-model runs." >&2
        exit 2
        ;;
    "")
        echo "ERROR: could not read 'model' from ${CONFIG_PATH}." >&2
        exit 2
        ;;
esac

module list
echo "HOSTNAME=$(hostname)"
echo "PWD=${PWD}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "MODEL=${MODEL}"
echo "EXP_NAME=${EXP_NAME:-<config default>}"
echo "JOB_IDX=${JOB_IDX:-<not set>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<not set>}"
nvidia-smi -L || true

# --- Guard: refuse to train on CPU. -------------------------------------------
# A cuPTI/CUDA mismatch makes JAX fall back to CpuDevice silently; a ViT run then
# crawls for hours and produces nothing useful. Fail fast instead.
# Set ALLOW_CPU=1 to bypass (local smoke tests).
if [[ "${ALLOW_CPU:-0}" != "1" ]]; then
    # Accept any non-CPU backend so a working GPU can never be false-aborted.
    # Do NOT swallow stderr: the JAX plugin prints the real reason there.
    if ! python3 -c 'import jax,sys; sys.exit(0 if jax.default_backend()!="cpu" or any(d.platform!="cpu" for d in jax.devices()) else 1)'; then
        echo "" >&2
        echo "ERROR: JAX sees no GPU -- would train on CPU. Aborting." >&2
        echo "--------------------------- GPU DIAGNOSTICS ---------------------------" >&2
        python3 -c 'import jax; print("devices:", jax.devices()); print("backend:", jax.default_backend())' >&2 2>&1 || true
        python3 -m pip list 2>/dev/null | grep -iE "^(jax|jaxlib|nvidia-|cuda)" >&2 || true
        echo "-----------------------------------------------------------------------" >&2
        echo "Set ALLOW_CPU=1 to override (not recommended for real runs)." >&2
        exit 3
    fi
fi

EXTRA_FLAGS=()
if [[ -n "${EXP_NAME}" ]]; then
  EXTRA_FLAGS+=(--exp_name="${EXP_NAME}")
fi
if [[ -n "${JOB_IDX:-}" ]]; then
  EXTRA_FLAGS+=(--job_idx="${JOB_IDX}")
fi

echo "=== ViT run: config=${CONFIG_PATH} model=${MODEL} job_idx=${JOB_IDX:-<none>} ==="
python3 train.py --config="${CONFIG_PATH}" "${EXTRA_FLAGS[@]}"
