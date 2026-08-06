#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/lustre/home/gtirpitz/plainCV"
CONFIG_PATH="${1:?need config path like config/lm_signum.yaml}"
EXP_NAME="${2:-}"
# "_" is a placeholder for empty when called from HTCondor (which rejects "")
[[ "${EXP_NAME}" == "_" ]] && EXP_NAME=""
CURV_BACKEND="${3:-}"
TMP_CONFIG_PATH=""

cleanup_tmp_config() {
  if [[ -n "${TMP_CONFIG_PATH}" && -f "${TMP_CONFIG_PATH}" ]]; then
    rm -f "${TMP_CONFIG_PATH}"
  fi
}
trap cleanup_tmp_config EXIT

cd "${REPO_DIR}"

mkdir -p job_outputs

source /etc/profile.d/modules.sh
module purge
module load cuda/12.9
module load cudnn/9.10.2
if [[ -n "${NCCL_MODULE:-}" ]]; then
  module load "${NCCL_MODULE}"
fi

source .venv/bin/activate

export WANDB_DIR="${PWD}/wandb"

# Keep CUDA runtime/tooling libs visible to JAX. NCCL is not part of this
# cluster's cuda/12.1 module, so also add NVIDIA wheel library dirs if the venv
# was installed with jax[cuda12] / nvidia-*-cu12 packages.
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

if [[ -n "${CURV_BACKEND}" ]]; then
  TMP_CONFIG_PATH="$(mktemp "${TMPDIR:-/tmp}/plaincv_lm_config.XXXXXX.yaml")"
  python3 - "${CONFIG_PATH}" "${TMP_CONFIG_PATH}" "${CURV_BACKEND}" <<'PY'
import sys
import yaml

src, dst, raw_backend = sys.argv[1:4]
backend_key = raw_backend.strip().lower()
aliases = {
    "ggn": ("ggn", "GGN", False),
    "efisher": ("fisher", "EFISHER", True),
    "fisher": ("fisher", "EFISHER", True),
    "ot": ("wasserstein", "OT", False),
    "wasserstein": ("wasserstein", "OT", False),
    "laplacian": ("wasserstein", "OT", False),
}
if backend_key not in aliases:
    choices = ", ".join(sorted(aliases))
    raise SystemExit(f"Unknown curvature backend '{raw_backend}'. Use one of: {choices}.")

backend, prefix, sort_by_abs = aliases[backend_key]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

optim = str(cfg.get("optim", "optimizer")).lower().replace("_", "-")
cfg["eigen_tracking_enabled"] = True
cfg["eigen_tracking_backend"] = backend
cfg["eigen_tracking_sort_by_abs"] = sort_by_abs
cfg["eigen_tracking_measurement_mode"] = cfg.get(
    "eigen_tracking_measurement_mode", "probe_update"
)
cfg["eigen_tracking_topk"] = cfg.get("eigen_tracking_topk", 6)
cfg["eigen_tracking_extra_modes"] = cfg.get("eigen_tracking_extra_modes", 6)
cfg["eigen_tracking_curvature_batches"] = cfg.get(
    "eigen_tracking_curvature_batches", 1
)
cfg["eigen_tracking_lanczos_iters"] = cfg.get(
    "eigen_tracking_lanczos_iters", 12
)

if backend == "wasserstein":
    cfg["wasserstein_lm_top_k"] = cfg.get("wasserstein_lm_top_k", 256)
    cfg["wasserstein_lm_max_positions"] = cfg.get(
        "wasserstein_lm_max_positions", 512
    )
    cfg["wasserstein_graph_temp"] = cfg.get("wasserstein_graph_temp", 0.0)
    cfg["wasserstein_graph_eps"] = cfg.get("wasserstein_graph_eps", 1e-6)
    cfg["wasserstein_laplacian_ridge"] = cfg.get(
        "wasserstein_laplacian_ridge", 1e-4
    )

existing_name = cfg.get("exp_name", "")
if existing_name.endswith("_seeds"):
    cfg["wandb_run_name"] = f"{prefix}_{optim}_seeds"
    cfg["exp_name"] = f"{prefix}_{optim}_seeds"
else:
    cfg["wandb_run_name"] = f"{prefix}_{optim}"
    cfg["exp_name"] = f"{prefix}_{optim}"

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  echo "Generated ${CURV_BACKEND} eigen-tracking config at ${TMP_CONFIG_PATH}"
  CONFIG_PATH="${TMP_CONFIG_PATH}"
fi

module list

echo "HOSTNAME=$(hostname)"
echo "PWD=${PWD}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "EXP_NAME=${EXP_NAME:-<config default>}"
echo "CURV_BACKEND=${CURV_BACKEND:-<config default>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-<not set>}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-<not set>}"
echo "XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-<not set>}"
echo "RUN_NCCL_PREFLIGHT=${RUN_NCCL_PREFLIGHT:-0}"
echo "NCCL_DEBUG=${NCCL_DEBUG:-<not set>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<not set>}"
nvidia-smi -L || true

python3 - <<'PY'
import ctypes
import ctypes.util
import glob
import os

print("ctypes.find_library('nccl'):", ctypes.util.find_library("nccl"))
for lib_dir in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    if not lib_dir:
        continue
    for candidate in glob.glob(os.path.join(lib_dir, "libnccl.so*"))[:3]:
        print("NCCL candidate:", candidate)

try:
    ctypes.CDLL("libnccl.so.2")
    print("libnccl.so.2 load: ok")
except OSError as exc:
    print(f"libnccl.so.2 load: FAILED: {exc}")
PY

if [[ "${RUN_NCCL_PREFLIGHT:-0}" == "1" ]]; then
  python3 - <<'PY'
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp

print("JAX devices:", jax.devices())
print("JAX local_device_count:", jax.local_device_count())

if jax.local_device_count() > 1:
    try:
        @partial(jax.pmap, axis_name="data")
        def psum_probe(x):
            return jax.lax.psum(x, axis_name="data")

        out = jax.device_get(psum_probe(jnp.arange(jax.local_device_count(), dtype=jnp.float32)))
        print("pmap collective probe: ok", out)
    except Exception as exc:
        print(f"pmap collective probe: FAILED: {exc}")
        if os.environ.get("ALLOW_SINGLE_GPU_FALLBACK", "0") != "1":
            print(
                "Refusing to start a multi-GPU job without working NCCL. "
                "Set ALLOW_SINGLE_GPU_FALLBACK=1 to train on one visible GPU anyway."
            )
            sys.exit(1)

    try:
        import torch  # noqa: F401

        @partial(jax.pmap, axis_name="data")
        def psum_after_torch_probe(x):
            return jax.lax.psum(x, axis_name="data")

        out = jax.device_get(
            psum_after_torch_probe(jnp.arange(jax.local_device_count(), dtype=jnp.float32))
        )
        print("pmap collective probe after torch import: ok", out)
    except Exception as exc:
        print(f"pmap collective probe after torch import: FAILED: {exc}")
PY
fi

EXTRA_FLAGS=()
if [[ -n "${EXP_NAME}" ]]; then
  EXTRA_FLAGS+=(--exp_name="${EXP_NAME}")
fi
if [[ -n "${JOB_IDX:-}" ]]; then
  EXTRA_FLAGS+=(--job_idx="${JOB_IDX}")
fi

python3 train_lm.py --config="${CONFIG_PATH}" "${EXTRA_FLAGS[@]}"
