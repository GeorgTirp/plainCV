#!/bin/bash
set -euo pipefail
#
# Cluster wrapper for the ViT (image) experiments -> train.py
#
# This is the IMAGE counterpart of nc_train_lm.sh / nc_train_ggn.sh, which drive
# train_lm.py. The two pipelines are deliberately separate entrypoints:
#
#     train.py      <- ViT / ResNet / MLP  (this wrapper)
#     train_lm.py   <- transformer / pythia LM
#
# Only the MEASUREMENT protocol is shared (optim/eigentools.py + the eigen-tracking
# CSV writers in utils.py). Never point a ViT config at train_lm.py: it hard-fails
# with "LM training expects model='transformer' or 'pythia*'".
#
# Usage (HTCondor):
#   executable  = nc_train_vit.sh
#   arguments   = <config.yaml>
#   environment = "JOB_IDX=<idx>"
#
# Also accepts the job index positionally for manual runs:
#   ./nc_train_vit.sh config/vit_tinyimagenet_ggn.yaml 0

cd "$(cd "$(dirname "$0")" && pwd)"

CONFIG="${1:?usage: nc_train_vit.sh <config.yaml> [job_idx]}"
# Condor passes the index via the environment; allow a positional override.
JOB_IDX="${JOB_IDX:-${2:-}}"

# --- CUDA modules -------------------------------------------------------------
# `module` is a shell function, which does NOT exist in a non-interactive Condor
# shell until the init script is sourced. Do this FIRST, and do NOT silence or
# `|| true` the load: if CUDA is missing, jaxlib cannot load cuPTI and silently
# falls back to CPU, which wastes the whole GPU slot.
if ! type module >/dev/null 2>&1; then
    for _mi in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/modules/init/bash; do
        [[ -r "${_mi}" ]] && source "${_mi}" && break
    done
fi
module load cuda/12.9

if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
else
    PY="python"
fi

# --- Guard: refuse to launch an LM config through the image entrypoint. -------
# Cheap structural check so a mis-wired .sub fails loudly here instead of
# burning a GPU slot and dying inside train.py.
MODEL="$("${PY}" - "${CONFIG}" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
print(str(cfg.get("model", "")).strip())
PY
)"
case "${MODEL}" in
    transformer|pythia*)
        echo "ERROR: ${CONFIG} has model='${MODEL}', which is an LM config." >&2
        echo "       Use nc_train_lm.sh / train_lm.py for language-model runs." >&2
        exit 2
        ;;
    "")
        echo "ERROR: could not read 'model' from ${CONFIG}." >&2
        exit 2
        ;;
esac

# --- Guard: refuse to train on CPU. -------------------------------------------
# A cuPTI/CUDA mismatch makes JAX fall back to CpuDevice silently; a ViT run then
# crawls for hours and produces nothing useful. Fail fast instead.
# Set ALLOW_CPU=1 to bypass (local smoke tests).
if [[ "${ALLOW_CPU:-0}" != "1" ]]; then
    if ! "${PY}" -c 'import jax,sys; sys.exit(0 if any(d.platform=="gpu" for d in jax.devices()) else 1)' 2>/dev/null; then
        echo "ERROR: JAX sees no GPU -- would train on CPU. Aborting." >&2
        echo "       Devices: $("${PY}" -c 'import jax; print(jax.devices())' 2>&1 | tail -1)" >&2
        echo "       Usually a CUDA/cuPTI load failure; check 'module load cuda/12.9' above." >&2
        echo "       Set ALLOW_CPU=1 to override." >&2
        exit 3
    fi
fi

echo "=== ViT run: config=${CONFIG} model=${MODEL} job_idx=${JOB_IDX:-<none>} ==="

if [[ -n "${JOB_IDX}" ]]; then
    exec "${PY}" train.py --config="${CONFIG}" --job_idx="${JOB_IDX}"
else
    exec "${PY}" train.py --config="${CONFIG}"
fi
