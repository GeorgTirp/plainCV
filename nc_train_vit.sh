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

module load cuda/12.9 >/dev/null 2>&1 || true

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

echo "=== ViT run: config=${CONFIG} model=${MODEL} job_idx=${JOB_IDX:-<none>} ==="

if [[ -n "${JOB_IDX}" ]]; then
    exec "${PY}" train.py --config="${CONFIG}" --job_idx="${JOB_IDX}"
else
    exec "${PY}" train.py --config="${CONFIG}"
fi
