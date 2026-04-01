#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$REPO_ROOT"

source "$REPO_ROOT/.venv/bin/activate"

mkdir -p "$REPO_ROOT/job_outputs"
mkdir -p "$REPO_ROOT/data/datasets/outputs"
mkdir -p "$REPO_ROOT/data/datasets/hf_cache"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/data/datasets/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$REPO_ROOT/data/datasets/hf_cache}"

# Default to the stage you need now because raw_dataset already exists.
# To rerun from scratch, submit with:
#   PREP_FLAGS="--download --tokenize --chunk" condor_submit cluster/preprocess/condor.sub
PREP_FLAGS="${PREP_FLAGS:---tokenize --chunk}"

python data/datasets/prepare.py \
  --out_path="$REPO_ROOT/data/datasets/outputs/fwedu/fwedu_sample_10BT_tokenizer_GPT2" \
  --cache_path="$REPO_ROOT/data/datasets/hf_cache" \
  ${PREP_FLAGS} \
  --dataset_path="HuggingFaceFW/fineweb-edu" \
  --dataset_split="train" \
  --dataset_name="sample-10BT" \
  --tokenizer="gpt2" \
  --seq_length=2048 \
  --split_train_valid \
  --n_tokens_valid=10000000 \
  --save_raw \
  --save_tokenized \
  --save_tokenizer
