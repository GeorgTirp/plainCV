#!/bin/bash

# This script will download and preprocess FineWebEdu-100BT.
# Expect some token loss by batched concat_chunk.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)"
OUT_ROOT="${REPO_ROOT}/data/datasets/outputs"
CACHE_PATH="${REPO_ROOT}/data/datasets/hf_cache"

mkdir -p "${OUT_ROOT}" "${CACHE_PATH}"
cd "${REPO_ROOT}"

PYTHONPATH=. python data/datasets/prepare.py \
  --out_path="${OUT_ROOT}/fwedu/fwedu_sample_100B_tokenizer_GPTNeoX" \
  --cache_path="${CACHE_PATH}" \
  --download --tokenize --chunk \
  --save_tokenized --save_tokenizer \
  --dataset_path="HuggingFaceFW/fineweb-edu" \
  --dataset_split="train" \
  --dataset_name="sample-100BT" \
  --tokenizer="EleutherAI/gpt-neox-20b" \
  --seq_length=2048 \
  --split_train_valid=True \
  --n_tokens_valid=10000000
