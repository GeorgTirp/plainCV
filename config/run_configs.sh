#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  "config/config.yaml"
  "config/config_shampoo.yaml"
  "config/config copy.yaml"
  "config/config copy 2.yaml"
  "config/config copy 3.yaml"
  #"config/config_shampoo.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "=================================================="
  echo "Running: python3 train.py --config $cfg"
  echo "Started at: $(date)"
  echo "=================================================="

  python3 train.py --config "$cfg"

  echo "Finished: $cfg at $(date)"
  echo
done

echo "All runs finished."
