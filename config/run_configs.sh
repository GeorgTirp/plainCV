#!/usr/bin/env bash
set -euo pipefail

RUNS=(
  # Format: "config_path|exp_folder_name"
  # Leave the exp folder name empty to use the config/default naming.
  # "config/config.yaml|"
  # "config/config_shampoo.yaml|"
  # "config/config copy.yaml|"
  #"config/config_vit copy 2.yaml| run_pns_eigenadam_vit_small_ggn"
  "config/config_vit.yaml|run_pns_eigenadam_vit_small_svgd"
  #"config/config_vit copy.yaml|run_pns_eigenadam_vit_small_LN_wasserstein"
  
  "config/config.yaml|run_pns_eigenadam_resnet18_wasserstein"
  "config/config copy.yaml|run_pns_eigenadam_resnet18_ggn"
)

for entry in "${RUNS[@]}"; do
  IFS='|' read -r cfg exp_name <<< "${entry}"
  if [[ -z "${cfg}" ]]; then
    continue
  fi

  args=(--config "$cfg")
  if [[ -n "${exp_name:-}" ]]; then
    args+=(--exp_name "$exp_name")
  fi

  echo "=================================================="
  echo "Running: python3 train.py ${args[*]}"
  echo "Started at: $(date)"
  echo "=================================================="

  python3 train.py "${args[@]}"

  echo "Finished: $cfg at $(date)"
  echo
done

echo "All runs finished."
