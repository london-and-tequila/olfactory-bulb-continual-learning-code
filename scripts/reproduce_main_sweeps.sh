#!/usr/bin/env bash
set -euo pipefail

NETWORKS=(ng1_v2 random_k_v2 topk_noinit_v2)
SEEDS=(0 1 2 3)

seed_suffix() {
  local seed="$1"
  if [[ "$seed" == "0" ]]; then
    printf ""
  else
    printf "_seed%s" "$seed"
  fi
}

for seed in "${SEEDS[@]}"; do
  suffix="$(seed_suffix "$seed")"
  for net in "${NETWORKS[@]}"; do
    python run_fig2_baseline.py \
      --network_type "$net" \
      --n_seeds 1 \
      --seed_start "$seed" \
      --output_root "results/v2_n1300_fig2${suffix}"

    python run_fig3_k_sweep.py \
      --network_type "$net" \
      --n_seeds 1 \
      --seed_start "$seed" \
      --output_root "results/v2_n1300_fig3${suffix}" \
      --resume

    python run_fig4_cl_sweep.py \
      --network_type "$net" \
      --n_seeds 1 \
      --seed_start "$seed" \
      --output_root "results/v2_n1300_fig4${suffix}" \
      --resume

    python run_fig5_corr_sweep.py \
      --network_type "$net" \
      --n_seeds 1 \
      --seed_start "$seed" \
      --output_root "results/v2_n1300_fig5${suffix}" \
      --resume
  done
done
