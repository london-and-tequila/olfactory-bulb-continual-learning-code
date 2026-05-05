#!/usr/bin/env bash
set -euo pipefail

python run_neurogenesis_multi_pair.py \
  --network_type ng1_v2 \
  --n_mitral 40 \
  --n_granule 100 \
  --n_granule_per_task 5 \
  --n_pretrain_pairs 2 \
  --n_train_pairs 5 \
  --n_test_pairs 2 \
  --n_epochs_per_pair 1 \
  --n_steps_to_steady 5 \
  --lr_B 0.3 \
  --n_seeds 1 \
  --seed 0 \
  --output_dir results/smoke \
  --exp_name ng1_v2
