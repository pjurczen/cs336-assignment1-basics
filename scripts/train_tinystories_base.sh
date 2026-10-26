#!/usr/bin/env bash
# TinyStories base run. Extra flags override the ones below, e.g.:
#   scripts/train_tinystories_base.sh --run_name tinystories-lr3e-3 --optimizer_lr_max 3e-3
set -euo pipefail

uv run cs336_basics/training.py \
  --train_data data/tinystories-10k/train_ids.npy \
  --val_data data/tinystories-10k/valid_ids.npy \
  --tokenizer_dir tokenizers/tinystories-10k \
  --runs_dir runs \
  --run_name tinystories-base-32b \
  --model_context_length 256 \
  --model_num_layers 4 \
  --model_d_model 512 \
  --model_num_heads 16 \
  --model_d_ff 1344 \
  --model_eps 1e-5 \
  --model_rope_theta 10000 \
  --optimizer_lr_min 1e-4 \
  --optimizer_lr_max 1e-3 \
  --optimizer_warmup_iters 120 \
  --optimizer_cosine_cycle_iters 5000 \
  --optimizer_weight_decay 0.01 \
  --optimizer_beta1 0.9 \
  --optimizer_beta2 0.999 \
  --optimizer_eps 1e-8 \
  --gradient_max_l2_norm 1.0 \
  --gradient_eps 1e-6 \
  --batch_size 32 \
  --iter 5000 \
  --log_interval 100 \
  --val_interval 500 \
  --val_batches 64 \
  --checkpoints_iter 500 \
  --device mps \
  "$@"
