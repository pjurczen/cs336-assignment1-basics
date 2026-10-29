#!/usr/bin/env bash
# Current best OpenWebText settings, from the A100 experiments (context 512, 15 min of training per run unless noted).
# Extra flags override the ones below, e.g. on Vertex:
#   scripts/submit_vertex_job.sh a100 owt-32k scripts/train_owt_best.sh --run_name owt-a100-... --time_budget_minutes 60
# or locally on a Mac:  scripts/train_owt_best.sh --device mps --run_name owt-mps-test --time_budget_minutes 5
#
# Model: 8 layers, d_model 768 (3.705) beat 4L/512 (3.795), 8L/512 (3.786), 8L/1024 (3.818) and 12L/768 (3.771)
#   at 15 minutes. At 60 A100 minutes 8L/768 still won, but narrowly: 3.395 vs 3.406 for 12L/768, and the gap shrank
#   steadily (0.060 at 15 min -> 0.011), so 12L/768 likely wins with more compute, e.g. 45 B200 minutes.
# Tied embeddings with sqrt(d_model) scaling, Muon for the block matrices, bf16 autocast: each won on TinyStories.
# Learning rate 2.8e-3: with weight decay 0.1 it beats 2e-3 (+0.011) and 1.4e-3 (+0.025); 4e-3 is clearly worse.
# Weight decay 0.1: 3.669 vs 3.705 at 0.01, 3.682 at 0.05, 3.662 at 0.2 (within noise of 0.1). 0.1 rather than 0.2
#   because the best weight decay tends to drop as training gets longer, and the final runs are much longer.
# AdamW beta2 0.999: 0.95 is worse (+0.04).
set -euo pipefail

uv run cs336_basics/training.py \
  --train_data data/owt-32k/train_ids.npy \
  --val_data data/owt-32k/valid_ids.npy \
  --tokenizer_dir tokenizers/owt-32k \
  --model_context_length 512 \
  --model_num_layers 8 \
  --model_d_model 768 \
  --model_num_heads 12 \
  --model_d_ff 2048 \
  --model_tie_embeddings \
  --optimizer_muon \
  --bf16 \
  --optimizer_lr_max 2.8e-3 \
  --optimizer_lr_min 2.8e-4 \
  --optimizer_warmup_iters 120 \
  --optimizer_weight_decay 0.1 \
  --optimizer_beta2 0.999 \
  --batch_size 64 \
  --iter 1000000 \
  --time_budget_minutes 20 \
  --time_reserve_minutes 5 \
  --val_interval 1000 \
  --val_batches 16 \
  --log_interval 100 \
  --checkpoints_iter 1000000 \
  --wandb_project cs336 \
  --device cuda \
  "$@"
