#!/usr/bin/env bash
# Current best OpenWebText settings: the leaderboard configuration, 45 minutes on one B200 (context 512).
# Extra flags override the ones below. Batch 256 needs ~145 GB of GPU memory, so on an A100 or a Mac use batch 64
# with the matching learning rate, e.g. the A100 experiments (15 min of training per run):
#   scripts/submit_vertex_job.sh a100-80gb owt-32k scripts/train_owt_best.sh --run_name owt-a100-80gb-... \
#     --batch_size 64 --optimizer_lr_max 4e-3 --optimizer_lr_min 4e-4 --time_budget_minutes 20 --time_reserve_minutes 5
# or locally on a Mac:  scripts/train_owt_best.sh --device mps --run_name owt-mps-test --batch_size 8 --time_budget_minutes 5
# Modal B200 runs (scripts/modal_train.py) call train.py directly: pass these flags in --args.
#
# B200, 45 minutes: 12L/768 + QK-norm, batch 256, learning rate 8e-3 reached 3.191 (9.2K steps at 275 ms, 1.21B tokens,
#   peak 145 GB, final validation 84 s). Without QK-norm at batch 64 and 2.8e-3: 3.238 (29.6K steps at 87 ms,
#   0.97B tokens). Batch 256 is only ~1.25x faster per token than batch 64: the B200 stays at ~15% of its peak.
# Model: 12 layers, d_model 768. On the B200 it beat 8L/768: 3.238 vs 3.270. On A100s 8L/768 won at 15 minutes (3.705,
#   vs 3.771 for 12L/768; also beat 4L/512 3.795, 8L/512 3.786, 8L/1024 3.818) and narrowly at 60 minutes
#   (3.395 vs 3.406), with the gap shrinking as compute grew.
# QK-norm: -0.016 vs the baseline at 15 A100 minutes (noise between identical runs ~0.005), and it makes higher learning
#   rates safe: with it, 4e-3, 5.6e-3 and 8e-3 all tied (3.690-3.693). z-loss 1e-4 and logit soft-cap 30 added nothing,
#   also at high learning rates, so they stay off.
# Learning rate 8e-3 for batch 256: sqrt scaling from 4e-3 at batch 64, the top of the range that tied there. The floor
#   is 10% of the peak: decaying to 0 was worse (+0.023).
# Tied embeddings with sqrt(d_model) scaling, Muon for the block matrices, bf16 autocast: each won on TinyStories.
# Weight decay 0.1: 3.669 vs 3.705 at 0.01, 3.682 at 0.05, 3.662 at 0.2 (within noise of 0.1). 0.1 rather than 0.2
#   because the best weight decay tends to drop as training gets longer, and the final runs are much longer.
# AdamW beta2 0.999: 0.95 is worse (+0.04).
# Time reserve 2 minutes: covers the final full validation (84 s), so the whole run stays under 45 minutes.
set -euo pipefail

uv run cs336_basics/train.py \
  --train_data data/owt-32k/train_ids.npy \
  --val_data data/owt-32k/valid_ids.npy \
  --tokenizer_dir tokenizers/owt-32k \
  --model_context_length 512 \
  --model_num_layers 12 \
  --model_d_model 768 \
  --model_num_heads 12 \
  --model_d_ff 2048 \
  --model_tie_embeddings \
  --model_qk_norm \
  --optimizer_muon \
  --bf16 \
  --optimizer_lr_max 8e-3 \
  --optimizer_lr_min 8e-4 \
  --optimizer_warmup_iters 120 \
  --optimizer_weight_decay 0.1 \
  --optimizer_beta2 0.999 \
  --batch_size 256 \
  --iter 1000000 \
  --time_budget_minutes 45 \
  --time_reserve_minutes 2 \
  --val_interval 1000 \
  --val_batches 16 \
  --log_interval 100 \
  --checkpoints_iter 1000000 \
  --wandb_project cs336 \
  --device cuda \
  "$@"
