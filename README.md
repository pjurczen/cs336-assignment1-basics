> **My work on this fork (self-study, 2025–26).** All parts of the assignment are complete: tokenizer (§2), model (§3), training (§4–5) and experiments (§7). On
> top of that, I pushed the OpenWebText model toward the leaderboard setting (45 minutes on one B200, context 512): **validation loss 3.191**.
>
> - **BPE tokenizer from scratch** with parallel, `mmap`-based pretokenization across all CPU cores and a bucketed pair counter that replaced the naive rescan —
    **merge loop 66 s → 5.2 s (12.7×)** on TinyStories; a 10K vocabulary trains in 34.8 s at ~1.5 GB peak RSS.
> - **32K-vocabulary BPE trained on the full 11.9 GB OpenWebText corpus** in 61 min at ~3.6 GB peak; both corpora tokenized to `uint16` id arrays. Measured
    compression 4.0 bytes/token (TinyStories) vs 4.5 (OpenWebText), and a 31% degradation from cross-domain tokenizer mismatch.
> - **Transformer components in PyTorch without `torch.nn` layers:** `Linear`, `Embedding`, `RMSNorm`, `SwiGLU`, softmax, scaled dot-product attention, RoPE,
    causal multi-head attention, the transformer block and the full `TransformerLM`, with FLOPs accounting annotated inline.
> - **Training:** cross-entropy, AdamW, cosine schedule with warmup, gradient clipping, memory-mapped data loading, checkpointing and the training loop, run on
    A100s (Vertex AI) and B200s (Modal).
> - **Experiments (§7):** learning-rate and batch-size sweeps (√ learning-rate scaling holds below the critical batch size), warmup, and ablations: no RMSNorm,
    post-norm, NoPE, SiLU vs SwiGLU. Plots are in [`figures/`](./figures).
> - ~35 tests of my own (`tests/test_pretoken_counter.py`, `test_pretokenization.py`, `test_bpe_tokenizer.py`) on top of the course suite; 82 tests pass, 2 are
    skipped.
> - Written answers with measurements: [`assignment_1_answers.md`](./assignment_1_answers.md).

## Results

All losses are validation cross-entropy in nats per token; the final value of each run is measured on the whole validation set.

**TinyStories** (4 layers, d_model 512, batch 64, 40M tokens, one A100):

| Setup                                                       | Validation loss | Time per step |
|-------------------------------------------------------------|-----------------|---------------|
| Handout architecture, tuned (learning rate 2e-3, warmup 60) | 1.623           | 60 ms         |
| + tied embeddings, without scaling                          | 1.797           |               |
| + tied embeddings, scaled by √d_model                       | 1.584           |               |
| + Muon (`torch.optim.Muon`) for the block matrices          | 1.518           | 67 ms         |
| + my own Muon instead                                       | 1.516           | 60 ms         |
| + bf16 autocast                                             | 1.516           | 45 ms         |

**OpenWebText** (32K vocabulary):

| Setup                                                                     | Hardware, training time | Validation loss |
|---------------------------------------------------------------------------|-------------------------|-----------------|
| Handout main experiment: 4L/512, context 256, batch 128, 328M tokens      | A100, 27 min            | 3.926           |
| Same model and tokens; context 512, tied embeddings, Muon, bf16, batch 64 | A100, 24 min            | 3.707           |
| 8L/768, weight decay 0.1                                                  | A100, 15 min            | 3.669           |
| 8L/768 (weight decay 0.01)                                                | A100, 60 min            | 3.395           |
| 12L/768                                                                   | B200, 45 min            | 3.238           |
| **12L/768 + QK-norm, batch 256, learning rate 8e-3**                      | **B200, 45 min**        | **3.191**       |

The best settings and the evidence behind each one are in [`scripts/train_owt_best.sh`](./scripts/train_owt_best.sh). With more compute the deeper model caught
up: 8L/768 beat 12L/768 by 0.06 at 15 A100 minutes, by 0.01 at 60, and lost by 0.03 at 45 B200 minutes.

## Changes beyond the handout

The handout's architecture and training setup are the default; each change below is a command-line flag of `cs336_basics/train.py` unless noted.

| Change                                                                                                                                                                                  | Flag                                              | Effect                                                                           |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|----------------------------------------------------------------------------------|
| Tied input and output embeddings, the input scaled by √d_model (Vaswani et al., §3.4)                                                                                                   | `--model_tie_embeddings`                          | TinyStories 1.623 → 1.584; unscaled tying was worse (1.797)                      |
| Muon for the 2-D matrices in the blocks (my implementation: Nesterov momentum, 5 Newton–Schulz steps in bf16, update size matched to AdamW, decoupled weight decay), AdamW for the rest | `--optimizer_muon`                                | TinyStories 1.584 → 1.516                                                        |
| bf16 autocast for the forward pass and the loss; the training loss skips the float32 copy of the logits (3.9 GiB at batch 64)                                                           | `--bf16`                                          | 25% less time per step at the same loss; the memory saving made 12 layers fit    |
| One fused QKV projection                                                                                                                                                                | always on                                         | Separate Q, K, V projections were 8% slower with no gain                         |
| QK-norm: RMSNorm on each head's queries and keys, before RoPE                                                                                                                           | `--model_qk_norm`                                 | −0.016 on OpenWebText (A100, 15 min), and learning rates up to 8e-3 train stably |
| z-loss                                                                                                                                                                                  | `--z_loss_weight`                                 | No measurable effect; off                                                        |
| Logit soft-capping, c · tanh(logits / c)                                                                                                                                                | `--model_logit_soft_cap`                          | No measurable effect; off                                                        |
| Cosine schedule over wall-clock time, ending at a time budget minus a reserve for the final validation                                                                                  | `--time_budget_minutes`, `--time_reserve_minutes` | Fits a run into a fixed time limit                                               |
| `torch.compile` and TF32 matmuls on CUDA                                                                                                                                                | always on                                         |                                                                                  |

What didn't help: AdamW β2 0.95 instead of 0.999 (+0.04), decaying the learning rate to 0 instead of 10% of the peak (+0.023), and a wider model, 8L/1024 (3.818
vs 3.705 for 8L/768 at 15 A100 minutes).

**Speed is now the limit.** Batch 256 is only about 1.25× faster per token than batch 64, and the B200 runs at roughly 15% of its bf16 peak. The time goes into
work that grows with every token rather than into the matrix multiplies, and the attention implementation, which stores the full 512 × 512 score matrix per
head, is the likely main cost. That is the next thing to profile.

## Infrastructure

- **Vertex AI A100s:** Terraform in [`infra/`](./infra) (bucket, image registry, service accounts, wandb secret), the image built with Cloud Build ([
  `scripts/build_image.sh`](./scripts/build_image.sh)), and jobs submitted with [`scripts/submit_vertex_job.sh`](./scripts/submit_vertex_job.sh) (`cpu`, `a100`,
  `a100-80gb`).
- **Modal B200s:** [`scripts/modal_train.py`](./scripts/modal_train.py). Linux uses PyPI's torch built for CUDA 13; the CUDA 12.6 build has no B200 kernels.
- **Tracking:** runs log to Weights & Biases and to `runs/<run_name>/` (`config.json` with the git commit, `metrics.jsonl`); plots are made in [
  `notebooks/metrics_plot.ipynb`](./notebooks/metrics_plot.ipynb).

---

The original assignment README follows.

# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment

We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using

```sh
uv run <python_file_path>
```

and the environment will be automatically solved and activated when necessary.

### Run unit tests

```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data

Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

