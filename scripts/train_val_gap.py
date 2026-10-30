"""Measures training and validation loss of one checkpoint the same way, to get a fair train/val gap.

During training, the logged training loss is a single batch, measured before the step that trains on it, so it is
noisy and not comparable with the validation average. Here both losses use the same frozen weights, the same
evaluate() function and the same number of non-overlapping windows from the start of each file.

    uv run scripts/train_val_gap.py --run_dir runs/a100/owt-a100-128b-328m
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from cs336_basics.checkpoint import upgrade_state_dict
from cs336_basics.data_loading import get_nonoverlapping_indices
from cs336_basics.decoding import latest_checkpoint_path
from cs336_basics.run_config import RunConfig
from cs336_basics.training import evaluate
from cs336_basics.transformer_lm import TransformerLM


def train_val_gap(args: argparse.Namespace):
    with open(os.path.join(args.run_dir, "config.json"), "r") as f:
        config: RunConfig = RunConfig.model_validate(json.load(f))
    model: torch.nn.Module = TransformerLM(
        vocab_size=config.model_vocab_size,
        context_length=config.model_context_length,
        num_layers=config.model_num_layers,
        d_model=config.model_d_model,
        num_heads=config.model_num_heads,
        d_ff=config.model_d_ff,
        eps=config.model_eps,
        theta=config.model_rope_theta,
        device=torch.device(args.device),
        dtype=torch.float32,
        tie_embeddings=config.model_tie_embeddings,
        qk_norm=config.model_qk_norm,
    )
    checkpoint_path = args.checkpoint if args.checkpoint is not None else latest_checkpoint_path(os.path.join(args.run_dir, "checkpoints"))
    print(f"Loading model from {checkpoint_path}...")
    checkpoint: dict[str, any] = torch.load(checkpoint_path, map_location=args.device)
    upgrade_state_dict(checkpoint["model"])  # older checkpoints: torch.compile key prefix
    model.load_state_dict(checkpoint["model"])

    train_dataset = np.load(file=config.train_data, mmap_mode='r')
    val_dataset = np.load(file=config.val_data, mmap_mode='r')

    losses: dict[str, float] = {}
    for name, dataset in [("train", train_dataset), ("val", val_dataset)]:
        indices = get_nonoverlapping_indices(dataset, config.model_context_length, args.windows)
        losses[name] = evaluate(model, dataset, indices, args.batch_size, config.model_context_length, args.device)
        print(f"{name}: {losses[name]:.4f} on {len(indices) * config.model_context_length / 1e6:.2f}M tokens")

    # Training draws random windows, so a given training window was seen with probability 1 - e^(-epochs).
    epochs = checkpoint["iteration"] * config.batch_size * config.model_context_length / train_dataset.size
    print(f"gap (val - train): {losses['val'] - losses['train']:+.4f}")
    print(f"epochs trained: {epochs:.3f}, so about {1 - math.exp(-epochs):.0%} of the measured training windows were trained on")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument("--run_dir", type=str, required=True)
    arg_parser.add_argument("--checkpoint", type=str, default=None)
    arg_parser.add_argument("--windows", type=int, default=2048, help="non-overlapping windows per split, same as the validation subset")
    arg_parser.add_argument("--batch_size", type=int, default=32, help="evaluation batch size; doesn't change the result, only memory and speed")
    arg_parser.add_argument("--device", type=str, default="mps")
    args = arg_parser.parse_args()

    train_val_gap(args)
