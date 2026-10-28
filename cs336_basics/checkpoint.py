import os
from typing import BinaryIO, IO, Optional

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
        elapsed_time: float | None = None,
        muon_optimizer: torch.optim.Optimizer | None = None,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if elapsed_time is not None:
        checkpoint["elapsed_time"] = elapsed_time
    if muon_optimizer is not None:
        checkpoint["muon_optimizer"] = muon_optimizer.state_dict()
    torch.save(checkpoint, out)


def upgrade_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """Converts model weights saved by older code to the current layout, in place. Returns whether anything changed."""
    # Checkpoints saved from the torch.compile wrapper prefix every key.
    consume_prefix_in_state_dict_if_present(state_dict, "_orig_mod.")

    # Attention used to store Q, K and V as one fused matrix, stacked along the output (first) dimension in that order.
    fused_keys = [key for key in state_dict if key.endswith("attn.qkv_proj.weight")]
    for fused_key in fused_keys:
        prefix = fused_key.removesuffix("qkv_proj.weight")  # e.g. "layers.0.attn."
        q_weight, k_weight, v_weight = torch.chunk(state_dict.pop(fused_key), 3, dim=0)
        state_dict[prefix + "q_proj.weight"] = q_weight
        state_dict[prefix + "k_proj.weight"] = k_weight
        state_dict[prefix + "v_proj.weight"] = v_weight
    return len(fused_keys) > 0


def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        muon_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, Optional[float]]:
    # Load onto the model's device, so a checkpoint saved on CUDA can be resumed on MPS and vice versa.
    checkpoint: dict[str, any] = torch.load(src, map_location=next(model.parameters()).device)
    if upgrade_state_dict(checkpoint["model"]):
        # The optimizer state belongs to the old fused parameter and can't be mapped onto q/k/v_proj.
        raise ValueError("This checkpoint uses the old fused qkv_proj layout: its weights load for decoding and evaluation, "
                         "but training can't be resumed from it")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if muon_optimizer is not None:
        if "muon_optimizer" not in checkpoint:
            raise ValueError("Resuming with Muon, but the checkpoint was trained without it")
        muon_optimizer.load_state_dict(checkpoint["muon_optimizer"])
    return checkpoint["iteration"], checkpoint.get("elapsed_time", None)
