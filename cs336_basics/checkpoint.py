import os
from typing import BinaryIO, IO, Optional

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
        elapsed_time: float | None = None
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if elapsed_time is not None:
        checkpoint["elapsed_time"] = elapsed_time
    torch.save(checkpoint, out)


def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
) -> tuple[int, Optional[float]]:
    # Load onto the model's device, so a checkpoint saved on CUDA can be resumed on MPS and vice versa.
    checkpoint: dict[str, any] = torch.load(src, map_location=next(model.parameters()).device)
    # Older checkpoints were saved from the torch.compile wrapper, which prefixes every key.
    consume_prefix_in_state_dict_if_present(checkpoint["model"], "_orig_mod.")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"], checkpoint.get("elapsed_time", None)
