import numpy as np
import numpy.typing as npt
import torch
from einops import rearrange


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str, dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    indices = np.random.randint(0, dataset.size - context_length, size=batch_size)
    indices = rearrange(indices, "... -> ... 1")

    input_offsets = np.arange(0, context_length)
    input_sequences = indices + input_offsets

    target_offsets = np.arange(1, context_length + 1)
    target_sequences = indices + target_offsets

    inputs = dataset[input_sequences]
    targets = dataset[target_sequences]

    inputs_tensor = torch.tensor(inputs, device=device, dtype=dtype)
    targets_tensor = torch.tensor(targets, device=device, dtype=dtype)

    return inputs_tensor, targets_tensor
