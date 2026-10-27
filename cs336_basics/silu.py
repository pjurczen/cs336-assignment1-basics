import torch

from cs336_basics.linear import Linear


class SiLU(torch.nn.Module):
    w1: Linear  # (d_ff, d_model)
    w2: Linear  # (d_model, d_ff)

    def __init__(self, d_model: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        d_ff = 4 * d_model
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    # x (batch_size, sequence_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
