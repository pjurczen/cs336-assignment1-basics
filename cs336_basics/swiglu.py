import torch

from cs336_basics.linear import Linear


class SwiGLU(torch.nn.Module):
    w_1: Linear  # (d_ff, d_model)
    w_2: Linear  # (d_model, d_ff)
    w_3: Linear  # (d_ff, d_model)

    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if d_ff is None:
            d_ff = round(8 * d_model / 3 / 64) * 64
        self.w_1 = Linear(d_model, d_ff, device, dtype)
        self.w_2 = Linear(d_ff, d_model, device, dtype)
        self.w_3 = Linear(d_model, d_ff, device, dtype)

    # x (batch_size, sequence_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self._silu(self.w_1(x)) * self.w_3(x))

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
