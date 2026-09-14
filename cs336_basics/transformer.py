import torch

from cs336_basics.attention import CasualMultiHeadSelfAttention
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(torch.nn.Module):
    norm_attention: RMSNorm
    attention: CasualMultiHeadSelfAttention

    feed_forward: SwiGLU
    norm_feed_forward: RMSNorm

    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, eps: float = 1e-5, max_seq_len: int | None = None, theta: float | None = None,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.norm_attention = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.attention = CasualMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.norm_feed_forward = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.feed_forward = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        y = x + self.attention(self.norm_attention(x), token_positions)
        y = y + self.feed_forward(self.norm_feed_forward(y))
        return y
