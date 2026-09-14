import torch

from cs336_basics.attention import CasualMultiHeadSelfAttentionOptimized
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(torch.nn.Module):
    ln1: RMSNorm
    attn: CasualMultiHeadSelfAttentionOptimized

    ffn: SwiGLU
    ln2: RMSNorm

    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, eps: float = 1e-5, max_seq_len: int | None = None, theta: float | None = None,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.attn = CasualMultiHeadSelfAttentionOptimized(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        y = x + self.attn(self.ln1(x), token_positions)
        y = y + self.ffn(self.ln2(y))
        return y
