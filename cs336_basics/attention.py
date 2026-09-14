import math

import torch
from jaxtyping import Bool, Float

from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
        Q: Float[torch.Tensor, " ... queries d_k"],
        K: Float[torch.Tensor, " ... keys d_k"],
        V: Float[torch.Tensor, " ... keys d_v"],
        mask: Bool[torch.Tensor, " ... queries keys"] | None
) -> Float[torch.Tensor, " ... seq_len d_v"]:
    d_k = Q.shape[-1]
    qk = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        qk = qk.masked_fill(~mask, float("-Inf"))
    qk = softmax(qk, dim=-1)
    return einsum(qk, V, "... queries keys, ... keys d_v -> ... queries d_v")


import torch
from einops import einsum, rearrange


class CasualMultiHeadSelfAttention(torch.nn.Module):
    device: torch.device | None
    d_k: int
    d_v: int
    d_model: int
    num_heads: int
    q_proj: Linear  # (num_heads * d_k, d_model) = (hd_k, d_model)
    k_proj: Linear  # (num_heads * d_k, d_model) = (hd_k, d_model)
    v_proj: Linear  # (num_heads * d_v, d_model) = (hd_v, d_model)
    output_proj: Linear  # (d_model, num_heads * d_v) = (d_model, hd_v)
    rope: RotaryPositionalEmbedding | None

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        if d_model % num_heads != 0:
            raise Exception("d_model must be divisible by num_heads!")
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = round(d_model / num_heads)
        self.d_v = self.d_k
        self.q_proj = Linear(d_model, num_heads * self.d_k, device, dtype)
        self.k_proj = Linear(d_model, num_heads * self.d_k, device, dtype)
        self.v_proj = Linear(d_model, num_heads * self.d_v, device, dtype)
        self.output_proj = Linear(num_heads * self.d_v, d_model, device, dtype)
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
        else:
            self.rope = None

    # x (..., seq_len, d_model)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        casual_mask = (torch.tril(torch.ones(seq_len, seq_len, device=self.device)) == 1)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=self.device)
        wq_x = self.q_proj.forward(x)  # (..., seq_len, hd_k)
        wk_x = self.k_proj.forward(x)  # (..., seq_len, hd_k)
        wv_x = self.v_proj.forward(x)  # (..., seq_len, hd_v)
        wq_x_i = rearrange(wq_x, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)  # (..., h, seq_len, d_k)
        wk_x_i = rearrange(wk_x, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)  # (..., h, seq_len, d_k)
        wv_x_i = rearrange(wv_x, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)  # (..., h, seq_len, d_v)
        if self.rope:
            wq_x_i = self.rope.forward(wq_x_i, token_positions)
            wk_x_i = self.rope.forward(wk_x_i, token_positions)
        result = scaled_dot_product_attention(wq_x_i, wk_x_i, wv_x_i, casual_mask)  # (..., h, seq_len, d_v)
        result = rearrange(result, "... h seq_len d_v -> ... seq_len (h d_v)")
        return self.output_proj.forward(result)


class CasualMultiHeadSelfAttentionOptimized(torch.nn.Module):
    device: torch.device | None
    d_k: int
    d_v: int
    d_model: int
    num_heads: int
    qkv_proj: Linear  # (hd_k + hd_k + hd_v, d_model)
    output_proj: Linear  # (d_model, num_heads * d_v) = (d_model, hd_v)
    rope: RotaryPositionalEmbedding | None

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        if d_model % num_heads != 0:
            raise Exception("d_model must be divisible by num_heads!")
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = round(d_model / num_heads)
        self.d_v = self.d_k
        self.qkv_proj = Linear(d_model, num_heads * self.d_k + num_heads * self.d_k + num_heads * self.d_v, device, dtype)
        self.output_proj = Linear(num_heads * self.d_v, d_model, device, dtype)
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
        else:
            self.rope = None

    # x (..., seq_len, d_model)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        casual_mask = (torch.tril(torch.ones(seq_len, seq_len, device=self.device)) == 1)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=self.device)
        w_x = self.qkv_proj.forward(x)  # (..., seq_len, hd_k + hd_k + hd_v)
        wq_x = w_x[..., :self.num_heads * self.d_k]
        wk_x = w_x[..., self.num_heads * self.d_k:2 * self.num_heads * self.d_k]
        wv_x = w_x[..., 2 * self.num_heads * self.d_k:]
        wq_x_i = rearrange(wq_x, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)  # (..., h, seq_len, d_k)
        wk_x_i = rearrange(wk_x, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)  # (..., h, seq_len, d_k)
        wv_x_i = rearrange(wv_x, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)  # (..., h, seq_len, d_v)
        if self.rope:
            wq_x_i = self.rope.forward(wq_x_i, token_positions)
            wk_x_i = self.rope.forward(wk_x_i, token_positions)
        result = scaled_dot_product_attention(wq_x_i, wk_x_i, wv_x_i, casual_mask)  # (..., h, seq_len, d_v)
        result = rearrange(result, "... h seq_len d_v -> ... seq_len (h d_v)")
        return self.output_proj.forward(result)
