import math

import torch
from einops import einsum
from jaxtyping import Bool, Float

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
