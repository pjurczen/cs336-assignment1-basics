import math
from typing import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> float:
    params = [p for p in parameters if p.grad is not None]
    l2_norm = math.sqrt(sum([p.grad.pow(2).sum() for p in params]))
    if l2_norm >= max_l2_norm:
        scaling_factor = max_l2_norm / (l2_norm + eps)
        for param in params:
            if param.grad is not None:
                param.grad = param.grad.multiply(scaling_factor)
    return l2_norm
