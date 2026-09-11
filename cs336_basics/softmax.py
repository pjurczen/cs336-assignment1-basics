import torch


def softmax(x: torch.Tensor, dim_i: int) -> torch.Tensor:
    x_max = x.max(dim=dim_i, keepdim=True).values
    x_norm = x.subtract(x_max)
    return x_norm.exp() / x_norm.exp().sum(dim=dim_i, keepdim=True)
