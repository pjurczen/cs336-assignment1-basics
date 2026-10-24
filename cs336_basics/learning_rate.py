import math


def cosine_lr_schedule(t: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if t < warmup_iters:
        return t / warmup_iters * lr_max
    if warmup_iters <= t <= cosine_cycle_iters:
        return lr_min + 1 / 2 * (1 + math.cos((t - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (lr_max - lr_min)
    return lr_min
