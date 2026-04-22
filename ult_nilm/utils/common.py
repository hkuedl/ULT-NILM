"""Common utility functions for tensor shape, sampling, and numeric helpers."""

from __future__ import annotations

import numpy as np
import torch.nn as nn


def make_divisible(value: float, divisor: int, min_val: int | None = None) -> int:
    """Round `value` to the nearest multiple of `divisor`.

    Ported from the TensorFlow MobileNet helper. Guarantees that a round-down
    does not drop more than 10% of the original value.
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(value + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * value:
        new_v += divisor
    return new_v


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, f"invalid kernel size: {kernel_size}"
        return get_same_padding(kernel_size[0]), get_same_padding(kernel_size[1])
    assert isinstance(kernel_size, int), "kernel size must be int or tuple"
    assert kernel_size % 2 > 0, "kernel size must be odd"
    return kernel_size // 2


def sub_filter_start_end(kernel_size: int, sub_kernel_size: int) -> tuple[int, int]:
    """Return the centred slice bounds that map a large kernel to a smaller one."""
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def min_divisible_value(n: int, v: int) -> int:
    """Return the largest v' <= v that divides n; used for group-conv sizing."""
    if v >= n:
        return n
    while n % v != 0:
        v -= 1
    return v


def val2list(value, repeat_time: int = 1):
    if isinstance(value, (list, np.ndarray)):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value for _ in range(repeat_time)]


def list_sum(x):
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x):
    return list_sum(x) / len(x)


def build_activation(act_func: str | None, inplace: bool = True):
    """Create an activation module from a short name."""
    if act_func is None or act_func == "none":
        return None
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    if act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    if act_func == "leaky_relu":
        return nn.LeakyReLU(inplace=inplace)
    if act_func == "tanh":
        return nn.Tanh()
    if act_func == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"unsupported activation: {act_func}")


class AverageMeter:
    """Running mean of streaming values."""

    def __init__(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
