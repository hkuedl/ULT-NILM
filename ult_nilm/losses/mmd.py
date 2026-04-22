"""Maximum-Mean-Discrepancy loss with a multi-scale RBF kernel.

Retained for the ablation study comparing Sinkhorn-based alignment against
kernel-MMD alignment (Table VI in the paper). Not part of the default
training recipe.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _gaussian_kernel(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
) -> torch.Tensor:
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    l2 = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2.data) / (n_samples**2 - n_samples)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    return sum(torch.exp(-l2 / (bandwidth * kernel_mul**i)) for i in range(kernel_num))


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5, fix_sigma: float | None = None):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.size(0) != target.size(0):
            raise ValueError(
                f"source batch size ({source.size(0)}) must match target batch size ({target.size(0)})"
            )
        batch_size = source.size(0)
        kernels = _gaussian_kernel(
            source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma
        )
        xx = kernels[:batch_size, :batch_size]
        yy = kernels[batch_size:, batch_size:]
        xy = kernels[:batch_size, batch_size:]
        yx = kernels[batch_size:, :batch_size]
        return torch.mean(xx + yy - xy - yx)
