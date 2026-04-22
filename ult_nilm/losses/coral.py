"""CORAL loss for second-order domain alignment.

Matches the Frobenius distance between the source and target feature
covariance matrices, as defined in Eq. (23)-(25) of the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _covariance(features: torch.Tensor) -> torch.Tensor:
    n = features.size(0)
    if n < 2:
        raise ValueError("CORAL loss requires batch size >= 2")
    ones = torch.ones((1, n), device=features.device, dtype=features.dtype)
    mean_outer = (ones @ features).t() @ (ones @ features)
    return (features.t() @ features - mean_outer / n) / (n - 1)


class CORALLoss(nn.Module):
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.size(0) != target.size(0):
            raise ValueError(
                f"source batch size ({source.size(0)}) must match target batch size ({target.size(0)})"
            )
        d = source.size(1)
        cs = _covariance(source)
        ct = _covariance(target)
        return torch.norm(cs - ct, p="fro") ** 2 / (4 * d * d)
