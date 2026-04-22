"""Sinkhorn divergence for unsupervised domain adaptation.

Implements Algorithm 1 from the paper: the entropy-regularised optimal
transport cost between two empirical distributions is computed via the
Sinkhorn-Knopp iteration, and the divergence is debiased by subtracting
the self-transport terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SinkhornLoss(nn.Module):
    """Entropy-regularised optimal-transport divergence between two batches.

    Parameters
    ----------
    epsilon
        Regularisation strength ``eta`` in the paper. Larger values yield
        smoother transport plans but bias the estimate. The paper uses 0.1.
    num_iterations
        Sinkhorn-Knopp iteration count ``J`` in the paper. Default 100.
    debiased
        If True, return the Sinkhorn divergence
        ``S(mu, nu) = W(mu, nu) - 0.5 W(mu, mu) - 0.5 W(nu, nu)``;
        if False, return the raw Sinkhorn transport cost ``W(mu, nu)``.
    reduction
        ``"mean"`` divides the total cost by the batch size; ``"sum"`` keeps
        the raw accumulated cost.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        num_iterations: int = 100,
        debiased: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.debiased = debiased
        self.reduction = reduction
        self.stability_eps = 1e-8

    def _transport_cost(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        n1, n2 = x1.size(0), x2.size(0)
        mu = torch.full((n1,), 1.0 / n1, device=x1.device, dtype=x1.dtype)
        nu = torch.full((n2,), 1.0 / n2, device=x2.device, dtype=x2.dtype)

        # Squared Euclidean cost matrix.
        cost_matrix = torch.cdist(x1, x2, p=2) ** 2

        kernel = torch.exp(-cost_matrix / self.epsilon)
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)
        for _ in range(self.num_iterations):
            v = nu / (kernel.t() @ u + self.stability_eps)
            u = mu / (kernel @ v + self.stability_eps)

        plan = u.unsqueeze(1) * kernel * v.unsqueeze(0)
        return torch.sum(plan * cost_matrix)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.size(0) != target.size(0):
            raise ValueError(
                f"source batch size ({source.size(0)}) must match target batch size ({target.size(0)})"
            )

        w_st = self._transport_cost(source, target)
        if self.debiased:
            w_ss = self._transport_cost(source, source)
            w_tt = self._transport_cost(target, target)
            cost = w_st - 0.5 * w_ss - 0.5 * w_tt
        else:
            cost = w_st

        if self.reduction == "mean":
            return cost / source.size(0)
        if self.reduction == "sum":
            return cost
        raise ValueError(f"unsupported reduction: {self.reduction}")
