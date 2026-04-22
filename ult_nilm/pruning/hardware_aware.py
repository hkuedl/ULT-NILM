"""Hardware-aware subnet pruning with the penalty formulation of Eq. (10).

Given a memory budget ``M_edge`` and a validation objective
``L_val(alpha)``, we minimise

.. math::

    \\widetilde{\\mathcal{L}}_\\text{val}(\\alpha) = \\mathcal{L}_\\text{val}
        (\\alpha) + \\rho \\,\\max\\{0, M(\\alpha) - M_\\text{edge}\\}

Two optimisers are provided:

* ``"sample"`` (default) — uniform sampling over the search space, score
  the augmented loss on a held-out validation loader, and return the
  lowest-scoring candidate. Fast and deterministic under a fixed RNG.
* ``"evolution"`` — mutation-based refinement of the best seed sample.

The more sophisticated continuous relaxation (Gumbel-softmax) hinted at
in the paper is intentionally left as a simple extension point.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from ult_nilm.networks.elastic import NILMSupernet
from ult_nilm.pruning.lookup_table import MemoryLookupTable


@dataclass
class PruneResult:
    config: dict
    memory_bytes: int
    val_loss: float
    penalised_loss: float


def _evaluate_config(
    supernet: NILMSupernet,
    config: dict,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn,
    max_batches: int | None,
) -> float:
    supernet.set_active_subnet(ks=config["ks"], e=config["e"], d=config["d"])
    supernet.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            pred = supernet(x)
            if supernet.seq2seq and y.dim() == pred.dim() - 1:
                y = y.unsqueeze(-1)
            total += loss_fn(pred, y).item()
            n += 1
    return total / max(1, n)


def _penalise(val_loss: float, memory_bytes: int, memory_budget: int, rho: float) -> float:
    overflow = max(0, memory_bytes - memory_budget)
    return val_loss + rho * overflow


def _mutate(config: dict, supernet: NILMSupernet, mutation_rate: float = 0.25) -> dict:
    new_config = {k: list(v) for k, v in config.items()}
    for i in range(len(new_config["ks"])):
        if random.random() < mutation_rate:
            new_config["ks"][i] = random.choice(supernet.ks_list)
        if random.random() < mutation_rate:
            new_config["e"][i] = random.choice(supernet.expand_ratio_list)
    for i in range(len(new_config["d"])):
        if random.random() < mutation_rate:
            new_config["d"][i] = random.choice(supernet.depth_list)
    return new_config


def prune_subnet(
    supernet: NILMSupernet,
    lookup_table: MemoryLookupTable,
    val_loader: DataLoader,
    memory_budget: int,
    rho: float = 1e6,
    method: str = "sample",
    num_samples: int = 256,
    num_generations: int = 10,
    max_batches_per_eval: int | None = 4,
    loss_fn=None,
    seed: int | None = None,
) -> PruneResult:
    """Search for a subnet minimising ``L_val + rho * max(0, M - M_edge)``.

    Parameters
    ----------
    supernet
        Trained ``NILMSupernet``. The supernet's active configuration will
        be left set to the returned best configuration.
    lookup_table
        Pre-built :class:`MemoryLookupTable` used to score memory cheaply.
    val_loader
        Validation data; ``max_batches_per_eval`` caps how many batches
        are consumed per candidate to keep the search tractable.
    memory_budget
        ``M_edge`` in bytes.
    rho
        Penalty coefficient. Using a very large value (``>> 1 / min_loss``)
        effectively converts the soft penalty back into a hard constraint,
        matching the paper's ``rho >> 0`` convention.
    method
        ``"sample"`` for a one-shot uniform sweep, ``"evolution"`` for
        ``num_generations`` rounds of random mutation around the current
        best candidate.
    loss_fn
        Objective used for ``L_val``; defaults to MSE on the model output.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = next(supernet.parameters()).device
    loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss()
    original_depth = list(supernet.runtime_depth)

    def _score(config: dict) -> tuple[float, int, float]:
        memory = lookup_table.lookup_from_sample(config, reduction="sum")
        val_loss = _evaluate_config(
            supernet, config, val_loader, device, loss_fn, max_batches_per_eval
        )
        return _penalise(val_loss, memory, memory_budget, rho), memory, val_loss

    best: PruneResult | None = None
    candidates: list[dict] = [supernet.sample_active_subnet() for _ in range(num_samples)]
    for config in candidates:
        penalised, memory, val_loss = _score(config)
        if best is None or penalised < best.penalised_loss:
            best = PruneResult(config=copy.deepcopy(config), memory_bytes=memory,
                               val_loss=val_loss, penalised_loss=penalised)

    if method == "evolution" and best is not None:
        for _ in range(num_generations):
            mutated = _mutate(best.config, supernet)
            penalised, memory, val_loss = _score(mutated)
            if penalised < best.penalised_loss:
                best = PruneResult(config=copy.deepcopy(mutated), memory_bytes=memory,
                                   val_loss=val_loss, penalised_loss=penalised)
    elif method not in {"sample", "evolution"}:
        raise ValueError(f"unsupported method: {method}")

    assert best is not None
    supernet.set_active_subnet(ks=best.config["ks"], e=best.config["e"], d=best.config["d"])
    # Restore runtime depth list length if search altered the stage count.
    if len(supernet.runtime_depth) != len(original_depth):
        supernet.runtime_depth = original_depth
    return best
