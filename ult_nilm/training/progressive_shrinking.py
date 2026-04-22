"""Progressive shrinking training with Boltzmann configuration sampling.

Implements Eq. (3)-(6) from the paper. At stage ``s`` a feasibility mask
defined by the monotonically tightening budget

.. math::

    M_1 \\ge M_2 \\ge \\cdots \\ge M_{T_s}, \\quad
    L_1 \\ge L_2 \\ge \\cdots \\ge L_{T_s}

is combined with a Boltzmann sampler over the remaining configurations:

.. math::

    \\pi_s(\\alpha) \\propto
        \\exp(-\\beta_s \\mathcal{J}(\\alpha))
        \\,\\mathbb{I}[\\alpha \\in \\mathcal{A}_s]

where the cost is
``J(alpha) = omega_M M(alpha) + omega_L L(alpha)``.
The sampler uses a rejection/importance hybrid: candidate configurations
are drawn uniformly from the feasibility mask, and one is selected with
probability proportional to ``exp(-beta J)``. This matches Eq. (6) up to
the finite candidate budget, which is controlled by
``num_candidates_per_step``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ult_nilm.networks.elastic import NILMSupernet
from ult_nilm.pruning.lookup_table import MemoryLookupTable


@dataclass
class ShrinkingStage:
    """Per-stage budgets used to define the feasibility mask A_s (Eq. 5)."""

    memory_budget: float
    latency_budget: float = math.inf
    beta: float = 1.0
    duration_epochs: int = 10
    ks_candidates: list[int] | None = None
    expand_candidates: list[int] | None = None
    depth_candidates: list[int] | None = None


@dataclass
class ProgressiveShrinkingConfig:
    stages: list[ShrinkingStage]
    omega_memory: float = 1.0
    omega_latency: float = 0.0
    num_candidates_per_step: int = 32
    num_samples_per_subepoch: int = 1


def _sample_uniform_config(
    supernet: NILMSupernet,
    ks_candidates: list[int],
    expand_candidates: list[int],
    depth_candidates: list[int],
) -> dict:
    ks = [random.choice(ks_candidates) for _ in range(len(supernet.blocks) - 1)]
    expand = [random.choice(expand_candidates) for _ in range(len(supernet.blocks) - 1)]
    depth = [random.choice(depth_candidates) for _ in range(len(supernet.block_group_info))]
    return {"ks": ks, "e": expand, "d": depth}


def _estimate_latency(lookup_table: MemoryLookupTable, config: dict) -> float:
    """Latency proxy: total elements processed by active blocks.

    The paper uses MCU-measured latency where available; when that is not
    present (open-source scenario), we fall back to the peak-SRAM proxy
    which correlates well with per-sample runtime on embedded CNNs.
    """
    return float(lookup_table.lookup_from_sample(config, reduction="sram"))


def boltzmann_sample_config(
    supernet: NILMSupernet,
    lookup_table: MemoryLookupTable,
    stage: ShrinkingStage,
    omega_memory: float,
    omega_latency: float,
    num_candidates: int,
) -> dict:
    """Draw one configuration from the Boltzmann distribution of Eq. (6)."""
    ks_cand = stage.ks_candidates or supernet.ks_list
    expand_cand = stage.expand_candidates or supernet.expand_ratio_list
    depth_cand = stage.depth_candidates or supernet.depth_list

    feasible: list[tuple[dict, float, float]] = []
    attempts = 0
    max_attempts = max(num_candidates * 8, 128)
    while len(feasible) < num_candidates and attempts < max_attempts:
        attempts += 1
        candidate = _sample_uniform_config(supernet, ks_cand, expand_cand, depth_cand)
        memory = lookup_table.lookup_from_sample(candidate, reduction="sum")
        latency = _estimate_latency(lookup_table, candidate)
        if memory <= stage.memory_budget and latency <= stage.latency_budget:
            feasible.append((candidate, memory, latency))

    if not feasible:
        # Safety valve: fall back to the smallest achievable config via
        # minimal width/depth/kernel. This should only fire if the stage
        # budget is tighter than every reachable configuration.
        return _sample_uniform_config(
            supernet, [min(ks_cand)], [min(expand_cand)], [min(depth_cand)]
        )

    costs = np.array(
        [omega_memory * memory + omega_latency * latency for _, memory, latency in feasible],
        dtype=np.float64,
    )
    costs = costs - costs.min()  # stabilise exp
    weights = np.exp(-stage.beta * costs)
    weights = weights / weights.sum()
    index = np.random.choice(len(feasible), p=weights)
    return feasible[index][0]


@dataclass
class _TrainingBuffers:
    train_loss: list[tuple[int, float]] = field(default_factory=list)
    val_loss: list[tuple[int, float]] = field(default_factory=list)
    configs: dict[int, dict] = field(default_factory=dict)


class ProgressiveShrinkingTrainer:
    """Orchestrates multi-stage progressive shrinking of a ``NILMSupernet``."""

    def __init__(
        self,
        model,
        lookup_table: MemoryLookupTable,
        config: ProgressiveShrinkingConfig,
    ):
        self.model = model
        self.lookup_table = lookup_table
        self.config = config

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        eval_percentage: float = 0.1,
        batch_size: int = 256,
        save_checkpoint: bool = True,
        save_period: int = 1,
        use_tqdm: bool = True,
        num_workers: int = 4,
    ) -> tuple[float, float]:
        model = self.model
        model.train()

        val_split_idx = int(len(X) * (1 - eval_percentage))
        X_train, Y_train = X[:val_split_idx], Y[:val_split_idx]
        X_val, Y_val = X[val_split_idx:], Y[val_split_idx:]
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        )

        buffers = _TrainingBuffers()
        global_epoch = 0
        start = time.perf_counter()
        total_epochs = sum(s.duration_epochs for s in self.config.stages)
        pbar = tqdm(total=total_epochs, desc="PS") if use_tqdm else None
        loss_val = float("inf")
        val_loss = float("inf")

        for stage_idx, stage in enumerate(self.config.stages):
            for _sub in range(stage.duration_epochs):
                for _ in range(self.config.num_samples_per_subepoch):
                    net_config = boltzmann_sample_config(
                        model, self.lookup_table, stage,
                        omega_memory=self.config.omega_memory,
                        omega_latency=self.config.omega_latency,
                        num_candidates=self.config.num_candidates_per_step,
                    )
                    model.set_active_subnet(ks=net_config["ks"], e=net_config["e"], d=net_config["d"])
                    buffers.configs[global_epoch] = net_config

                    for batch_X, batch_Y in train_loader:
                        batch_X = batch_X.to(model.device)
                        batch_Y = batch_Y.to(model.device)
                        model.optimizer.zero_grad()
                        outputs = model.forward(batch_X)
                        if model.seq2seq:
                            batch_Y = batch_Y.unsqueeze(-1)
                        loss = model.loss_fn(outputs, batch_Y)
                        loss.backward()
                        model.optimizer.step()
                        loss_val = loss.item()
                        buffers.train_loss.append((global_epoch, loss_val))

                model.eval()
                total = 0.0
                with torch.no_grad():
                    for val_batch_X, val_batch_Y in val_loader:
                        val_batch_X = val_batch_X.to(model.device)
                        val_batch_Y = val_batch_Y.to(model.device)
                        preds = model.forward(val_batch_X)
                        if model.seq2seq:
                            val_batch_Y = val_batch_Y.unsqueeze(-1)
                        total += model.loss_fn(preds, val_batch_Y).item()
                val_loss = total / max(1, len(val_loader))
                buffers.val_loss.append((global_epoch, val_loss))
                model.train()

                if save_checkpoint and global_epoch % save_period == 0:
                    model.save_checkpoint(
                        global_epoch, cat="ps",
                        filename=f"stage{stage_idx}_e{global_epoch}_vl{val_loss:.4f}",
                    )

                if pbar is not None:
                    pbar.set_description(
                        f"stage {stage_idx + 1}/{len(self.config.stages)} "
                        f"TrL:{loss_val:.4f} VaL:{val_loss:.4f}"
                    )
                    pbar.update(1)
                global_epoch += 1

        if pbar is not None:
            pbar.close()
        if save_checkpoint:
            model.save_checkpoint(global_epoch, cat="ps", filename=f"final_e{global_epoch}_vl{val_loss:.4f}")

        elapsed = time.perf_counter() - start
        model.eval()
        self._dump_logs(buffers)
        return val_loss, elapsed

    # ------------------------------------------------------------ logging
    def _dump_logs(self, buffers: _TrainingBuffers) -> None:
        work_dir = self.model.work_dir
        with open(os.path.join(work_dir, "ps_train_loss.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss"])
            writer.writerows(buffers.train_loss)
        with open(os.path.join(work_dir, "ps_val_loss.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "val_loss"])
            writer.writerows(buffers.val_loss)
        with open(os.path.join(work_dir, "ps_net_configs.json"), "w") as f:
            json.dump(buffers.configs, f, indent=2)


def build_default_stages(
    supernet: NILMSupernet,
    lookup_table: MemoryLookupTable,
    num_stages: int = 4,
    final_memory_budget: float | None = None,
    epochs_per_stage: int = 10,
    beta_min: float = 1e-6,
    beta_max: float = 1.0,
) -> list[ShrinkingStage]:
    """Build a monotonically tightening schedule of shrinking stages.

    ``final_memory_budget`` defaults to the minimum achievable memory (the
    cheapest config in the lookup table); the first stage always uses the
    maximum memory to keep every configuration feasible. ``beta`` grows
    linearly from ``beta_min`` to ``beta_max`` across stages so early
    stages sample nearly uniformly and later stages concentrate on the
    cheapest configurations as prescribed by Eq. (6).
    """
    memories = [entry.total for entry in lookup_table.block_table.values()]
    min_mem = min(memories) + lookup_table.fixed_param_bytes + lookup_table.fixed_activation_bytes
    max_mem = max(memories) * len(supernet.blocks) + lookup_table.fixed_param_bytes + lookup_table.fixed_activation_bytes
    final_budget = final_memory_budget if final_memory_budget is not None else min_mem * 2

    stages: list[ShrinkingStage] = []
    for i in range(num_stages):
        t = i / max(1, num_stages - 1)
        budget = max_mem * (1 - t) + final_budget * t
        beta = beta_min + (beta_max - beta_min) * t
        stages.append(
            ShrinkingStage(
                memory_budget=budget,
                beta=beta,
                duration_epochs=epochs_per_stage,
            )
        )
    return stages
