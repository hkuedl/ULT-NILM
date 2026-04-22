"""Unsupervised / semi-supervised domain adaptation training loop.

Implements the LCT optimisation described in Section III.C of the paper:
the total loss combines a supervised task term on the source domain with
a multi-layer domain-alignment term (Sinkhorn + CORAL by default), with
parameter updates restricted to the lightweight residual transfer blocks
by the caller via ``NILMElasticModel.enable_domain_adaptation``.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def _dump_csv(path: str, rows: list[tuple[int, float]], header: tuple[str, str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def train_domain_adaptation(
    model,
    source_data: np.ndarray,
    source_labels: np.ndarray | None,
    target_data: np.ndarray,
    epochs: int = 1000,
    batch_size: int = 32,
    domain_loss_weight: float = 0.3,
    save_model: bool = True,
    save_period: int = 100,
    use_tqdm: bool = True,
    record_best_from: int = 100,
    early_stop_min_epochs: int = 200,
    early_stop_threshold_value: float = 1e-4,
    early_stop_threshold_steps: int = 250,
    num_workers: int = 4,
) -> tuple[float, float]:
    """Run the unsupervised LCT loop.

    Parameters
    ----------
    model
        A :class:`ult_nilm.model.NILMElasticModel` (or any supernet exposing
        ``forward_domain_adaptation``, ``enable_domain_adaptation`` and
        ``set_domain_loss_weight``).
    source_data, target_data
        Stacked input tensors in the format expected by the supernet
        (``[N, 1, 1, T]`` for the paper's Seq2Seq configuration).
    source_labels
        Per-sample ground-truth for the source domain. When ``None`` the
        loss is the domain-alignment term only.
    domain_loss_weight
        Weight ``lambda`` in ``L_total = L_sup + lambda * L_domain``
        (Eq. 27). The paper uses 0.3.
    """
    model.train_loss = []
    model.best_loss = float("inf")
    model.best_epoch = -1

    source_data_t = torch.tensor(source_data, dtype=torch.float32)
    target_data_t = torch.tensor(target_data, dtype=torch.float32)
    has_labels = source_labels is not None
    if has_labels:
        source_labels_t = torch.tensor(source_labels, dtype=torch.float32)
        source_dataset = TensorDataset(source_data_t, source_labels_t)
    else:
        source_dataset = TensorDataset(source_data_t)
    target_dataset = TensorDataset(target_data_t)

    n_source = len(source_data_t)
    n_target = len(target_data_t)
    adjusted_batch_size = min(
        batch_size,
        max(1, n_source // max(1, n_source // batch_size)),
        max(1, n_target // max(1, n_target // batch_size)),
    )

    source_loader = DataLoader(
        source_dataset,
        batch_size=adjusted_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=adjusted_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.enable_domain_adaptation(True)
    if has_labels:
        model.set_domain_loss_weight(domain_loss_weight)

    lr = model.optimizer.param_groups[0]["lr"]
    model.optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    epoch = 0
    start = time.perf_counter()
    model.train()

    loss_val = 0.0
    pbar = tqdm(range(epochs), desc="DA") if use_tqdm else None
    last_best_value: float | None = None
    last_best_step: int | None = None
    stop = False

    target_iter = iter(target_loader)
    while not stop:
        for batch_data in source_loader:
            try:
                batch_target_X = next(target_iter)[0]
            except StopIteration:
                target_iter = iter(target_loader)
                batch_target_X = next(target_iter)[0]

            batch_target_X = batch_target_X.to(model.device)
            model.optimizer.zero_grad()

            if has_labels:
                batch_source_X, batch_source_Y = batch_data
                batch_source_X = batch_source_X.to(model.device)
                batch_source_Y = batch_source_Y.to(model.device)
                source_pred, domain_loss = model.forward_domain_adaptation(batch_source_X, batch_target_X)
                if model.seq2seq:
                    batch_source_Y = batch_source_Y.unsqueeze(-1)
                task_loss = model.loss_fn(source_pred, batch_source_Y)
                loss = task_loss + model.domain_loss_weight * domain_loss
            else:
                batch_source_X = batch_data[0].to(model.device)
                _source_pred, domain_loss = model.forward_domain_adaptation(batch_source_X, batch_target_X)
                loss = domain_loss

            loss.backward()
            model.optimizer.step()
            loss_val = loss.item()
            model.train_loss.append((epoch, loss_val))

            if pbar is not None:
                pbar.set_description(f"{loss_val:.4f} (Best:{model.best_loss:.4f}@e{model.best_epoch})")
                pbar.update(1)

            epoch += 1
            if save_model and epoch % save_period == 0:
                model.save_checkpoint(epoch, cat="da", filename=f"period_e{epoch}_l{loss_val:.4f}")

            if loss_val < model.best_loss and epoch > record_best_from:
                model.best_loss = loss_val
                model.best_epoch = epoch
                if save_model:
                    model.save_checkpoint(epoch, cat="da", filename=f"best_e{epoch}_l{loss_val:.4f}")
                if (
                    epoch > early_stop_min_epochs
                    and last_best_value is not None
                    and last_best_value - loss_val < early_stop_threshold_value
                ):
                    stop = True
                last_best_value = loss_val
                last_best_step = epoch

            if (
                epoch > early_stop_min_epochs
                and last_best_step is not None
                and epoch - last_best_step >= early_stop_threshold_steps
            ):
                stop = True

            if epoch >= epochs or stop:
                break
        if epoch >= epochs or stop:
            break

    if pbar is not None:
        pbar.close()
    if save_model:
        model.save_checkpoint(epoch, cat="da", filename=f"final_e{epoch}_l{loss_val:.4f}")

    elapsed = time.perf_counter() - start
    model.enable_domain_adaptation(False)
    model.eval()
    _dump_csv(os.path.join(model.work_dir, "da_loss.csv"), model.train_loss, ("epoch", "loss"))
    return loss_val, elapsed
