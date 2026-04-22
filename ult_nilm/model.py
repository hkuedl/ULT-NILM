"""High-level NILM model wrapper.

``NILMElasticModel`` pairs the elastic supernet with training, evaluation,
and checkpoint utilities used by the scripts in ``scripts/``. The class
presents a minimal API surface so paper experiments can be reproduced
without the auxiliary infrastructure (experiment tracking, plotting,
deployment exporters) that was used during development.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ult_nilm.data.seq2point import build_seq2point_dataset
from ult_nilm.data.seq2seq import build_seq2seq_dataset
from ult_nilm.networks.elastic import NILMSupernet
from ult_nilm.utils.metrics import compute_metrics

FIXED_WIDTH = 599
FIXED_WIDTH_SEQ2SEQ = 600


def reload_dataset(
    params: dict,
    dataset_path: str,
    specify_scaler=None,
    specify_test_scaler=None,
    seq2seq: bool = True,
):
    """Build a paper-configuration dataset from the given parameter dict.

    ``params`` must contain the keys used by
    :func:`ult_nilm.data.seq2seq.build_seq2seq_dataset`
    (``dataset``, ``houses``, ``houses_test``, ``device``, ``device_test``,
    ``scale``, ``sr``, ``nas``).
    """
    builder = build_seq2seq_dataset if seq2seq else build_seq2point_dataset
    window = FIXED_WIDTH_SEQ2SEQ if seq2seq else FIXED_WIDTH
    x_train, y_train, x_test, y_test, df_train, df_test, scaler, test_scaler = builder(
        params["houses"],
        params["houses_test"],
        dataset=params["dataset"],
        test_dataset=params["dataset"],
        device=params["device"],
        device_test=params["device_test"],
        w=window,
        ds=params["sr"],
        nas=params["nas"],
        dataset_path=dataset_path,
        standardize=params["scale"],
        standardize_type="standard",
        use_uni_scaler=True,
        specify_scaler=specify_scaler,
        specify_test_scaler=specify_test_scaler,
        limit_samples=params.get("limit_samples", -1),
    )

    x_train = x_train.reshape(-1, 1, 1, window)
    if x_test is not None:
        x_test = x_test.reshape(-1, 1, 1, window)
    if seq2seq:
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, window)
        if y_test is not None and y_test.ndim == 1:
            y_test = y_test.reshape(-1, window)
    else:
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_test is not None and y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
    return x_train, y_train, x_test, y_test, df_train, df_test, scaler, test_scaler


class NILMElasticModel(NILMSupernet):
    """Elastic NILM network with training/evaluation plumbing."""

    def __init__(
        self,
        name: str = "unnamed",
        work_dir: str = "unnamed-result",
        bn_param: tuple[float, float] = (0.1, 1e-3),
        dropout_rate: float = 0.1,
        width_mult: float = 1.0,
        ks_list: list[int] = (3, 5),
        expand_ratio_list: list[int] = (2, 3),
        depth_list: list[int] = (2, 3),
        learning_rate: float = 1e-3,
        device: str = "cpu",
        n_classes: int = 1,
        first_stage_kernel_sizes: list[int] = (7, 5),
        first_stage_width: list[int] = (24, 48),
        base_stage_width: list[int] = (64, 128),
        last_stage_width: int = 256,
        first_stage_stride: int = 1,
        base_stage_stride: int = 1,
        act_func: str = "relu",
        use_frequency_features: bool = False,
        domain_adaptation_method: str = "sinkhorn_coral",
        seq2seq: bool = False,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 100,
        sinkhorn_coral_weight: float = 0.6,
        domain_loss_weight: float = 0.3,
        **_kwargs,
    ):
        self.name = name
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

        super().__init__(
            n_classes=n_classes,
            bn_param=bn_param,
            dropout_rate=dropout_rate,
            width_mult=width_mult,
            ks_list=list(ks_list),
            expand_ratio_list=list(expand_ratio_list),
            depth_list=list(depth_list),
            data_channels=1,
            first_stage_kernel_sizes=list(first_stage_kernel_sizes),
            first_stage_width=list(first_stage_width),
            first_stage_strides=[first_stage_stride] * len(first_stage_kernel_sizes),
            base_stage_width=list(base_stage_width),
            base_stage_strides=[base_stage_stride] * len(base_stage_width),
            last_stage_width=last_stage_width,
            act_func=act_func,
            use_frequency_features=use_frequency_features,
            domain_adaptation_method=domain_adaptation_method,
            seq2seq=seq2seq,
            seq_length=FIXED_WIDTH_SEQ2SEQ if seq2seq else FIXED_WIDTH,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            sinkhorn_coral_weight=sinkhorn_coral_weight,
            domain_loss_weight=domain_loss_weight,
        )

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate
        )
        self.device = device
        self.to(device)

        self.mconfig = {
            "bn_param": bn_param,
            "dropout_rate": dropout_rate,
            "width_mult": width_mult,
            "ks_list": list(ks_list),
            "expand_ratio_list": list(expand_ratio_list),
            "depth_list": list(depth_list),
            "learning_rate": learning_rate,
            "device": device,
            "n_classes": n_classes,
            "first_stage_kernel_sizes": list(first_stage_kernel_sizes),
            "first_stage_width": list(first_stage_width),
            "base_stage_width": list(base_stage_width),
            "last_stage_width": last_stage_width,
            "first_stage_stride": first_stage_stride,
            "base_stage_stride": base_stage_stride,
            "act_func": act_func,
            "use_frequency_features": use_frequency_features,
            "domain_adaptation_method": domain_adaptation_method,
            "seq2seq": seq2seq,
            "optimizer": "adamw",
        }
        self.checkpoints: dict[str, str] = {}

    # --------------------------------------------------------- lr/checkpoint
    def set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def save_checkpoint(self, tag: int | str, cat: str = "train", filename: str = "") -> str:
        path = os.path.join(self.work_dir, "checkpoints", cat)
        os.makedirs(path, exist_ok=True)
        if not filename:
            filename = f"checkpoint_{tag}.pth"
        if not filename.endswith(".pth"):
            filename = f"{filename}.pth"
        save_path = os.path.join(path, filename)
        torch.save(self.state_dict(), save_path)
        self.checkpoints[f"{cat}_{tag}"] = save_path
        return save_path

    def save_model(self, filename: str) -> None:
        torch.save(self.state_dict(), filename)

    def load_model(self, filename: str) -> None:
        self.load_state_dict(torch.load(filename, map_location=self.device, weights_only=True))

    # ----------------------------------------------------------- training
    def train_supervised(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        eval_percentage: float = 0.2,
        eval_period_div: int = 2,
        epochs: int = 100,
        batch_size: int = 256,
        save_checkpoint: bool = True,
        save_period: int = 100,
        use_tqdm: bool = True,
        record_best_from: int = 100,
        early_stop_threshold_loss: float | None = 1e-4,
        early_stop_threshold_evals: int | None = 5,
        num_workers: int = 4,
    ) -> tuple[float, float]:
        self.train_loss: list[tuple[int, float]] = []
        self.val_loss_history: list[tuple[int, float]] = []
        self.best_loss = float("inf")
        self.best_epoch = -1

        val_split_idx = int(len(X) * (1 - eval_percentage))
        X_train, Y_train = X[:val_split_idx], Y[:val_split_idx]
        X_val, Y_val = X[val_split_idx:], Y[val_split_idx:]

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        eval_period = max(1, len(train_loader) // eval_period_div)

        step = 0
        start = time.perf_counter()
        loss_val = float("inf")
        vali_loss_val = float("inf")
        total_steps = epochs * len(train_loader)
        pbar = tqdm(range(total_steps), desc="train") if use_tqdm else None

        last_best_eval_loss: float | None = None
        last_best_eval_step: int | None = None
        stop = False
        self.train()

        while not stop:
            for batch_X, batch_Y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.forward(batch_X)
                if self.seq2seq:
                    batch_Y = batch_Y.unsqueeze(-1)
                loss = self.loss_fn(outputs, batch_Y)
                loss.backward()
                self.optimizer.step()
                loss_val = loss.item()
                self.train_loss.append((step, loss_val))

                if step % eval_period == 0 and step > 0:
                    vali_loss_val = self._run_validation(val_loader)
                    self.val_loss_history.append((step, vali_loss_val))

                    if step > record_best_from:
                        if save_checkpoint:
                            self.save_checkpoint(
                                step, cat="train", filename=f"eval_e{step}_vl{vali_loss_val:.4f}"
                            )
                        if vali_loss_val < self.best_loss:
                            self.best_loss = vali_loss_val
                            self.best_epoch = step
                            if (
                                last_best_eval_loss is not None
                                and early_stop_threshold_loss is not None
                                and last_best_eval_loss - vali_loss_val < early_stop_threshold_loss
                            ):
                                stop = True
                            last_best_eval_loss = vali_loss_val
                            last_best_eval_step = step

                    if (
                        last_best_eval_step is not None
                        and early_stop_threshold_evals is not None
                        and step - last_best_eval_step >= early_stop_threshold_evals * eval_period
                    ):
                        stop = True

                if pbar is not None:
                    pbar.set_description(
                        f"TrL:{loss_val:.4f} VaL:{vali_loss_val:.4f} (Best:{self.best_loss:.4f}@e{self.best_epoch})"
                    )
                    pbar.update(1)

                step += 1
                if save_checkpoint and step % save_period == 0:
                    self.save_checkpoint(step, cat="train", filename=f"period_e{step}_l{loss_val:.4f}")
                if step >= total_steps or stop:
                    break
            if step >= total_steps or stop:
                break

        if pbar is not None:
            pbar.close()
        if save_checkpoint:
            self.save_checkpoint(step, cat="train", filename=f"final_e{step}_l{loss_val:.4f}")

        self.eval()
        elapsed = time.perf_counter() - start
        self._dump_loss_csv("train_loss.csv", self.train_loss)
        self._dump_loss_csv("val_loss.csv", self.val_loss_history)
        return loss_val, elapsed

    def _run_validation(self, val_loader: DataLoader) -> float:
        self.eval()
        total = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)
                outputs = self.forward(batch_X)
                if self.seq2seq:
                    batch_Y = batch_Y.unsqueeze(-1)
                total += self.loss_fn(outputs, batch_Y).item()
        self.train()
        return total / max(1, len(val_loader))

    def _dump_loss_csv(self, filename: str, rows: list[tuple[int, float]]) -> None:
        path = os.path.join(self.work_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            writer.writerows(rows)

    # --------------------------------------------------------- inference
    def predict(
        self, X: np.ndarray, batch_size: int = 256, num_workers: int = 4
    ) -> tuple[np.ndarray, float]:
        self.eval()
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            start = time.perf_counter()
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                predictions.append(self.forward(batch_X).cpu().numpy())
            latency = time.perf_counter() - start
        return np.concatenate(predictions, axis=0), latency

    def test(
        self,
        Xt: np.ndarray,
        Yt: np.ndarray,
        test_data: pd.DataFrame,
        scaler: Optional[MinMaxScaler] = None,
        batch_size: int = 256,
        override_thres: float | None = None,
        min_thres: float | None = None,
        subdir: str = "",
        average: str = "binary",
    ) -> dict:
        out_dir = os.path.join(self.work_dir, subdir) if subdir else self.work_dir
        os.makedirs(out_dir, exist_ok=True)

        is_seq2seq = bool(self.seq2seq)
        if not is_seq2seq and Yt.ndim == 1:
            Yt = Yt.reshape(-1, 1)
        if is_seq2seq:
            Xt = Xt[::FIXED_WIDTH_SEQ2SEQ, :, :, :]
            Yt = Yt[::FIXED_WIDTH_SEQ2SEQ, :]

        predictions, latency = self.predict(Xt, batch_size)
        X_raw = test_data["power"].to_numpy().reshape(-1, 1)

        if is_seq2seq:
            predictions = np.concatenate(predictions, axis=0).reshape(-1, 1)
            Yt = np.concatenate(Yt, axis=0).reshape(-1, 1)
            m = min(len(X_raw), len(predictions), len(Yt))
            predictions = predictions[:m]
            Yt = Yt[:m]
            X_raw = X_raw[:m]

        if scaler is not None:
            predictions = scaler.inverse_transform(predictions)
            Yt = scaler.inverse_transform(Yt)
            X_raw = scaler.inverse_transform(X_raw)

        predictions[predictions < 0] = 0
        metrics = compute_metrics(
            Yt, predictions, override_thres, min_thres, average=average, seq2seq=is_seq2seq
        )
        metrics["Latency"] = latency

        df = pd.DataFrame(
            {
                "time": np.arange(X_raw.shape[0], dtype=np.int32),
                "aggregate": X_raw.flatten(),
                "ground_truth": Yt.flatten(),
                "result": predictions.flatten(),
            }
        )
        df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    # --------------------------------------------------------- subnet utils
    def set_max_constraint(self) -> None:
        self.set_constraint(include_list=self.mconfig["ks_list"], constraint_type="kernel_size")
        self.set_constraint(include_list=self.mconfig["expand_ratio_list"], constraint_type="expand_ratio")
        self.set_constraint(include_list=self.mconfig["depth_list"], constraint_type="depth")

    # ---------------------------------------------------------- stdout log
    def log(self, msg: str) -> None:
        print(f"[{self.name}] {msg}", file=sys.stderr, flush=True)
