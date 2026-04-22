"""Evaluation metrics for NILM: MAE and SAE as defined in the paper, plus a
handful of auxiliary classification metrics used during analysis."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)


def _threshold(consumption: np.ndarray, thres_val: float) -> np.ndarray:
    return np.where(consumption > thres_val, 1, 0)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    override_thres: float | None = None,
    min_thres: float | None = None,
    average: str = "binary",
    sae_window: int = 600,
    seq2seq: bool = False,
) -> dict[str, float]:
    """Compute NILM disaggregation metrics.

    ``sae_window`` is the paper's ``T_eval`` (600 samples = one hour at
    the 6 s sampling period used in Table III).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if seq2seq:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
    else:
        assert y_true.ndim <= 1 or y_true.shape[1] == 1, "multi-label not supported"
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()

    mse = mean_squared_error(y_true, y_pred)
    mae = float(np.abs(y_true - y_pred).mean())
    rmse = math.sqrt(mse)

    sae_blocks = y_true.shape[0] // sae_window
    if sae_blocks > 0:
        saes = []
        for i in range(sae_blocks):
            start = -(i + 1) * sae_window
            end = -i * sae_window if i > 0 else None
            saes.append(abs(np.sum(y_true[start:end]) - np.sum(y_pred[start:end])) / sae_window)
        sae = float(np.mean(saes))
        sae_max = float(np.max(saes))
        sae_min = float(np.min(saes))
    else:
        sae = sae_max = sae_min = float("nan")

    if override_thres is None:
        thres = (y_true.mean() - y_true.min()) / 4 + y_true.min()
        if min_thres is not None:
            thres = max(thres, min_thres)
    else:
        thres = override_thres

    y_pred_thres = _threshold(y_pred, thres)
    y_true_thres = _threshold(y_true, thres)

    precision = precision_score(y_true_thres, y_pred_thres, average=average, zero_division=1)
    recall = recall_score(y_true_thres, y_pred_thres, average=average, zero_division=1)
    acc = accuracy_score(y_true_thres, y_pred_thres)
    f1 = f1_score(y_true_thres, y_pred_thres, average=average, zero_division=1)

    return {
        "MAE": mae,
        "SAE": sae,
        "SAE_max": sae_max,
        "SAE_min": sae_min,
        "MSE": float(mse),
        "RMSE": rmse,
        "Precision": float(precision),
        "Accuracy": float(acc),
        "Recall": float(recall),
        "F1Score": float(f1),
        "Threshold": float(thres),
    }
