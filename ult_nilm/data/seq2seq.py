"""Seq2Seq windowing for NILM aggregate signals.

The paper's default recipe: 6-second sampling, window length T=600,
strided by one sample to form the training set.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ult_nilm.data.preprocessing import _load_dataset, get_scaler


def build_seq2seq_dataset(
    houses: list[int],
    test_houses: list[int],
    dataset: str,
    test_dataset: str,
    device: str,
    device_test: str,
    w: int,
    nas: str = "drop",
    ds: int = 1,
    dataset_path: str = ".",
    standardize: bool = True,
    standardize_type: Literal["minmax", "standard", "robust"] = "standard",
    standardize_args: dict | None = None,
    use_uni_scaler: bool = False,
    specify_scaler: StandardScaler | None = None,
    specify_test_scaler: StandardScaler | None = None,
    limit_samples: int = -1,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    pd.DataFrame,
    pd.DataFrame | None,
    StandardScaler | None,
    StandardScaler | None,
]:
    df_device = _load_dataset(dataset, houses, ds, nas, dataset_path)
    df_device = df_device.rename(columns={"main": "power"})[["power", device]].dropna()
    if limit_samples > 0:
        df_device = df_device.head(limit_samples)

    if standardize:
        scaler = specify_scaler or get_scaler(standardize_type, standardize_args)
        if specify_scaler is None:
            df_device["power"] = scaler.fit_transform(df_device["power"].values.reshape(-1, 1)).reshape(-1)
        else:
            df_device["power"] = scaler.transform(df_device["power"].values.reshape(-1, 1)).reshape(-1)
        df_device[device] = scaler.transform(df_device[device].values.reshape(-1, 1)).reshape(-1)
    else:
        scaler = None

    x = np.lib.stride_tricks.sliding_window_view(df_device["power"].values, window_shape=w)
    y = np.lib.stride_tricks.sliding_window_view(df_device[device].values, window_shape=w)

    x_test = y_test = df_device_test = test_scaler = None
    if test_houses:
        df_device_test = _load_dataset(test_dataset, test_houses, ds, nas, dataset_path)
        df_device_test = df_device_test.rename(columns={"main": "power"})[["power", device_test]].dropna()
        if limit_samples > 0:
            df_device_test = df_device_test.head(limit_samples)

        if standardize:
            test_scaler = scaler
            if not use_uni_scaler:
                if specify_test_scaler is None:
                    test_scaler = get_scaler(standardize_type, standardize_args).fit(
                        df_device_test["power"].values.reshape(-1, 1)
                    )
                else:
                    test_scaler = specify_test_scaler
            df_device_test["power"] = test_scaler.transform(
                df_device_test["power"].values.reshape(-1, 1)
            ).reshape(-1)
            df_device_test[device_test] = test_scaler.transform(
                df_device_test[device_test].values.reshape(-1, 1)
            ).reshape(-1)

        x_test = np.lib.stride_tricks.sliding_window_view(df_device_test["power"].values, window_shape=w)
        y_test = np.lib.stride_tricks.sliding_window_view(df_device_test[device_test].values, window_shape=w)

        min_len = min(x.shape[0], y.shape[0])
        x = x[:min_len]
        y = y[:min_len]
        min_len_test = min(x_test.shape[0], y_test.shape[0])
        x_test = x_test[:min_len_test]
        y_test = y_test[:min_len_test]

    return x, y, x_test, y_test, df_device, df_device_test, scaler, test_scaler
