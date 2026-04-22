"""Seq2Point windowing for NILM aggregate signals.

Each input sample is a length-``w`` window; the target is the central
sample of the matching device-level sequence. Provided alongside the
Seq2Seq loader so that both task framings in the paper can be reproduced.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ult_nilm.data.preprocessing import _load_dataset, get_scaler


def build_seq2point_dataset(
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

    if standardize:
        scaler = specify_scaler or get_scaler(standardize_type, standardize_args)
        if specify_scaler is None:
            df_device["power"] = scaler.fit_transform(df_device["power"].values.reshape(-1, 1)).reshape(-1)
        else:
            df_device["power"] = scaler.transform(df_device["power"].values.reshape(-1, 1)).reshape(-1)
        df_device[device] = scaler.transform(df_device[device].values.reshape(-1, 1)).reshape(-1)
    else:
        scaler = None

    if limit_samples > 0:
        df_device = df_device.head(limit_samples)

    power = np.pad(df_device["power"], w // 2, mode="constant", constant_values=0)
    device_power = np.pad(df_device[device], w // 2, mode="constant", constant_values=0)

    x = np.lib.stride_tricks.as_strided(
        power,
        shape=(len(power) - w + 1, w),
        strides=(power.strides[0], power.strides[0]),
    )
    y = np.zeros(len(df_device))
    for i in range(len(y)):
        y[i] = device_power[i + w // 2]

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

        power_test = np.pad(df_device_test["power"], w // 2, mode="constant", constant_values=0)
        device_power_test = np.pad(df_device_test[device_test], w // 2, mode="constant", constant_values=0)

        x_test = np.lib.stride_tricks.as_strided(
            power_test,
            shape=(len(power_test) - w + 1, w),
            strides=(power_test.strides[0], power_test.strides[0]),
        )
        y_test = np.zeros(len(df_device_test))
        for i in range(len(y_test)):
            y_test[i] = device_power_test[i + w // 2]

        if x.shape[0] > y.shape[0]:
            x = x[: y.shape[0], :]
        if x_test.shape[0] > y_test.shape[0]:
            x_test = x_test[: y_test.shape[0], :]

    return x, y, x_test, y_test, df_device, df_device_test, scaler, test_scaler
