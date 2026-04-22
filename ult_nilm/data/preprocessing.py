"""Dataset loading and preprocessing helpers for REDD and UK-DALE."""

from __future__ import annotations

import os
from typing import Literal

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def _downsample(house: pd.DataFrame, dataset: str, frequency: int = 10) -> pd.DataFrame:
    """Resample a per-house dataframe to ``frequency`` second intervals."""
    if dataset == "redd":
        house.index = pd.to_timedelta(house.time, unit="s")
        house = house.drop(columns=["time"])
        house_dsed = house.resample(f"{frequency}S").mean().dropna()
    elif dataset == "ukdale":
        house_dsed = house.resample(f"{frequency}S").mean().dropna()
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    if len(house_dsed) % 2 != 0:
        house_dsed = house_dsed.iloc[:-1]
    return house_dsed


def get_scaler(
    standardize_type: Literal["minmax", "standard", "robust"],
    standardize_args: dict | None = None,
):
    args = standardize_args or {}
    if standardize_type == "minmax":
        return MinMaxScaler(**args)
    if standardize_type == "standard":
        return StandardScaler(**args)
    if standardize_type == "robust":
        return RobustScaler(**args)
    raise ValueError(f"unsupported scaler: {standardize_type}")


def process_redd(
    houses: list[int], ds: int, nas: str, dataset_path: str
) -> pd.DataFrame:
    """Load and concatenate selected REDD houses."""
    house_dfs: list[pd.DataFrame] = []
    all_columns: set[str] = set()

    for i in houses:
        house = pd.read_csv(os.path.join(dataset_path, "redd", f"house_{i}.csv"))
        if house.columns[0] == "Unnamed: 0":
            house = house.rename(columns={"Unnamed: 0": "time"})
        if ds > 3:
            house = _downsample(house, frequency=ds, dataset="redd").reset_index(drop=True)
        elif nas == "interpolate":
            house = house.interpolate(method="linear", limit=120)
        elif nas == "drop":
            house = house.dropna()
        house = house.dropna()
        if not house.empty:
            house_dfs.append(house)
            all_columns.update(house.columns)

    for idx, house_df in enumerate(house_dfs):
        for col in all_columns - set(house_df.columns):
            house_df[col] = 0
        house_dfs[idx] = house_df

    return pd.concat(house_dfs, ignore_index=True)


def process_ukdale(houses: list[int], ds: int, dataset_path: str) -> pd.DataFrame:
    """Load and concatenate selected UK-DALE buildings."""
    house_dfs: list[pd.DataFrame] = []
    all_columns: set[str] = set()

    for i in houses:
        house = pd.read_feather(os.path.join(dataset_path, "ukdale", f"building_{i}.feather"))
        if "mains" in house.columns:
            house = house.rename(columns={"mains": "main"})
        if ds > 6:
            house = _downsample(house, frequency=ds, dataset="ukdale").reset_index(drop=True)
        house_dfs.append(house)
        all_columns.update(house.columns)

    for idx, house_df in enumerate(house_dfs):
        for col in all_columns - set(house_df.columns):
            house_df[col] = 0
        house_dfs[idx] = house_df

    return pd.concat(house_dfs, ignore_index=True)


def _load_dataset(dataset: str, houses: list[int], ds: int, nas: str, dataset_path: str) -> pd.DataFrame:
    if dataset == "redd":
        return process_redd(houses, ds, nas, dataset_path)
    if dataset == "ukdale":
        return process_ukdale(houses, ds, dataset_path)
    raise ValueError(f"unsupported dataset: {dataset}")
