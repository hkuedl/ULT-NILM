"""Shared helpers for the command-line entry scripts."""

from __future__ import annotations

import argparse
import os
import random
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_appliance(cfg: dict, alias: str, dataset: str) -> str:
    table = cfg.get("appliance_aliases", {}).get(alias)
    if table is None:
        raise KeyError(f"unknown appliance alias '{alias}'")
    if dataset not in table:
        raise KeyError(f"alias '{alias}' has no mapping for dataset '{dataset}'")
    return table[dataset]


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(preferred: str | None) -> str:
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def dataset_params_from_scenario(
    cfg: dict, scenario_name: str, appliance_alias: str, limit_samples: int = -1
) -> tuple[dict, dict | None]:
    """Return (source_params, target_params_or_None) dicts for ``build_*_dataset``."""
    scenario = cfg["scenarios"][scenario_name]
    source = scenario["source"]
    source_params = {
        "dataset": source["dataset"],
        "houses": source["houses"],
        "houses_test": source["houses_test"],
        "device": resolve_appliance(cfg, appliance_alias, source["dataset"]),
        "device_test": resolve_appliance(cfg, appliance_alias, source["dataset"]),
        "scale": True,
        "sr": 6,
        "nas": "drop",
        "limit_samples": limit_samples,
    }
    target = scenario.get("target")
    if target is None:
        return source_params, None
    target_params = {
        "dataset": target["dataset"],
        "houses": target["houses"],
        "houses_test": target["houses_test"],
        "device": resolve_appliance(cfg, appliance_alias, target["dataset"]),
        "device_test": resolve_appliance(cfg, appliance_alias, target["dataset"]),
        "scale": True,
        "sr": 6,
        "nas": "drop",
        "limit_samples": limit_samples,
    }
    return source_params, target_params


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset-path", type=str, default=None, help="Override dataset root")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cpu / cuda)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--work-dir", type=str, default="runs/default")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)


def resolve_dataset_path(cfg: dict, override: str | None) -> str:
    path = override or cfg.get("dataset_path")
    if not path or path == "/path/to/datasets":
        raise SystemExit(
            "dataset path is unset; edit configs/default.yaml or pass --dataset-path"
        )
    if not os.path.isdir(path):
        raise SystemExit(f"dataset path does not exist: {path}")
    return path


def build_model(cfg: dict, args: argparse.Namespace):
    from ult_nilm.model import NILMElasticModel

    supernet_cfg = cfg["supernet"]
    training_cfg = cfg["training"]
    da_cfg = cfg.get("domain_adaptation", {})
    device = pick_device(args.device)
    model = NILMElasticModel(
        name=os.path.basename(args.work_dir.rstrip("/")) or "ult_nilm",
        work_dir=args.work_dir,
        bn_param=tuple(supernet_cfg["bn_param"]),
        dropout_rate=supernet_cfg["dropout_rate"],
        width_mult=supernet_cfg["width_mult"],
        ks_list=supernet_cfg["ks_list"],
        expand_ratio_list=supernet_cfg["expand_ratio_list"],
        depth_list=supernet_cfg["depth_list"],
        learning_rate=training_cfg["learning_rate"],
        device=device,
        n_classes=supernet_cfg["n_classes"],
        first_stage_kernel_sizes=supernet_cfg["first_stage_kernel_sizes"],
        first_stage_width=supernet_cfg["first_stage_width"],
        base_stage_width=supernet_cfg["base_stage_width"],
        last_stage_width=supernet_cfg["last_stage_width"],
        first_stage_stride=supernet_cfg["first_stage_stride"],
        base_stage_stride=supernet_cfg["base_stage_stride"],
        act_func=supernet_cfg["act_func"],
        use_frequency_features=supernet_cfg["use_frequency_features"],
        domain_adaptation_method=supernet_cfg["domain_adaptation_method"],
        seq2seq=supernet_cfg["seq2seq"],
        sinkhorn_epsilon=da_cfg.get("sinkhorn_epsilon", 0.1),
        sinkhorn_iterations=da_cfg.get("sinkhorn_iterations", 100),
        sinkhorn_coral_weight=da_cfg.get("sinkhorn_coral_weight", 0.6),
        domain_loss_weight=da_cfg.get("domain_loss_weight", 0.3),
    )
    return model
