"""Compute MAE and SAE on the held-out test split.

Reproduces the reporting format used in Table III: single-device, single-
scenario evaluation with aggregate de-normalisation so the reported
metrics are in watts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._common import (
    add_common_arguments,
    build_model,
    dataset_params_from_scenario,
    load_config,
    resolve_dataset_path,
    set_seed,
)
from ult_nilm.model import reload_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained ULT-NILM model")
    add_common_arguments(parser)
    parser.add_argument("--scenario", required=True, choices=["intra_redd", "intra_ukdale", "redd_to_ukdale", "ukdale_to_redd"])
    parser.add_argument("--appliance", required=True, choices=["washing_machine", "dishwasher", "fridge", "microwave"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subnet-config", type=str, default=None,
                        help="Optional pruned_subnet.json (set_active_subnet before evaluation)")
    parser.add_argument("--min-thres", type=float, default=20.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    dataset_path = resolve_dataset_path(cfg, args.dataset_path)

    source_params, target_params = dataset_params_from_scenario(cfg, args.scenario, args.appliance)
    eval_params = target_params or source_params
    _x_train, _y_train, x_test, y_test, _df_train, df_test, _scaler, test_scaler = reload_dataset(
        eval_params, dataset_path=dataset_path, seq2seq=cfg["supernet"]["seq2seq"]
    )

    model = build_model(cfg, args)
    model.load_model(args.checkpoint)
    model.eval()

    if args.subnet_config and os.path.exists(args.subnet_config):
        with open(args.subnet_config, "r") as f:
            payload = json.load(f)
        c = payload["config"]
        model.set_active_subnet(ks=c["ks"], e=c["e"], d=c["d"])
    else:
        model.set_max_net()

    metrics = model.test(
        x_test,
        y_test,
        df_test,
        scaler=test_scaler,
        batch_size=args.batch_size or cfg["training"]["batch_size"],
        min_thres=args.min_thres,
        subdir=f"eval_{args.scenario}_{args.appliance}",
    )
    print(json.dumps({"MAE": metrics["MAE"], "SAE": metrics["SAE"]}, indent=2))


if __name__ == "__main__":
    main()
