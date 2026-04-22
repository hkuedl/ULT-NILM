"""Baseline supervised training of the elastic supernet.

Trains the maximum-configuration subnet on a single appliance + scenario
to initialise the shared weights before progressive shrinking.
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Supervised training of the ULT-NILM supernet")
    add_common_arguments(parser)
    parser.add_argument("--scenario", required=True, choices=["intra_redd", "intra_ukdale", "redd_to_ukdale", "ukdale_to_redd"])
    parser.add_argument("--appliance", required=True, choices=["washing_machine", "dishwasher", "fridge", "microwave"])
    parser.add_argument("--limit-samples", type=int, default=-1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    dataset_path = resolve_dataset_path(cfg, args.dataset_path)

    source_params, _target = dataset_params_from_scenario(
        cfg, args.scenario, args.appliance, args.limit_samples
    )
    x_train, y_train, _x_test, _y_test, _df_train, _df_test, _scaler, _test_scaler = reload_dataset(
        source_params, dataset_path=dataset_path, seq2seq=cfg["supernet"]["seq2seq"]
    )

    model = build_model(cfg, args)
    model.set_max_net()
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    final_loss, elapsed = model.train_supervised(
        x_train,
        y_train,
        eval_percentage=cfg["training"]["eval_percentage"],
        eval_period_div=cfg["training"]["eval_period_div"],
        epochs=epochs,
        batch_size=batch_size,
        early_stop_threshold_evals=cfg["training"]["early_stop_threshold_evals"],
        num_workers=args.num_workers,
    )
    model.save_model(os.path.join(model.work_dir, "supernet_maxnet.pth"))
    print(f"training finished: loss={final_loss:.4f} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
