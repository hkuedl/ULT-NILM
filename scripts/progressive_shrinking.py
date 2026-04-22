"""Progressive shrinking training driven by Boltzmann configuration sampling.

Assumes ``scripts/train_supernet.py`` has produced a warm-start checkpoint
at the maximum configuration; loads that checkpoint, builds the memory
lookup table, and runs the multi-stage shrinking schedule defined in
``ult_nilm.training.progressive_shrinking``.
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
from ult_nilm.pruning.lookup_table import MemoryLookupTable
from ult_nilm.training.progressive_shrinking import (
    ProgressiveShrinkingConfig,
    ProgressiveShrinkingTrainer,
    build_default_stages,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive shrinking for ULT-NILM")
    add_common_arguments(parser)
    parser.add_argument("--scenario", required=True, choices=["intra_redd", "intra_ukdale", "redd_to_ukdale", "ukdale_to_redd"])
    parser.add_argument("--appliance", required=True, choices=["washing_machine", "dishwasher", "fridge", "microwave"])
    parser.add_argument("--checkpoint", required=True, help="Warm-start supernet checkpoint (.pth)")
    parser.add_argument("--lookup-table-path", type=str, default=None)
    parser.add_argument("--final-memory-budget", type=int, default=None, help="bytes")
    parser.add_argument("--num-stages", type=int, default=None)
    parser.add_argument("--epochs-per-stage", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    dataset_path = resolve_dataset_path(cfg, args.dataset_path)

    source_params, _ = dataset_params_from_scenario(cfg, args.scenario, args.appliance)
    x_train, y_train, _x_test, _y_test, *_ = reload_dataset(
        source_params, dataset_path=dataset_path, seq2seq=cfg["supernet"]["seq2seq"]
    )

    model = build_model(cfg, args)
    model.load_model(args.checkpoint)

    lookup = MemoryLookupTable(model)
    if args.lookup_table_path and os.path.exists(args.lookup_table_path):
        lookup.load(args.lookup_table_path)
    else:
        lookup.build_from_supernet(model)
        if args.lookup_table_path:
            lookup.save(args.lookup_table_path)

    ps_cfg = cfg["progressive_shrinking"]
    stages = build_default_stages(
        model,
        lookup,
        num_stages=args.num_stages or ps_cfg["num_stages"],
        epochs_per_stage=args.epochs_per_stage or ps_cfg["epochs_per_stage"],
        final_memory_budget=args.final_memory_budget,
        beta_min=ps_cfg["beta_min"],
        beta_max=ps_cfg["beta_max"],
    )
    trainer = ProgressiveShrinkingTrainer(
        model,
        lookup,
        ProgressiveShrinkingConfig(
            stages=stages,
            omega_memory=ps_cfg["omega_memory"],
            omega_latency=ps_cfg["omega_latency"],
            num_candidates_per_step=ps_cfg["num_candidates_per_step"],
        ),
    )
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    final_loss, elapsed = trainer.train(
        x_train,
        y_train,
        eval_percentage=cfg["training"]["eval_percentage"],
        batch_size=batch_size,
        num_workers=args.num_workers,
    )
    model.save_model(os.path.join(model.work_dir, "supernet_ps.pth"))
    print(f"progressive shrinking finished: val_loss={final_loss:.4f} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
