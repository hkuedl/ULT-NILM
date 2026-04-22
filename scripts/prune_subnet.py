"""Hardware-aware subnet pruning.

Builds the memory lookup table for a trained supernet and runs the
penalty-driven search of ``ult_nilm.pruning.hardware_aware.prune_subnet``
to find the best configuration satisfying a given memory budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

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
from ult_nilm.pruning.hardware_aware import prune_subnet
from ult_nilm.pruning.lookup_table import MemoryLookupTable


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware-aware subnet pruning")
    add_common_arguments(parser)
    parser.add_argument("--scenario", required=True, choices=["intra_redd", "intra_ukdale", "redd_to_ukdale", "ukdale_to_redd"])
    parser.add_argument("--appliance", required=True, choices=["washing_machine", "dishwasher", "fridge", "microwave"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--memory-budget", required=True, type=int, help="M_edge in bytes")
    parser.add_argument("--rho", type=float, default=1e6, help="Penalty coefficient in Eq. 10")
    parser.add_argument("--method", default="sample", choices=["sample", "evolution"])
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--num-generations", type=int, default=10)
    parser.add_argument("--max-batches-per-eval", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    dataset_path = resolve_dataset_path(cfg, args.dataset_path)

    source_params, _ = dataset_params_from_scenario(cfg, args.scenario, args.appliance)
    x_train, y_train, _x_test, _y_test, *_ = reload_dataset(
        source_params, dataset_path=dataset_path, seq2seq=cfg["supernet"]["seq2seq"]
    )

    # Use the tail 10% of training data as a held-out validation set for scoring.
    split = int(len(x_train) * 0.9)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_train[split:], dtype=torch.float32), torch.tensor(y_train[split:], dtype=torch.float32)),
        batch_size=args.batch_size or cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(cfg, args)
    model.load_model(args.checkpoint)
    model.eval()

    lookup = MemoryLookupTable(model).build_from_supernet(model)
    result = prune_subnet(
        model,
        lookup,
        val_loader,
        memory_budget=args.memory_budget,
        rho=args.rho,
        method=args.method,
        num_samples=args.num_samples,
        num_generations=args.num_generations,
        max_batches_per_eval=args.max_batches_per_eval,
        seed=args.seed,
    )

    out_path = os.path.join(model.work_dir, "pruned_subnet.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "memory_budget_bytes": args.memory_budget,
                "rho": args.rho,
                "memory_bytes": result.memory_bytes,
                "val_loss": result.val_loss,
                "penalised_loss": result.penalised_loss,
                "config": result.config,
            },
            f,
            indent=2,
        )
    print(
        f"pruned subnet: memory={result.memory_bytes} B  val_loss={result.val_loss:.4f}  "
        f"penalised_loss={result.penalised_loss:.4f}\nwritten to {out_path}"
    )


if __name__ == "__main__":
    main()
