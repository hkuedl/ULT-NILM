"""Unsupervised domain adaptation entry point for ULT-NILM.

Covers the four transfer scenarios of Table III. For same-dataset
scenarios ``--scenario intra_*`` the source and target datasets coincide
and DA reduces to fine-tuning on the held-out house.

By default DA runs over the maximum-configuration subnet. Pass
``--subnet-config pruned_subnet.json`` to adapt the pruned subnet
directly, which mirrors the edge-deployment flow of Section III.A:
the cloud prunes a device-specific subnet, the subnet is shipped to the
device, and adaptation runs on-device with the frozen backbone +
residual-only update path already enabled by
``enable_domain_adaptation``.
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
from ult_nilm.training.domain_adaptation import train_domain_adaptation


def main() -> None:
    parser = argparse.ArgumentParser(description="LCT domain adaptation for ULT-NILM")
    add_common_arguments(parser)
    parser.add_argument("--scenario", required=True, choices=["intra_redd", "intra_ukdale", "redd_to_ukdale", "ukdale_to_redd"])
    parser.add_argument("--appliance", required=True, choices=["washing_machine", "dishwasher", "fridge", "microwave"])
    parser.add_argument("--checkpoint", required=True, help="Post-progressive-shrinking supernet checkpoint (.pth)")
    parser.add_argument(
        "--subnet-config",
        type=str,
        default=None,
        help="Optional pruned_subnet.json. If provided, DA is applied to the pruned subnet; "
             "otherwise DA is applied to the maximum-configuration subnet.",
    )
    parser.add_argument("--limit-samples", type=int, default=-1)
    parser.add_argument("--no-source-labels", action="store_true",
                        help="Drop the supervised task term; align features only.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    dataset_path = resolve_dataset_path(cfg, args.dataset_path)

    source_params, target_params = dataset_params_from_scenario(
        cfg, args.scenario, args.appliance, args.limit_samples
    )
    if target_params is None:
        raise SystemExit(
            f"scenario '{args.scenario}' has no target domain; domain adaptation requires a target."
        )

    seq2seq = cfg["supernet"]["seq2seq"]
    source_x, source_y, *_rest = reload_dataset(source_params, dataset_path=dataset_path, seq2seq=seq2seq)
    target_x, _target_y, *_rest_target = reload_dataset(
        target_params, dataset_path=dataset_path, seq2seq=seq2seq
    )

    model = build_model(cfg, args)
    model.load_model(args.checkpoint)

    if args.subnet_config is not None:
        if not os.path.exists(args.subnet_config):
            raise SystemExit(f"subnet config not found: {args.subnet_config}")
        with open(args.subnet_config) as f:
            payload = json.load(f)
        cfg_alpha = payload["config"]
        model.set_active_subnet(ks=cfg_alpha["ks"], e=cfg_alpha["e"], d=cfg_alpha["d"])
        save_name = "subnet_da.pth"
        print(
            f"adapting pruned subnet: memory={payload.get('memory_bytes', 'n/a')} B, "
            f"ks={cfg_alpha['ks']} e={cfg_alpha['e']} d={cfg_alpha['d']}"
        )
    else:
        model.set_max_net()
        save_name = "supernet_da.pth"

    da_cfg = cfg["domain_adaptation"]
    epochs = args.epochs or da_cfg["epochs"]
    batch_size = args.batch_size or da_cfg["batch_size"]
    final_loss, elapsed = train_domain_adaptation(
        model,
        source_data=source_x,
        source_labels=None if args.no_source_labels else source_y,
        target_data=target_x,
        epochs=epochs,
        batch_size=batch_size,
        domain_loss_weight=da_cfg["domain_loss_weight"],
        early_stop_min_epochs=da_cfg["early_stop_min_epochs"],
        num_workers=args.num_workers,
    )
    model.save_model(os.path.join(model.work_dir, save_name))
    print(f"domain adaptation finished: loss={final_loss:.4f} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
