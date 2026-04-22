# ULT-NILM: Unsupervised Lightweight Transfer Learning for Edge NILM

Reference implementation for the paper **"Unsupervised Lightweight Transfer Learning for Edge Non-Intrusive Load Monitoring"**, submitted to *IEEE Transactions on Smart Grid*.

The repository contains code for reproducing the paper: the elastic supernet, progressive shrinking training, hardware-aware subnet pruning, and unsupervised domain adaptation.

## Installation

```bash
git clone https://github.com/hkuedl/ULT-NILM
cd ULT-NILM
uv sync
# OR pip install -e ., not recommended
```

## Pre-run setup

Download REDD and UK-DALE and place them under a common root:

```
/path/to/datasets/
├── redd/
│   ├── house_1.csv, house_2.csv, ...
└── ukdale/
    ├── building_1.feather, building_2.feather, ...
```

> For REDD, we use cleand dataset as CSV format. For UK-DALE, we use `tools/nilmtk_converter.py` to generate .feather files from the raw dataset for fast read.

Either set `dataset_path` in `configs/default.yaml` or pass `--dataset-path /path/to/datasets` to every script.

The supernet hyper-parameters in `configs/default.yaml` match Table II in the paper.

## Examples

Every script accepts `--scenario` (`intra_redd` / `intra_ukdale` / `redd_to_ukdale` / `ukdale_to_redd`) and `--appliance` ( `washing_machine` / `dishwasher` / `fridge` / `microwave`).

```bash
# 1. Train the supernet at the maximum configuration.
python scripts/train_supernet.py \
    --scenario intra_redd --appliance fridge \
    --work-dir runs/intra_redd_fridge

# 2. Run progressive shrinking.
python scripts/progressive_shrinking.py \
    --scenario intra_redd --appliance fridge \
    --checkpoint runs/intra_redd_fridge/supernet_maxnet.pth \
    --work-dir runs/intra_redd_fridge

# 3. Prune the supernet for a target memory budget.
python scripts/prune_subnet.py \
    --scenario intra_redd --appliance fridge \
    --checkpoint runs/intra_redd_fridge/supernet_ps.pth \
    --memory-budget 499712 \
    --work-dir runs/intra_redd_fridge

# 4. Unsupervised domain adaptation on the pruned subnet.
python scripts/domain_adaptation.py \
    --scenario redd_to_ukdale --appliance fridge \
    --checkpoint runs/intra_redd_fridge/supernet_ps.pth \
    --subnet-config runs/intra_redd_fridge/pruned_subnet.json \
    --work-dir runs/redd_to_ukdale_fridge

# 5. Evaluate the adapted subnet.
python scripts/evaluate.py \
    --scenario redd_to_ukdale --appliance fridge \
    --checkpoint runs/redd_to_ukdale_fridge/subnet_da.pth \
    --subnet-config runs/intra_redd_fridge/pruned_subnet.json \
    --work-dir runs/redd_to_ukdale_fridge
```

> The value `499712` in step 3 approximates the 488 KB budget for our STM32F429 testbed. Adjust to target different MCU classes.

## Repo layout

```
ult_nilm/
├── networks/        Elastic supernet + dynamic operators + backbone
├── modules/         Time-frequency encoder + static layers
├── losses/          Sinkhorn + CORAL + MMD
├── training/        Progressive shrinking + domain adaptation
├── pruning/         Memory lookup table + hardware-aware search
├── data/            REDD / UK-DALE loaders + Seq2Seq / Seq2Point windowing
├── utils/           Base module + tensor helpers + metrics
└── model.py         High-level model wrapper
scripts/             Command-line helpers for each pipeline stage
configs/default.yaml Default config file for the pipeline
```

## Citation

```bibtex
@article{ult_nilm_2026,
  title   = {Unsupervised Lightweight Transfer Learning for Edge Non-Intrusive Load Monitoring},
  author  = {Lu, Taoyu and Li, Yehui and Yao, Ruiyang and Wang, Yi},
  journal = {IEEE Transactions on Smart Grid},
  year    = {2026},
  note    = {Under review}
}
```
