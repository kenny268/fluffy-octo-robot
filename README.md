# Point-Supervised Remote Sensing Segmentation

**Partial cross-entropy loss** for semantic segmentation of aerial imagery under **sparse point supervision**.

## Problem

Remote sensing segmentation usually needs expensive pixel-wise labels. This project simulates **point annotations** from masks, trains a **U-Net** with **partial CE** (loss only on labeled points), and measures how label budget and sampling strategy affect validation mIoU.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── run_train.sh
├── train.py
├── docs/Technical_Report.md
├── scripts/eval.py
├── src/
├── Data/aerial imagery/
└── outputs/          # checkpoints + metrics (gitignored except README)
```

## Dataset

8 tiles × 3 parts = **24** RGB patches; **6 classes** (Water, Land, Road, Building, Vegetation, Unlabeled). Default split: **Tiles 1–6** train, **7–8** val. Details: [Data/README.md](Data/README.md).

## Setup

```bash
pip install -r requirements.txt
# ResNet34 encoder (optional):
pip install segmentation-models-pytorch
```

Python 3.10+, PyTorch.

## Train & evaluate

**Reproduce reported experiments:**

```bash
./run_train.sh --experiment all --epochs 20 --batch-size 4
```

**Other experiment groups:**

```bash
./run_train.sh --experiment points      # 100 / 500 / 2000 / 8000 points
./run_train.sh --experiment strategy    # random / stratified / inverse_freq
./run_train.sh --experiment full        # full-pixel CE baseline
./run_train.sh --experiment resnet34 --epochs 20 --batch-size 2
./run_train.sh --experiment multiseed --epochs 15
python scripts/eval.py --checkpoint outputs/points_8000.pt --num-samples 3
```

**Quick smoke test:** `./run_train.sh --quick --experiment all --epochs 5`

**macOS:** use `./run_train.sh` if you hit OpenMP / `libomp` errors.

## Training options

| Flag | Description |
|------|-------------|
| `--class-weights` | Median-frequency weighted loss |
| `--fixed-points-per-epoch` | Stable point samples within each epoch |
| `--pseudo-labels` | Merge high-confidence predictions each epoch |
| `--experiment crossval` | Leave-one-tile-out evaluation |
| `--patience 8` | Early stopping (default on) |
| `--arch resnet34_unet` | ImageNet ResNet34 encoder |

Artifacts: `outputs/experiment_results.json`, `outputs/<run>.pt`, `outputs/figures/`.

## Results

Best validation **mIoU 0.369** (`points_8000`, stratified, 20 epochs). A packaged copy for submission lives in **`Submit-ready/`**. Full tables: [docs/Technical_Report.md](docs/Technical_Report.md).
