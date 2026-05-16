# Point-Supervised Remote Sensing Segmentation

Technical assessment implementation: **partial cross-entropy loss** for semantic segmentation of aerial imagery under **sparse point supervision**.

## Problem

Remote sensing segmentation usually needs expensive pixel-wise labels. This project simulates cheaper **point annotations** (random samples from full masks), trains a **U-Net** with **partial CE** (loss only on labeled points), and compares factors that affect validation performance.

## Repository layout

```
.
├── README.md                 # This file
├── Technical_Report.md       # Method + experiments (submission write-up)
├── requirements.txt
├── train.py                  # Training and experiment runner
├── assessment_demo.ipynb     # Short demo (loss, point sampling, results)
├── src/
│   ├── partial_ce_loss.py    # Partial cross-entropy
│   ├── dataset.py            # Data loading + point simulation
│   ├── model.py              # U-Net
│   ├── metrics.py            # mIoU, pixel accuracy
│   └── constants.py            # Classes, colors, paths
├── Data/
│   └── aerial imagery/       # Images, masks, classes.json
└── outputs/                  # Created by train.py (gitignored)
```

Local interview notes (structure, design decisions, function walkthroughs) live in **`prep/`** — that folder is **gitignored** and not pushed to GitHub.

## Dataset

**Aerial imagery** under `Data/aerial imagery/`:

- 8 tiles × 3 image parts = **24** RGB patches  
- Polygon masks (PNG) with **6 classes**: Water, Land, Road, Building, Vegetation, Unlabeled  
- Train/val split by tile: **Tile 1–6** train, **Tile 7–8** val (no tile leakage)

Place the dataset in the path above before training. If the repo is too large for GitHub, host `Data/` separately (Drive/LFS) and document the download link here.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires **Python 3.10+** and **PyTorch**.

## Usage

**Run all experiments** (point count + sampling strategy + full-supervised baseline):

```bash
python train.py --experiment all --epochs 20 --batch-size 4
```

**Smoke test** (fewer epochs):

```bash
python train.py --quick --experiment all --epochs 5
```

**Single experiment groups:**

```bash
python train.py --experiment points    # 100 / 500 / 2000 / 8000 points
python train.py --experiment strategy  # random vs stratified @ 500 points
python train.py --experiment full      # full-pixel CE baseline
```

Metrics are written to `outputs/experiment_results.json`. Checkpoints are saved as `outputs/<run_name>.pt`.

**Notebook:**

```bash
jupyter notebook assessment_demo.ipynb
```

## Assessment mapping

| Task | Deliverable |
|------|-------------|
| 1. Partial cross-entropy | `src/partial_ce_loss.py` |
| 2. RSI data + point labels + network | `src/dataset.py`, `src/model.py`, `train.py` |
| 3. Experiments + report | `train.py` experiments, `Technical_Report.md` |

## Experiments (summary)

Documented in `Technical_Report.md`:

1. **Number of point labels** per image (100, 500, 2000, …)  
2. **Sampling strategy**: random vs **stratified** (balanced per class)

Validation uses **full dense masks** (mIoU, accuracy) even though training uses sparse points.

## Citation / context

Partial CE for point supervision follows weakly supervised segmentation practice (e.g. Bearman et al.; recent RSI point-supervision work). See `Technical_Report.md` for formulation and discussion.

## Author

Submitted as part of a technical interview assessment.
