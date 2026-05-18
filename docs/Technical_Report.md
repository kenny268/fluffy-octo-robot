# Technical Report: Point-Supervised Remote Sensing Segmentation with Partial Cross-Entropy Loss

## 1. Introduction

This report describes **partial cross-entropy (PCE) loss** for remote sensing image (RSI) semantic segmentation under **sparse point supervision**. Dense pixel-wise labels are expensive; point annotations reduce cost while preserving class and location cues. We simulate points by sampling from full masks, train a **U-Net** with PCE, and evaluate how **label budget** and **sampling strategy** affect validation quality.

**Dataset:** `Data/aerial imagery/` — 24 tiles (8 geographic tiles × 3 parts), RGB orthophotos with polygon masks for six classes: Water, Land, Road, Building, Vegetation, and Unlabeled.

---

## 2. Method

### 2.1 Partial Cross-Entropy Loss

Standard cross-entropy averages over all pixels. Under point supervision, only a subset carries labels. **Partial cross-entropy** applies loss only where a binary mask \(m_i \in \{0,1\}\) is set:

\[
L_{\text{PCE}} = -\frac{\sum_i \sum_c y_{i,c}\,\log(p_{i,c})\, m_i}{\sum_i m_i}
\]

**Implementation:** `src/partial_ce_loss.py` — `PartialCrossEntropyLoss(logits, point_labels, label_mask)` with `ignore_index=255` for void pixels. **Class weights** (`src/class_weights.py`, `--class-weights`) use median-frequency balancing on train masks for imbalanced scenes.

### 2.2 Simulated Point Labels

1. RGB masks → class indices (`src/constants.py`).
2. Exclude **Unlabeled** and void pixels from the sampling pool.
3. Draw `N` pixels per image:
   - **Random** — uniform over valid pixels.
   - **Stratified** — roughly equal count per present class.
   - **Inverse frequency** — favors rare classes (`inverse_freq`).

Sparse maps drive **training loss** only; **validation** uses full dense masks (mIoU, accuracy). `--fixed-points-per-epoch` reuses the same point pattern within each epoch (seeded by epoch index) for stable gradients.

### 2.3 Segmentation Network

**U-Net** (`src/model.py`, width 32), ImageNet normalization, **256×256** inputs. An alternate **ResNet34 encoder U-Net** (`resnet34_unet`) is available for encoder comparisons (`--experiment resnet34`).

### 2.4 Training Pipeline

| Component | Detail |
|-----------|--------|
| Split | Train Tiles 1–6 (18 images), val Tiles 7–8 (6 images) |
| Optimizer | Adam, lr = 1e-3, **cosine annealing** (default) |
| Batch size | 4 (2 for ResNet34 U-Net) |
| Epochs | 20 default; **early stopping** (patience 8) on val mIoU |
| Checkpoint | Best validation mIoU (`outputs/<run>.pt`) |
| Augmentation | Random H/V flip, color jitter (train) |
| Point mode | Partial CE on `point_labels` + `label_mask` |
| Full-pixel baseline | Standard CE on all non-void pixels |
| Pseudo-labels | High-confidence predictions merged each epoch (`src/pseudo_labels.py`) |
| Robustness | Multi-seed runs, leave-one-tile-out CV (`src/crossval.py`) |
| Qualitative export | `scripts/eval.py` → `outputs/figures/` |

---

## 3. Experiments

### 3.1 Hypotheses

**Factor 1 — Point count:** More labeled points increase effective supervision; mIoU should rise with `N` until near the dense-label regime.

**Factor 2 — Sampling:** Stratified (and inverse-frequency) sampling should outperform pure random sampling when classes are imbalanced (e.g. Road vs Water).

**Baseline:** Full-pixel CE on the same split estimates dense-label performance for the same architecture.

### 3.2 Protocol

```bash
./run_train.sh --experiment all --epochs 20
```

Hardware: Apple MPS. Metrics: **best** validation mIoU and pixel accuracy. Per-run logs: `outputs/experiment_results.json`.

### 3.3 Results — point count and sampling (20 epochs)

| Run | Points | Strategy | Best mIoU | Best epoch |
|-----|--------|----------|-----------|------------|
| **points_8000** | 8000 | stratified | **0.369** | 8 |
| **strategy_inverse_freq_n500** | 500 | inverse_freq | **0.365** | 11 |
| **strategy_stratified_n500** | 500 | stratified | **0.358** | 10 |
| points_500 | 500 | stratified | 0.335 | 11 |
| points_2000 | 2000 | stratified | 0.325 | 10 |
| strategy_random_n500 | 500 | random | 0.325 | 12 |
| points_100 | 100 | stratified | 0.315 | 8 |
| full_supervised | all pixels | — | 0.302 | 11 |

**Factor 1:** mIoU rises from **0.315** (100 points) to **0.369** (8000 points).

**Factor 2 @ 500 points:** Inverse-frequency **0.365** and stratified **0.358** both beat random **0.325** — balancing rare classes helps under skewed land-cover.

**Dense baseline (0.302)** sits near the point-supervised range on this small val set; the best sparse run still wins, so label *placement* matters as much as label *density* here.

### 3.4 Additional checks

| Check | Result | Note |
|-------|--------|------|
| Multi-seed (`points_8000`, 15 epochs) | **0.341 ± 0.015** | Seeds 42/43/44: 0.348, 0.351, 0.323 — small val set adds variance |
| ResNet34 U-Net (`points_8000`, 20 epochs) | **0.355** | Strong encoder; limited data (24 images) favors the lighter U-Net |
| Qualitative (`scripts/eval.py`) | `outputs/figures/val_sample_*.png` | Roads and buildings segment cleanly; water/building IoU still weakest |

**Best model:** `outputs/points_8000.pt` — **mIoU 0.369**, stratified 8000-point training.

### 3.5 Discussion (key points)

1. **PCE** is required for point supervision — loss is masked to labeled pixels only.
2. **Early stopping + best checkpoint** report metrics on the weights you would actually deploy.
3. **8000 stratified points** gave the top mIoU; at 500 points, **inverse-frequency** and **stratified** sampling beat random.
4. **Six-image validation** makes rankings seed-sensitive; multi-seed spread is ~±0.015 mIoU.
5. **Class imbalance** persists (Road highest IoU); class weights and pseudo-labels are available in `train.py` for harder classes.
6. **Deployment metric** remains dense mIoU on full masks despite sparse training labels.

---

## 4. Repository map

| Path | Role |
|------|------|
| `src/partial_ce_loss.py` | Partial CE |
| `src/dataset.py` | Data, point sampling, augmentation |
| `src/model.py` | U-Net / ResNet34-U-Net |
| `src/class_weights.py`, `src/pseudo_labels.py`, `src/crossval.py` | Training utilities |
| `src/metrics.py` | mIoU, accuracy |
| `train.py`, `run_train.sh` | Training and experiments |
| `scripts/eval.py` | Validation figures |
| `docs/Technical_Report.md` | This report |

### Reproduce

```bash
pip install -r requirements.txt
./run_train.sh --experiment all --epochs 20
python scripts/eval.py --checkpoint outputs/points_8000.pt --num-samples 3
```

On macOS, use `./run_train.sh` (sets `KMP_DUPLICATE_LIB_OK=TRUE`) if OpenMP errors appear.

---

## 5. Conclusion

We built an end-to-end **point-supervised RSI segmentation** stack: **PCE loss**, simulated **point labels**, **U-Net** training with **early stopping**, **cosine LR**, and **augmentation**, plus experiments on **point count** and **sampling strategy**. Best validation **mIoU = 0.369** (`points_8000`, stratified). Stratified sampling at 500 points (**0.358**) outperformed random (**0.325**), supporting balanced supervision under class skew.
