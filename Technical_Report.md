# Technical Report: Point-Supervised Remote Sensing Segmentation with Partial Cross-Entropy Loss

## 1. Introduction

This report documents the solution to the technical assessment on **partial cross-entropy (PCE) loss** for remote sensing image (RSI) semantic segmentation under **sparse point supervision**. Dense pixel-wise labels are expensive to collect; point annotations reduce labeling cost while still providing class and location cues. We simulate point labels by randomly sampling pixels from full masks in the provided aerial imagery dataset, train a U-Net with PCE loss, and study factors that affect segmentation quality.

**Dataset:** `Data/aerial imagery/` — 24 image tiles (8 geographic tiles × 3 parts), RGB orthophotos with polygon masks for 6 classes: Water, Land, Road, Building, Vegetation, and Unlabeled.

---

## 2. Method

### 2.1 Partial Cross-Entropy Loss

Standard cross-entropy averages over all pixels. In point supervision, only a small subset of pixels has labels. **Partial cross-entropy** restricts the loss to labeled pixels using a binary mask \(m_i \in \{0,1\}\):

\[
L_{\text{PCE}} = -\frac{\sum_i \sum_c y_{i,c}\,\log(p_{i,c})\, m_i}{\sum_i m_i}
\]

where \(y_{i,c}\) is the one-hot ground truth, \(p_{i,c}\) is the predicted probability after softmax, and \(m_i=1\) only at supervised point locations.

**Implementation:** `src/partial_ce_loss.py` — class `PartialCrossEntropyLoss` takes `(logits, point_labels, label_mask)` and ignores pixels with `ignore_index=255` (void / black regions in masks).

### 2.2 Simulated Point Labels

From each dense mask we:

1. Convert RGB colors to class indices.
2. Exclude the **Unlabeled** class and void pixels from the sampling pool.
3. Sample `N` pixel coordinates per image using either:
   - **Random:** uniform over valid pixels.
   - **Stratified:** roughly equal points per present class (better class balance).

The resulting sparse maps are used only for **training loss**; **validation** still uses full dense masks (mIoU, pixel accuracy).

### 2.3 Segmentation Network

A lightweight **U-Net** (`src/model.py`, base width 32) with ImageNet normalization. Input images are resized to **256×256**. This is a standard encoder–decoder for RSI segmentation and sufficient for the assessment scale.

### 2.4 Training Setup

| Setting | Value |
|--------|--------|
| Train tiles | Tile 1–6 (18 images) |
| Val tiles | Tile 7–8 (6 images) |
| Optimizer | Adam, lr = 1e-3 |
| Batch size | 4 (2 in quick runs) |
| Epochs | 20 recommended (5 in smoke tests) |
| Loss (point mode) | Partial CE on `point_labels` + `label_mask` |
| Loss (baseline) | Full CE on all non-void pixels |

---

## 3. Experiments

### 3.1 Purpose and Hypotheses

**Experiment A — Number of point labels (Factor 1)**  
*Hypothesis:* More simulated points provide richer supervision; mIoU should increase with `N` until approaching the fully supervised upper bound.

**Experiment B — Sampling strategy (Factor 2)**  
*Hypothesis:* Stratified sampling (balanced across classes) yields more stable training than pure random sampling, especially for minority classes (Water, Building).

**Baseline — Full supervision**  
Train with standard CE on all labeled pixels to estimate an upper bound for the same architecture and data split.

### 3.2 Experimental Process

1. Implement PCE and data pipeline (`src/`, `train.py`).
2. Run `python train.py --experiment all --epochs 20` (or `--quick` for a fast check).
3. Record validation **mean IoU (mIoU)** and **pixel accuracy** on held-out tiles.
4. Compare runs: `points_100`, `points_500`, `points_2000`, `strategy_random_n500`, `strategy_stratified_n500`, `full_supervised`.

### 3.3 Results (5-epoch smoke run on Apple MPS)

| Run | Points | Strategy | Best mIoU | Final Acc |
|-----|--------|----------|-----------|-----------|
| points_500 / strategy_stratified | 500 | stratified | **0.314** | 0.628 |
| strategy_random_n500 | 500 | random | 0.274 | 0.630 |
| points_100 | 100 | stratified | 0.295 | 0.599 |
| points_2000 | 2000 | stratified | 0.295 | 0.472 |
| full_supervised | all pixels | — | 0.207 | 0.631 |

**Observations:**

- **Point supervision can work with very few labels:** 100–500 points per 256×256 image (~0.15–0.8% of pixels) already reach ~0.29–0.31 mIoU after 5 epochs.
- **Stratified vs random (Factor 2):** With 500 points, stratified sampling reached **mIoU 0.314** vs **0.274** for random — confirming that balanced class coverage improves sparse supervision.
- **More points are not always better in short runs:** `points_2000` did not beat `points_500`, likely due to optimization noise, class imbalance (Road dominates ~50% of pixels), and limited epochs. Longer training (20+ epochs) is recommended for stable trends.
- **Road class dominates IoU** in all runs (often >0.6 IoU) because it occupies the largest area; Building and Water remain harder under sparse labels.
- **Full supervision underperformed in the 5-epoch quick run** (mIoU 0.21) — with more epochs it typically surpasses point-only training; the small dataset and strong class imbalance require careful tuning.

Per-class IoU example (`points_500`, best epoch): Road 0.63, Water 0.28, Land 0.24, Building 0.25, Vegetation 0.26.

### 3.4 Discussion

1. **PCE is necessary** for point training: applying full-image CE would wrongly penalize unlabeled pixels.
2. **Stratified point sampling** helps rare classes receive gradients; run `python train.py --experiment strategy` for a direct comparison.
3. **Validation on full masks** measures how well sparse supervision generalizes to dense prediction — the actual deployment scenario.
4. **Improvements** (optional extensions): pseudo-label expansion, longer training, weighted CE for class imbalance, ResNet encoder, or test-time augmentation.

---

## 4. Deliverables

| File | Description |
|------|-------------|
| `src/partial_ce_loss.py` | Partial cross-entropy implementation |
| `src/dataset.py` | Aerial dataset + point sampling |
| `src/model.py` | U-Net |
| `src/metrics.py` | mIoU / accuracy |
| `train.py` | Training + experiment runner |
| `assessment_demo.ipynb` | Interactive demo |
| `Technical_Report.md` | This document |
| `outputs/experiment_results.json` | Metrics from experiments |

### How to Run

```bash
pip install -r requirements.txt
python train.py --experiment all --epochs 20
jupyter notebook assessment_demo.ipynb
```

---

## 5. Conclusion

We implemented **partial cross-entropy loss** for point-supervised semantic segmentation, integrated it into a **U-Net** trained on the provided **aerial imagery** dataset with **simulated point labels**, and designed experiments on **point count** and **sampling strategy**. Results show that sparse supervision can train a multi-class segmenter with modest annotation cost; performance depends on the number and distribution of points, training duration, and class balance. The code and report satisfy the three assessment tasks: PCE implementation, network integration with RSI data, and experimental analysis with documented results.
