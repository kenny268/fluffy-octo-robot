"""Segmentation evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch

from .constants import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


def confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    mask = target != ignore_index
    pred = pred[mask].astype(np.int64)
    target = target[mask].astype(np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(pred, target):
        if 0 <= p < num_classes and 0 <= t < num_classes:
            cm[t, p] += 1
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    ious = []
    accs = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float("nan"))
        denom_acc = cm[c, :].sum()
        accs.append(tp / denom_acc if denom_acc > 0 else float("nan"))

    valid_ious = [x for x in ious if not np.isnan(x)]
    return {
        "pixel_accuracy": cm.trace() / cm.sum() if cm.sum() > 0 else 0.0,
        "mean_iou": float(np.mean(valid_ious)) if valid_ious else 0.0,
        "per_class_iou": {CLASS_NAMES[i]: ious[i] for i in range(len(ious))},
    }


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for batch in loader:
        images = batch["image"].to(device)
        gt = batch["gt"].numpy()
        logits = model(images)
        pred = logits.argmax(dim=1).cpu().numpy()
        for i in range(pred.shape[0]):
            cm_total += confusion_matrix(pred[i], gt[i])
    return metrics_from_confusion(cm_total)
