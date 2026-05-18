"""Class frequency weights from training masks (median-frequency balancing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .constants import DATA_ROOT, NUM_CLASSES
from .dataset import rgb_mask_to_class_indices


def compute_class_pixel_counts(
    root: Path | str = DATA_ROOT,
    tile_ids: list[str] | None = None,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Count pixels per class across train masks."""
    root = Path(root)
    if tile_ids is None:
        tile_ids = sorted(
            p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("Tile")
        )

    counts = np.zeros(num_classes, dtype=np.float64)
    for tile in tile_ids:
        mask_dir = root / tile / "mask"
        if not mask_dir.exists():
            continue
        for mask_path in mask_dir.glob("*.png"):
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            gt = rgb_mask_to_class_indices(mask_rgb)
            for c in range(num_classes):
                counts[c] += (gt == c).sum()
    return counts


def class_weights_from_counts(
    counts: np.ndarray,
    ignore_unlabeled_class: int = 5,
) -> torch.Tensor:
    """
    Median-frequency balancing: w_c = median(freq) / freq_c for classes with support.
    """
    counts = counts.astype(np.float64)
    num_classes = len(counts)
    weights = np.ones(num_classes, dtype=np.float64)

    total = counts.sum()
    if total <= 0:
        return torch.ones(num_classes, dtype=torch.float32)

    freq = counts / total
    present = freq > 0
    if not present.any():
        return torch.ones(num_classes, dtype=torch.float32)

    median_freq = np.median(freq[present])
    for c in range(num_classes):
        if freq[c] > 0:
            weights[c] = median_freq / freq[c]

    weights = np.clip(weights, 0.1, 10.0)
    if ignore_unlabeled_class < num_classes:
        weights[ignore_unlabeled_class] = 1.0

    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.mean().clamp(min=1e-8)


def get_train_class_weights(
    train_tiles: list[str],
    root: Path | str = DATA_ROOT,
) -> torch.Tensor:
    counts = compute_class_pixel_counts(root=root, tile_ids=train_tiles)
    return class_weights_from_counts(counts)
