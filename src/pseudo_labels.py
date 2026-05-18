"""High-confidence pseudo-label expansion for point-supervised training."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .constants import IGNORE_INDEX
from .dataset import AerialSegmentationDataset


@torch.no_grad()
def refresh_pseudo_labels(
    model: torch.nn.Module,
    dataset: AerialSegmentationDataset,
    device: torch.device,
    threshold: float = 0.9,
    batch_size: int = 4,
) -> int:
    """
    Predict on train set (no shuffle) and store pseudo labels on the dataset.

    Pseudo pixels are merged with point labels in __getitem__ (union of masks).
    Returns number of pseudo pixels added (approximate).
    """
    if dataset.use_full_mask:
        return 0

    dataset.clear_pseudo_labels()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    total_pseudo = 0
    idx_offset = 0

    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)

        b = images.size(0)
        for i in range(b):
            idx = idx_offset + i
            conf_np = conf[i].cpu().numpy()
            pred_np = pred[i].cpu().numpy().astype(np.int64)
            pl = np.full(pred_np.shape, IGNORE_INDEX, dtype=np.int64)
            pm = np.zeros(pred_np.shape, dtype=np.float32)
            high = conf_np >= threshold
            pl[high] = pred_np[high]
            pm[high] = 1.0
            dataset.set_pseudo_labels(idx, pl, pm)
            total_pseudo += int(high.sum())
        idx_offset += b

    return total_pseudo
