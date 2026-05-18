"""
Partial Cross-Entropy (PCE) loss for point-supervised segmentation.

Computes cross-entropy only on pixels marked as labeled in the supervision mask.
Unlabeled pixels (m_i = 0) do not contribute to the loss or gradients.

    L_PCE = -sum_i (y_i * log(p_i) * m_i) / sum_i(m_i)

Reference: weakly supervised / point-supervised remote sensing segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialCrossEntropyLoss(nn.Module):
    """
    Cross-entropy averaged over labeled pixels only.

    Args:
        ignore_index: class index excluded from loss (e.g. void / unlabeled GT).
        reduction: kept for API compatibility; always returns scalar mean over labeled pixels.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self._class_weights = class_weights

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (N, C, H, W) raw network output.
            targets: (N, H, W) integer class labels at each pixel.
            label_mask: (N, H, W) binary/float mask; 1 = use pixel in loss, 0 = ignore.
        """
        n, c, h, w = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, c)
        targets_flat = targets.reshape(-1).long()
        mask_flat = label_mask.reshape(-1).float()

        valid = mask_flat > 0
        if targets_flat.dtype != torch.long:
            targets_flat = targets_flat.long()

        # Drop pixels with void target even if mask says labeled
        if self.ignore_index is not None:
            valid = valid & (targets_flat != self.ignore_index)

        if valid.sum() == 0:
            return logits.sum() * 0.0

        weight = (
            self._class_weights.to(logits.device) if self._class_weights is not None else None
        )
        loss_per_pixel = F.cross_entropy(
            logits_flat,
            targets_flat,
            weight=weight,
            reduction="none",
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        masked_loss = loss_per_pixel * mask_flat
        return masked_loss.sum() / mask_flat[valid].sum().clamp(min=1.0)


def full_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard pixel-wise CE on all non-ignored pixels (fully supervised baseline)."""
    weight = class_weights.to(logits.device) if class_weights is not None else None
    return F.cross_entropy(
        logits,
        targets.long(),
        weight=weight,
        ignore_index=ignore_index,
    )
