#!/usr/bin/env python3
"""Visualize predictions and export qualitative figures for the report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from src.dataset import AerialSegmentationDataset, train_val_split
from src.model import build_model


def denormalize(img: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = img.cpu().numpy().transpose(1, 2, 0)
    x = np.clip(x * std + mean, 0, 1)
    return x


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    cmap = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))[:, :3]
    out = np.zeros((*mask.shape, 3), dtype=np.float32)
    for c in range(NUM_CLASSES):
        out[mask == c] = cmap[c]
    out[mask == IGNORE_INDEX] = 0.2
    return out


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--arch", choices=["unet", "resnet34_unet"], default="unet")
    p.add_argument("--output-dir", type=str, default="outputs/figures")
    p.add_argument("--num-samples", type=int, default=3)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _, val_tiles = train_val_split()
    ds = AerialSegmentationDataset(tile_ids=val_tiles, augment=False, num_points=500)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = build_model(args.arch, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(loader):
        if i >= args.num_samples:
            break
        image = batch["image"].to(device)
        gt = batch["gt"][0].numpy()
        pred = model(image).argmax(dim=1)[0].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(denormalize(batch["image"][0]))
        axes[0].set_title("Image")
        axes[1].imshow(colorize_mask(gt))
        axes[1].set_title("Ground truth")
        axes[2].imshow(colorize_mask(pred))
        axes[2].set_title("Prediction")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Val sample {i} — classes: {', '.join(CLASS_NAMES)}")
        fig.tight_layout()
        path = out_dir / f"val_sample_{i}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
