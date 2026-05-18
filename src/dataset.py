"""
Aerial imagery dataset with simulated point-label supervision.

Point labels are sampled randomly from dense masks to mimic sparse annotation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import COLOR_TO_CLASS, DATA_ROOT, IGNORE_INDEX


def rgb_mask_to_class_indices(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB annotation mask to (H, W) class index map."""
    h, w = mask_rgb.shape[:2]
    label_map = np.full((h, w), IGNORE_INDEX, dtype=np.int64)
    for rgb, class_id in COLOR_TO_CLASS.items():
        match = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
        label_map[match] = class_id
    return label_map


def sample_point_labels(
    gt: np.ndarray,
    num_points: int,
    strategy: Literal["random", "stratified", "inverse_freq"] = "stratified",
    rng: random.Random | None = None,
    exclude_classes: tuple[int, ...] = (5,),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate sparse point supervision from dense ground truth.

    Returns:
        point_label_map: (H, W) with class id at sampled points, IGNORE elsewhere.
        label_mask: (H, W) binary mask, 1 at supervised points.
    """
    rng = rng or random.Random()
    h, w = gt.shape
    point_map = np.full((h, w), IGNORE_INDEX, dtype=np.int64)
    label_mask = np.zeros((h, w), dtype=np.float32)

    valid_mask = np.ones_like(gt, dtype=bool)
    for c in exclude_classes:
        valid_mask &= gt != c
    valid_mask &= gt != IGNORE_INDEX

    ys, xs = np.where(valid_mask)
    if len(ys) == 0:
        return point_map, label_mask

    if strategy == "stratified":
        classes_present = np.unique(gt[valid_mask])
        n_classes = len(classes_present)
        per_class = max(1, num_points // n_classes)
        selected = []
        for cls in classes_present:
            idx = np.where((gt == cls) & valid_mask)
            coords = list(zip(idx[0], idx[1]))
            rng.shuffle(coords)
            selected.extend(coords[:per_class])
        rng.shuffle(selected)
        selected = selected[:num_points]
    elif strategy == "inverse_freq":
        classes_present = np.unique(gt[valid_mask])
        pool: list[tuple[int, int]] = []
        weights: list[float] = []
        for cls in classes_present:
            idx = np.where((gt == cls) & valid_mask)
            coords = list(zip(idx[0], idx[1]))
            freq = max(len(coords), 1)
            for y, x in coords:
                pool.append((y, x))
                weights.append(1.0 / freq)
        n = min(num_points, len(pool))
        selected = []
        if n > 0:
            chosen = rng.choices(range(len(pool)), weights=weights, k=n)
            seen = set()
            for j in chosen:
                if pool[j] not in seen:
                    seen.add(pool[j])
                    selected.append(pool[j])
            if len(selected) < n:
                remaining = [c for c in pool if c not in seen]
                rng.shuffle(remaining)
                selected.extend(remaining[: n - len(selected)])
    else:
        indices = rng.sample(range(len(ys)), min(num_points, len(ys)))
        selected = [(ys[i], xs[i]) for i in indices]

    for y, x in selected:
        point_map[y, x] = gt[y, x]
        label_mask[y, x] = 1.0

    return point_map, label_mask


class AerialSegmentationDataset(Dataset):
    def __init__(
        self,
        root: Path | str = DATA_ROOT,
        tile_ids: list[str] | None = None,
        image_size: tuple[int, int] = (256, 256),
        num_points: int = 500,
        point_strategy: Literal["random", "stratified", "inverse_freq"] = "stratified",
        augment: bool = False,
        seed: int = 42,
        use_full_mask: bool = False,
        fixed_points_per_epoch: bool = False,
        color_jitter: bool = True,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.num_points = num_points
        self.point_strategy = point_strategy
        self.augment = augment
        self.base_seed = seed
        self.rng = random.Random(seed)
        self.use_full_mask = use_full_mask
        self.fixed_points_per_epoch = fixed_points_per_epoch
        self.color_jitter = color_jitter and augment
        self._epoch = 0
        self._pseudo_maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        all_tiles = sorted(
            [p.name for p in self.root.iterdir() if p.is_dir() and p.name.startswith("Tile")]
        )
        self.tiles = tile_ids if tile_ids else all_tiles

        self.samples: list[tuple[Path, Path]] = []
        for tile in self.tiles:
            img_dir = self.root / tile / "images"
            mask_dir = self.root / tile / "mask"
            for img_path in sorted(img_dir.glob("*.jpg")):
                mask_path = mask_dir / img_path.name.replace(".jpg", ".png")
                if mask_path.exists():
                    self.samples.append((img_path, mask_path))

        self._color_jitter_tf = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        self.img_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.mask_resize = transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.NEAREST
        )

    def set_epoch(self, epoch: int) -> None:
        """When fixed_points_per_epoch=True, point samples are stable within an epoch."""
        self._epoch = epoch

    def clear_pseudo_labels(self) -> None:
        self._pseudo_maps.clear()

    def set_pseudo_labels(
        self,
        sample_idx: int,
        pseudo_label_map: np.ndarray,
        pseudo_mask: np.ndarray,
    ) -> None:
        self._pseudo_maps[sample_idx] = (pseudo_label_map, pseudo_mask)

    def _merge_pseudo(
        self,
        idx: int,
        point_labels: np.ndarray,
        label_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if idx not in self._pseudo_maps:
            return point_labels, label_mask
        pl, pm = self._pseudo_maps[idx]
        add = (pm > 0) & (label_mask == 0)
        point_labels = point_labels.copy()
        label_mask = label_mask.copy()
        point_labels[add] = pl[add]
        label_mask = np.maximum(label_mask, pm)
        return point_labels, label_mask

    def _point_rng(self, sample_idx: int) -> random.Random:
        if self.fixed_points_per_epoch:
            return random.Random(self.base_seed + self._epoch * 10_000 + sample_idx)
        return self.rng

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

        if self.augment:
            if self.rng.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask_rgb = np.fliplr(mask_rgb).copy()
            if self.rng.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask_rgb = np.flipud(mask_rgb).copy()
            if self.color_jitter and self.rng.random() > 0.5:
                image = self._color_jitter_tf(image)

        gt = rgb_mask_to_class_indices(mask_rgb)
        mask_pil = Image.fromarray(gt.astype(np.uint8), mode="L")
        gt_small = np.array(self.mask_resize(mask_pil), dtype=np.int64)

        image_t = self.img_transform(image)

        if self.use_full_mask:
            point_labels = gt_small.copy()
            label_mask = (gt_small != IGNORE_INDEX).astype(np.float32)
        else:
            point_labels, label_mask = sample_point_labels(
                gt_small,
                self.num_points,
                strategy=self.point_strategy,
                rng=self._point_rng(idx),
            )
            point_labels, label_mask = self._merge_pseudo(idx, point_labels, label_mask)

        return {
            "image": image_t,
            "gt": torch.from_numpy(gt_small).long(),
            "point_labels": torch.from_numpy(point_labels).long(),
            "label_mask": torch.from_numpy(label_mask).float(),
        }


def train_val_split(
    root: Path | str = DATA_ROOT,
    val_tiles: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split by tiles to avoid leakage between train and validation."""
    root = Path(root)
    all_tiles = sorted(
        [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("Tile")]
    )
    if val_tiles is None:
        val_tiles = ["Tile 7", "Tile 8"]
    train_tiles = [t for t in all_tiles if t not in val_tiles]
    return train_tiles, val_tiles
