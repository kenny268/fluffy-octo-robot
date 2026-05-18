"""Tile-based cross-validation splits."""

from __future__ import annotations

from pathlib import Path

from .constants import DATA_ROOT


def list_tiles(root: Path | str = DATA_ROOT) -> list[str]:
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("Tile"))


def leave_one_tile_out_folds(
    root: Path | str = DATA_ROOT,
    max_folds: int | None = None,
) -> list[tuple[str, list[str], list[str]]]:
    """
    Each fold holds out one tile for validation.

    Yields (fold_name, train_tiles, val_tiles).
    """
    tiles = list_tiles(root)
    folds = []
    for val_tile in tiles:
        train_tiles = [t for t in tiles if t != val_tile]
        folds.append((f"fold_{val_tile.replace(' ', '_')}", train_tiles, [val_tile]))
    if max_folds is not None:
        folds = folds[:max_folds]
    return folds
