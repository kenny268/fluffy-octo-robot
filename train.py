#!/usr/bin/env python3
"""
Train U-Net on aerial imagery with partial cross-entropy (point supervision).

Experiments (assessment Task 3):
  - Factor 1: number of simulated point labels per image
  - Factor 2: point sampling strategy (random vs stratified)

Usage:
  python train.py --experiment all
  python train.py --experiment points --epochs 15
  python train.py --experiment strategy --epochs 15
  python train.py --mode full_supervised  # baseline upper bound
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.constants import IGNORE_INDEX, NUM_CLASSES
from src.dataset import AerialSegmentationDataset, train_val_split
from src.metrics import evaluate
from src.model import UNet
from src.partial_ce_loss import PartialCrossEntropyLoss, full_cross_entropy_loss


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_run(
    name: str,
    train_ds: AerialSegmentationDataset,
    val_ds: AerialSegmentationDataset,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    use_full_supervision: bool,
    output_dir: Path,
) -> dict:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pce_loss = PartialCrossEntropyLoss(ignore_index=IGNORE_INDEX)

    history = []
    best_miou = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"{name} e{epoch}", leave=False):
            images = batch["image"].to(device)
            optimizer.zero_grad()

            logits = model(images)
            if use_full_supervision:
                loss = full_cross_entropy_loss(logits, batch["gt"].to(device), IGNORE_INDEX)
            else:
                loss = pce_loss(
                    logits,
                    batch["point_labels"].to(device),
                    batch["label_mask"].to(device),
                )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(len(train_loader), 1),
            **val_metrics,
        }
        history.append(row)
        best_miou = max(best_miou, val_metrics["mean_iou"])
        print(
            f"  [{name}] epoch {epoch}: loss={row['train_loss']:.4f} "
            f"mIoU={val_metrics['mean_iou']:.4f} acc={val_metrics['pixel_accuracy']:.4f}"
        )

    ckpt_path = output_dir / f"{name}.pt"
    torch.save(model.state_dict(), ckpt_path)

    result = {
        "name": name,
        "num_points": train_ds.num_points,
        "point_strategy": train_ds.point_strategy,
        "use_full_supervision": use_full_supervision,
        "best_mean_iou": best_miou,
        "final_mean_iou": history[-1]["mean_iou"],
        "final_pixel_accuracy": history[-1]["pixel_accuracy"],
        "per_class_iou": history[-1]["per_class_iou"],
        "history": history,
        "checkpoint": str(ckpt_path),
    }
    return result


def run_experiments(args: argparse.Namespace) -> list[dict]:
    device = get_device()
    print(f"Device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tiles, val_tiles = train_val_split()
    results: list[dict] = []

    def make_datasets(num_points: int, strategy: str, full: bool = False):
        train = AerialSegmentationDataset(
            tile_ids=train_tiles,
            num_points=num_points,
            point_strategy=strategy,
            augment=True,
            seed=args.seed,
            use_full_mask=full,
        )
        val = AerialSegmentationDataset(
            tile_ids=val_tiles,
            num_points=num_points,
            point_strategy=strategy,
            augment=False,
            seed=args.seed + 1,
            use_full_mask=full,
        )
        return train, val

    if args.experiment in ("all", "points", "full"):
        point_counts = [100, 500, 2000] if args.quick else [100, 500, 2000, 8000]
        for n in point_counts:
            train_ds, val_ds = make_datasets(n, "stratified")
            r = train_one_run(
                f"points_{n}",
                train_ds,
                val_ds,
                device,
                args.epochs,
                args.lr,
                args.batch_size,
                use_full_supervision=False,
                output_dir=output_dir,
            )
            results.append(r)

    if args.experiment in ("all", "strategy") and not args.quick:
        for strategy in ("random", "stratified"):
            train_ds, val_ds = make_datasets(500, strategy)
            r = train_one_run(
                f"strategy_{strategy}_n500",
                train_ds,
                val_ds,
                device,
                args.epochs,
                args.lr,
                args.batch_size,
                use_full_supervision=False,
                output_dir=output_dir,
            )
            results.append(r)

    if args.experiment in ("all", "full") or args.mode == "full_supervised":
        train_ds, val_ds = make_datasets(0, "stratified", full=True)
        r = train_one_run(
            "full_supervised",
            train_ds,
            val_ds,
            device,
            args.epochs,
            args.lr,
            args.batch_size,
            use_full_supervision=True,
            output_dir=output_dir,
        )
        results.append(r)

    results_path = output_dir / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print_summary(results)
    return results


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for r in sorted(results, key=lambda x: -x["best_mean_iou"]):
        print(
            f"  {r['name']:30s}  mIoU={r['best_mean_iou']:.4f}  "
            f"acc={r['final_pixel_accuracy']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Point-supervised RSI segmentation")
    p.add_argument("--experiment", choices=["all", "points", "strategy", "full"], default="all")
    p.add_argument("--mode", choices=["point", "full_supervised"], default="point")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--quick", action="store_true", help="Fewer epochs/points for smoke test")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 5)
    torch.manual_seed(args.seed)
    run_experiments(args)
