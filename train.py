#!/usr/bin/env python3
"""
Train segmentation models on aerial imagery with partial cross-entropy (point supervision).
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.class_weights import get_train_class_weights
from src.constants import IGNORE_INDEX, NUM_CLASSES
from src.crossval import leave_one_tile_out_folds
from src.dataset import AerialSegmentationDataset, train_val_split
from src.metrics import evaluate
from src.model import build_model
from src.partial_ce_loss import PartialCrossEntropyLoss, full_cross_entropy_loss
from src.pseudo_labels import refresh_pseudo_labels


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
    *,
    arch: str = "unet",
    patience: int = 8,
    early_stopping: bool = True,
    use_scheduler: bool = True,
    class_weights: torch.Tensor | None = None,
    pseudo_labels: bool = False,
    pseudo_threshold: float = 0.9,
    pseudo_start_epoch: int = 3,
) -> dict:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(arch, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_scheduler
        else None
    )
    pce_loss = PartialCrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
        class_weights=class_weights,
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)

    history = []
    best_miou = 0.0
    best_epoch = 0
    best_state: dict | None = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)

        if pseudo_labels and not use_full_supervision and epoch >= pseudo_start_epoch:
            n_pseudo = refresh_pseudo_labels(
                model, train_ds, device, threshold=pseudo_threshold, batch_size=batch_size
            )
            print(f"  [{name}] epoch {epoch}: refreshed {n_pseudo} pseudo pixels")

        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"{name} e{epoch}", leave=False):
            images = batch["image"].to(device)
            optimizer.zero_grad()

            logits = model(images)
            if use_full_supervision:
                loss = full_cross_entropy_loss(
                    logits,
                    batch["gt"].to(device),
                    IGNORE_INDEX,
                    class_weights=class_weights,
                )
            else:
                loss = pce_loss(
                    logits,
                    batch["point_labels"].to(device),
                    batch["label_mask"].to(device),
                )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(len(train_loader), 1),
            "lr": optimizer.param_groups[0]["lr"],
            **val_metrics,
        }
        history.append(row)

        if val_metrics["mean_iou"] > best_miou:
            best_miou = val_metrics["mean_iou"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        print(
            f"  [{name}] epoch {epoch}: loss={row['train_loss']:.4f} "
            f"mIoU={val_metrics['mean_iou']:.4f} acc={val_metrics['pixel_accuracy']:.4f}"
            f"{' *best*' if epoch == best_epoch else ''}"
        )

        if early_stopping and stale_epochs >= patience:
            print(f"  [{name}] early stop at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = model.state_dict()

    ckpt_path = output_dir / f"{name}.pt"
    torch.save(best_state, ckpt_path)

    final_metrics = evaluate(model, val_loader, device)
    best_row = history[best_epoch - 1] if best_epoch > 0 else history[-1]

    return {
        "name": name,
        "arch": arch,
        "num_points": train_ds.num_points,
        "point_strategy": train_ds.point_strategy,
        "use_full_supervision": use_full_supervision,
        "pseudo_labels": pseudo_labels,
        "best_mean_iou": best_miou,
        "best_epoch": best_epoch,
        "final_mean_iou": final_metrics["mean_iou"],
        "final_pixel_accuracy": final_metrics["pixel_accuracy"],
        "per_class_iou": final_metrics["per_class_iou"],
        "per_class_iou_at_best": best_row.get("per_class_iou", {}),
        "history": history,
        "checkpoint": str(ckpt_path),
        "early_stopped": len(history) < epochs,
    }


def run_experiments(args: argparse.Namespace) -> list[dict]:
    device = get_device()
    print(f"Device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    if args.experiment == "multiseed":
        seeds = [42, 43, 44] if not args.quick else [42, 43]
        fold_results = []
        for seed in seeds:
            torch.manual_seed(seed)
            train_tiles, val_tiles = train_val_split()
            cw = get_train_class_weights(train_tiles) if args.class_weights else None
            train_ds = AerialSegmentationDataset(
                tile_ids=train_tiles,
                num_points=8000,
                point_strategy="stratified",
                augment=True,
                seed=seed,
                fixed_points_per_epoch=args.fixed_points_per_epoch,
            )
            val_ds = AerialSegmentationDataset(
                tile_ids=val_tiles,
                num_points=8000,
                augment=False,
                seed=seed + 1,
            )
            r = train_one_run(
                f"multiseed_{seed}",
                train_ds,
                val_ds,
                device,
                args.epochs,
                args.lr,
                args.batch_size,
                use_full_supervision=False,
                output_dir=output_dir,
                arch=args.arch,
                patience=args.patience,
                early_stopping=not args.no_early_stop,
                use_scheduler=not args.no_scheduler,
                class_weights=cw,
            )
            fold_results.append(r)
            results.append(r)
        mious = [r["best_mean_iou"] for r in fold_results]
        summary = {
            "name": "multiseed_summary",
            "seeds": seeds,
            "mean_best_miou": statistics.mean(mious),
            "std_best_miou": statistics.stdev(mious) if len(mious) > 1 else 0.0,
            "runs": [r["name"] for r in fold_results],
        }
        results.append(summary)
        print(
            f"\nMulti-seed: mIoU = {summary['mean_best_miou']:.4f} "
            f"± {summary['std_best_miou']:.4f}"
        )
        _save_results(output_dir, results)
        print_summary([r for r in results if r["name"] != "multiseed_summary"])
        return results

    if args.experiment == "crossval":
        folds = leave_one_tile_out_folds(max_folds=2 if args.quick else None)
        fold_mious = []
        for fold_name, train_tiles, val_tiles in folds:
            torch.manual_seed(args.seed)
            train_ds = AerialSegmentationDataset(
                tile_ids=train_tiles,
                num_points=args.num_points,
                point_strategy="stratified",
                augment=True,
                seed=args.seed,
            )
            val_ds = AerialSegmentationDataset(
                tile_ids=val_tiles,
                num_points=args.num_points,
                augment=False,
                seed=args.seed + 1,
            )
            r = train_one_run(
                fold_name,
                train_ds,
                val_ds,
                device,
                args.epochs,
                args.lr,
                args.batch_size,
                use_full_supervision=False,
                output_dir=output_dir,
                arch=args.arch,
                patience=args.patience,
                early_stopping=not args.no_early_stop,
                use_scheduler=not args.no_scheduler,
            )
            fold_mious.append(r["best_mean_iou"])
            results.append(r)
        cv_summary = {
            "name": "crossval_summary",
            "mean_best_miou": statistics.mean(fold_mious),
            "std_best_miou": statistics.stdev(fold_mious) if len(fold_mious) > 1 else 0.0,
            "folds": len(folds),
        }
        results.append(cv_summary)
        print(
            f"\nCross-val: mIoU = {cv_summary['mean_best_miou']:.4f} "
            f"± {cv_summary['std_best_miou']:.4f} ({len(folds)} folds)"
        )
        _save_results(output_dir, results)
        print_summary([r for r in results if not r["name"].endswith("_summary")])
        return results

    train_tiles, val_tiles = train_val_split()
    class_weights = None
    if args.class_weights:
        class_weights = get_train_class_weights(train_tiles)
        print(f"Class weights: {class_weights.tolist()}")

    def make_datasets(
        num_points: int,
        strategy: str,
        full: bool = False,
        train_ids: list[str] | None = None,
        val_ids: list[str] | None = None,
    ):
        tr = train_ids if train_ids is not None else train_tiles
        va = val_ids if val_ids is not None else val_tiles
        train = AerialSegmentationDataset(
            tile_ids=tr,
            num_points=num_points,
            point_strategy=strategy,
            augment=True,
            seed=args.seed,
            use_full_mask=full,
            fixed_points_per_epoch=args.fixed_points_per_epoch,
            color_jitter=not args.no_color_jitter,
        )
        val = AerialSegmentationDataset(
            tile_ids=va,
            num_points=num_points,
            point_strategy=strategy,
            augment=False,
            seed=args.seed + 1,
            use_full_mask=full,
            fixed_points_per_epoch=True,
        )
        return train, val

    train_kwargs = dict(
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_dir=output_dir,
        arch=args.arch,
        patience=args.patience,
        early_stopping=not args.no_early_stop,
        use_scheduler=not args.no_scheduler,
        class_weights=class_weights,
        pseudo_labels=args.pseudo_labels,
        pseudo_threshold=args.pseudo_threshold,
        pseudo_start_epoch=args.pseudo_start_epoch,
    )

    if args.experiment in ("all", "points", "full"):
        point_counts = [100, 500, 2000] if args.quick else [100, 500, 2000, 8000]
        for n in point_counts:
            train_ds, val_ds = make_datasets(n, "stratified")
            results.append(
                train_one_run(
                    f"points_{n}",
                    train_ds,
                    val_ds,
                    use_full_supervision=False,
                    **train_kwargs,
                )
            )

    if args.experiment in ("all", "strategy") and not args.quick:
        for strategy in ("random", "stratified", "inverse_freq"):
            train_ds, val_ds = make_datasets(500, strategy)
            results.append(
                train_one_run(
                    f"strategy_{strategy}_n500",
                    train_ds,
                    val_ds,
                    use_full_supervision=False,
                    **train_kwargs,
                )
            )

    if args.experiment in ("all", "full") or args.mode == "full_supervised":
        train_ds, val_ds = make_datasets(0, "stratified", full=True)
        results.append(
            train_one_run(
                "full_supervised",
                train_ds,
                val_ds,
                use_full_supervision=True,
                **train_kwargs,
            )
        )

    if args.experiment == "resnet34":
        train_ds, val_ds = make_datasets(8000, "stratified")
        resnet_kwargs = {**train_kwargs, "arch": "resnet34_unet"}
        results.append(
            train_one_run(
                "resnet34_points_8000",
                train_ds,
                val_ds,
                use_full_supervision=False,
                **resnet_kwargs,
            )
        )

    if args.experiment == "pseudo":
        train_ds, val_ds = make_datasets(2000, "stratified")
        train_kwargs["pseudo_labels"] = True
        results.append(
            train_one_run(
                "pseudo_points_2000",
                train_ds,
                val_ds,
                use_full_supervision=False,
                **train_kwargs,
            )
        )

    _save_results(output_dir, results)
    print_summary([r for r in results if "summary" not in r["name"]])
    return results


def _save_results(output_dir: Path, results: list[dict]) -> None:
    path = output_dir / "experiment_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY (best val mIoU)")
    print("=" * 60)
    for r in sorted(results, key=lambda x: -x.get("best_mean_iou", 0)):
        if "best_mean_iou" not in r:
            continue
        arch = r.get("arch", "unet")
        print(
            f"  {r['name']:32s}  mIoU={r['best_mean_iou']:.4f}  "
            f"@ep {r.get('best_epoch', '?')}  arch={arch}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Point-supervised RSI segmentation")
    p.add_argument(
        "--experiment",
        choices=[
            "all",
            "points",
            "strategy",
            "full",
            "resnet34",
            "pseudo",
            "crossval",
            "multiseed",
        ],
        default="all",
    )
    p.add_argument("--mode", choices=["point", "full_supervised"], default="point")
    p.add_argument("--arch", choices=["unet", "resnet34_unet"], default="unet")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-points", type=int, default=2000, help="For crossval / single runs")
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--no-scheduler", action="store_true")
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--fixed-points-per-epoch", action="store_true")
    p.add_argument("--no-color-jitter", action="store_true")
    p.add_argument("--pseudo-labels", action="store_true", help="Expand labels each epoch")
    p.add_argument("--pseudo-threshold", type=float, default=0.9)
    p.add_argument("--pseudo-start-epoch", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 5)
    torch.manual_seed(args.seed)
    run_experiments(args)
