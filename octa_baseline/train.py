from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# TensorBoard may import TensorFlow if it is installed. Hide its unrelated CUDA messages;
# PyTorch device selection and CUDA execution are unaffected.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from octa.data import make_loaders
from octa.engine import run_epoch
from octa.losses import BCEDiceLoss
from octa.model import UNet
from octa.utils import seed_everything, write_json


METRIC_NAMES = ("loss", "dice", "iou", "precision", "recall", "specificity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCTA-500 OCTA-only U-Net baseline")
    parser.add_argument("--data-root", type=Path, default=Path("../data/OCTA-500数据集"))
    parser.add_argument("--scan-size", choices=("3mm", "6mm"), default="3mm")
    parser.add_argument(
        "--projection",
        choices=("OCTA(FULL)", "OCTA(ILM_OPL)", "OCTA(OPL_BM)"),
        default="OCTA(ILM_OPL)",
    )
    parser.add_argument("--target", default="GT_Capillary")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/unet_3mm_ilm_opl"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-visualizations", type=int, default=8)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: UNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_dice: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_dice": best_dice,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        },
        path,
    )


@torch.no_grad()
def save_predictions(
    model: UNet,
    loader,
    device: torch.device,
    threshold: float,
    output_path: Path,
    count: int,
) -> None:
    model.eval()
    rows: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for batch in loader:
        images = batch["image"].to(device)
        probabilities = torch.sigmoid(model(images)).cpu()
        for index, case_id in enumerate(batch["case_id"]):
            rows.append(
                (case_id, images[index, 0].cpu(), batch["mask"][index, 0], probabilities[index, 0])
            )
            if len(rows) >= count:
                break
        if len(rows) >= count:
            break

    if not rows:
        return
    figure, axes = plt.subplots(len(rows), 4, figsize=(12, 3 * len(rows)), squeeze=False)
    columns = ("OCTA input", "Ground truth", "Probability", "Binary prediction")
    for row_index, (case_id, image, mask, probability) in enumerate(rows):
        panels = (image, mask, probability, probability >= threshold)
        for column_index, panel in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(panel.numpy(), cmap="gray", vmin=0, vmax=1)
            title = columns[column_index]
            axis.set_title(f"Case {case_id} | {title}")
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    print(f"Device: {device}; AMP: {use_amp}")

    loaders = make_loaders(
        root=args.data_root,
        scan_size=args.scan_size,
        projection=args.projection,
        target=args.target,
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    print("Split sizes:", {name: len(loader.dataset) for name, loader in loaders.items()})

    model = UNet(base=args.base_channels).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_dice = -1.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint["best_dice"])

    writer = SummaryWriter(args.output_dir / "tensorboard")
    history_path = args.output_dir / "history.csv"
    if start_epoch == 1:
        with history_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                ["epoch", "learning_rate"]
                + [f"train_{name}" for name in METRIC_NAMES]
                + [f"val_{name}" for name in METRIC_NAMES]
            )

    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            args.threshold,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            description=f"Epoch {epoch} train",
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            args.threshold,
            use_amp=use_amp,
            description=f"Epoch {epoch} val",
        )
        scheduler.step(val_metrics["dice"])
        learning_rate = optimizer.param_groups[0]["lr"]

        with history_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [epoch, learning_rate]
                + [train_metrics[name] for name in METRIC_NAMES]
                + [val_metrics[name] for name in METRIC_NAMES]
            )
        for name in METRIC_NAMES:
            writer.add_scalars(name, {"train": train_metrics[name], "val": val_metrics[name]}, epoch)
        writer.add_scalar("learning_rate", learning_rate, epoch)
        print(
            f"Epoch {epoch:03d} | train loss={train_metrics['loss']:.4f} "
            f"dice={train_metrics['dice']:.4f} | val loss={val_metrics['loss']:.4f} "
            f"dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f}"
        )

        improved = val_metrics["dice"] > best_dice
        if improved:
            best_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_dice,
                args,
            )
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_dice,
            args,
        )
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    best_checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        args.threshold,
        use_amp=use_amp,
        description="Test",
    )
    test_metrics["best_epoch"] = int(best_checkpoint["epoch"])
    write_json(args.output_dir / "test_metrics.json", test_metrics)
    save_predictions(
        model,
        loaders["test"],
        device,
        args.threshold,
        args.output_dir / "test_predictions.png",
        args.num_visualizations,
    )
    writer.close()
    print("Test metrics:", test_metrics)
    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
