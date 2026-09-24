"""Early fusion and single-modality controls, using the baseline training core."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import PairedDataset
from octa.engine import run_epoch
from octa.losses import BCEDiceLoss
from octa.model import UNet
from octa.utils import seed_everything, write_json

HERE = Path(__file__).resolve().parent
METRICS = ("loss", "dice", "iou", "precision", "recall", "specificity")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=HERE.parent / "data/OCTA-500数据集")
    p.add_argument("--mode", choices=("octa", "oct", "fusion"), default="fusion")
    p.add_argument(
        "--target",
        choices=("GT_Capillary", "GT_Artery", "GT_Vein", "GT_LargeVessel"),
        default="GT_Capillary",
    )
    p.add_argument("--scan-size", choices=("3mm", "6mm"), default="3mm")
    p.add_argument("--slab", choices=("ILM_OPL", "FULL", "OPL_BM"), default="ILM_OPL")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--check-data", action="store_true", help="Audit all pairs and write a training-case preview; no training")
    p.add_argument("--skip-test", action="store_true", help="For smoke tests or development; do not evaluate the held-out test set")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or not 0 <= args.threshold <= 1:
        p.error("Invalid epochs, batch-size, workers or threshold")
    if args.output_dir is None:
        args.output_dir = HERE / "runs" / f"{args.mode}_{args.scan_size}_{args.slab}_{args.target}_seed{args.seed}"
    return args


@torch.no_grad()
def plot_cases(dataset, path, model=None, device="cpu", threshold=0.5, count=4):
    if model is not None:
        model.eval()
    count = min(count, len(dataset))
    columns = 3 if model is None else 6
    fig, axes = plt.subplots(count, columns, figsize=(3 * columns, 3 * count), squeeze=False)
    for row in range(count):
        sample = dataset[row]
        pair, mask = sample["pair"].numpy(), sample["mask"][0].numpy()
        panels = [pair[0], pair[1], mask]
        titles = ["OCTA input", "OCT input", "Ground truth"]
        if model is not None:
            prob = torch.sigmoid(model(sample["image"].unsqueeze(0).to(device)))[0, 0].cpu().numpy()
            pred, truth = prob >= threshold, mask > 0
            error = np.stack([pred & ~truth, pred & truth, ~pred & truth], axis=-1).astype(float)
            panels += [prob, pred, error]
            titles += ["Probability", f"Prediction t={threshold:g}", "Error: TP green / FP red / FN blue"]
        for col, (panel, title) in enumerate(zip(panels, titles)):
            axes[row, col].imshow(panel, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{sample['case_id']} | {title}", fontsize=9)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = arguments()
    seed_everything(args.seed)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new --output-dir; existing results protected: {output}")
    datasets = {split: PairedDataset(args.data_root, split, args.mode, args.target,
                                    args.scan_size, args.slab, augment=split == "train" and not args.check_data)
                for split in ("train", "val", "test")}
    output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["in_channels"] = 2 if args.mode == "fusion" else 1
    config["channel_order"] = ["OCTA", "OCT"] if args.mode == "fusion" else [args.mode.upper()]
    config["split_ids"] = {s: d.ids for s, d in datasets.items()}
    write_json(output / "config.json", config)
    if args.check_data:
        report = {}
        for split, dataset in datasets.items():
            shapes = set()
            for i in range(len(dataset)):
                sample = dataset[i]
                shapes.add(tuple(sample["image"].shape))
                if not torch.isfinite(sample["image"]).all():
                    raise ValueError(f"Nonfinite data: {sample['case_id']}")
            report[split] = {"count": len(dataset), "shapes": sorted(shapes)}
        report["alignment_note"] = "Matching ID and dimensions are checked; anatomical registration still needs visual inspection."
        write_json(output / "data_check.json", report)
        plot_cases(datasets["train"], output / "paired_preview.png")
        print(json.dumps(report, indent=2))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=s == "train",
                            num_workers=args.workers, pin_memory=device.type == "cuda",
                            persistent_workers=args.workers > 0) for s, d in datasets.items()}
    model = UNet(in_channels=config["in_channels"], base=args.base_channels).to(device)
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["device"] = str(device)
    write_json(output / "config.json", config)
    loss = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best, stale = -1.0, 0
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "learning_rate"] +
                                [f"{s}_{m}" for s in ("train", "val") for m in METRICS])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
            for split in ("train", "val"):
                values = run_epoch(model, loaders[split], loss, device, args.threshold,
                                   optimizer=optimizer if split == "train" else None,
                                   scaler=scaler, use_amp=use_amp, description=f"{epoch} {split}")
                row.update({f"{split}_{k}": v for k, v in values.items()})
            scheduler.step(row["val_dice"])
            writer.writerow(row)
            stream.flush()
            improved = row["val_dice"] > best
            if improved:
                best, stale = row["val_dice"], 0
            else:
                stale += 1
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                          "epoch": epoch, "best_dice": best, "args": config}
            if improved:
                torch.save(checkpoint, output / "best.pt")
            torch.save(checkpoint, output / "last.pt")
            print(f"Epoch {epoch}: train Dice={row['train_dice']:.4f}, val Dice={row['val_dice']:.4f}")
            if args.patience > 0 and stale >= args.patience:
                break
    checkpoint = torch.load(output / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    if not args.skip_test:
        metrics = run_epoch(model, loaders["test"], loss, device, args.threshold,
                            use_amp=use_amp, description="Test (best validation checkpoint)")
        metrics["best_epoch"] = checkpoint["epoch"]
        write_json(output / "test_metrics.json", metrics)
        plot_cases(datasets["test"], output / "test_predictions.png", model, device, args.threshold)
        print(metrics)
    else:
        plot_cases(datasets["val"], output / "val_predictions.png", model, device, args.threshold)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
