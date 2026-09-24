"""Train UGR-Fusion and matched controls on the established 3 mm split."""
import argparse
import csv
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from common import ROOT, PairedDataset
from model import build_model
from losses import objective
from octa.metrics import BinarySegmentationMetrics
from octa.utils import seed_everything, write_json

HERE = Path(__file__).resolve().parent


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("ugr", "octa", "early"), default="ugr")
    p.add_argument(
        "--target",
        choices=("GT_Capillary", "GT_Artery", "GT_Vein", "GT_LargeVessel"),
        default="GT_Capillary",
    )
    p.add_argument("--data-root", type=Path, default=ROOT / "data/OCTA-500数据集")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--oct-base", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--uncertainty", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aux-source", choices=("oct", "octa"), default="oct")
    p.add_argument("--aux-weight", type=float, default=0.3)
    p.add_argument("--retention-weight", type=float, default=0.1)
    p.add_argument("--cldice-weight", type=float, default=0.0)
    p.add_argument("--correction-limit", type=float, default=2.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--skip-test", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.base_channels, args.oct_base) < 1 or args.workers < 0:
        p.error("Invalid size/epoch/worker setting")
    if min(args.aux_weight, args.retention_weight, args.cldice_weight, args.warmup) < 0:
        p.error("Loss weights and warmup must be nonnegative")
    if not 0 < args.threshold < 1 or args.correction_limit <= 0:
        p.error("Invalid threshold or correction limit")
    if args.output_dir is None:
        args.output_dir = HERE / "runs" / f"{args.model}_{args.target}_seed{args.seed}"
    return args


def epoch_run(model, loader, device, config, optimizer=None, scaler=None, strength=1.0):
    training = optimizer is not None
    model.train(training)
    metric = BinarySegmentationMetrics(config["threshold"])
    sums, count = {}, 0
    for batch in tqdm(loader, leave=False, desc="Train" if training else "Eval"):
        x, y = batch["image"].to(device), batch["mask"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=config["amp"] and device.type == "cuda"):
                out = model.forward_details(x, strength=strength)
            # Loss reductions and soft morphology in float32.
            loss, parts = objective(out, y, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training/evaluation loss")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        metric.update(out["logits"].detach(), y)
        values = {"loss": loss.detach(), **parts}
        if "gate" in out:
            values.update(gate_mean=out["gate"].detach().mean(),
                          correction_abs=out["correction"].detach().abs().mean())
        for name, value in values.items():
            sums[name] = sums.get(name, 0) + float(value) * x.shape[0]
        count += x.shape[0]
    return {**metric.compute(), **{k: v / count for k, v in sums.items()}}


def main():
    args = arguments()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(scan_size="3mm", slab="ILM_OPL", channel_order=["OCTA", "OCT"])
    seed_everything(args.seed)
    datasets = {s: PairedDataset(args.data_root, s, mode="fusion", target=args.target, augment=s == "train")
                for s in ("train", "val", "test")}
    # Explicit generators keep sample order independent of model parameter count.
    loaders = {s: DataLoader(ds, batch_size=args.batch_size, shuffle=s == "train",
                            num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                            generator=torch.Generator().manual_seed(args.seed + i),
                            persistent_workers=args.workers > 0)
               for i, (s, ds) in enumerate(datasets.items())}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    seed_everything(args.seed)  # Align worker=0 augmentation RNG across control architectures.
    config["split_ids"] = {s: ds.ids for s, ds in datasets.items()}
    config["parameters"] = sum(p.numel() for p in model.parameters())
    config["torch_version"], config["device"] = str(torch.__version__), str(device)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    print(f"Model={args.model}; parameters={config['parameters']}; device={device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best, stale, writer = -1., 0, None
    with (output / "history.csv").open("w", newline="") as stream:
        for epoch in range(1, args.epochs + 1):
            strength = min(1., epoch / max(1, args.warmup))
            row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "strength": strength}
            train = epoch_run(model, loaders["train"], device, config, optimizer, scaler, strength)
            val = epoch_run(model, loaders["val"], device, config)  # Full-strength deployment inference.
            row.update({f"{split}_{k}": v for split, result in (("train", train), ("val", val)) for k, v in result.items()})
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            scheduler.step(val["dice"])
            improved = val["dice"] > best
            best, stale = (val["dice"], 0) if improved else (best, stale + 1)
            state = {"model": model.state_dict(), "args": config, "epoch": epoch, "best_dice": best,
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()}
            if improved:
                torch.save(state, output / "best.pt")
            torch.save(state, output / "last.pt")
            print(f"Epoch {epoch}: train Dice={train['dice']:.4f}; val Dice={val['dice']:.4f}; stale={stale}")
            if args.patience > 0 and stale >= args.patience:
                print("Early stopping")
                break
    state = torch.load(output / "best.pt", map_location=device)
    model.load_state_dict(state["model"])
    split = "val" if args.skip_test else "test"
    metrics = epoch_run(model, loaders[split], device, config)
    metrics["best_epoch"] = state["epoch"]
    write_json(output / f"{split}_metrics.json", metrics)
    from evaluate import save_preview
    save_preview(model, datasets[split], output / f"{split}_predictions.png", device, args.threshold)
    print(metrics)


if __name__ == "__main__":
    main()
