"""Inference-only modality ablations; validation is the default research split."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from common import PairedDataset
from model import build_model
from octa.metrics import BinarySegmentationMetrics
from octa.utils import write_json


@torch.no_grad()
def save_preview(model, dataset, path, device, threshold=0.5):
    model.eval()
    count = min(4, len(dataset))
    fig, axes = plt.subplots(count, 8, figsize=(24, count * 3), squeeze=False)
    for i in range(count):
        batch = dataset[i]
        out = model.forward_details(batch["image"][None].to(device))
        p = out["logits"].sigmoid()[0, 0].cpu().numpy()
        a = out["anchor"].sigmoid()[0, 0].cpu().numpy()
        y = batch["mask"][0].numpy() > 0
        pred = p >= threshold
        error = np.stack((pred & ~y, pred & y, ~pred & y), -1).astype(float)
        gate = out.get("gate", torch.zeros_like(out["logits"]))[0, 0].cpu().numpy()
        correction = out.get("correction", torch.zeros_like(out["logits"]))[0, 0].cpu().numpy()
        panels = [batch["pair"][0], batch["pair"][1], y, a, pred, error, gate, correction]
        titles = ["OCTA", "OCT", "Ground truth", "Anchor probability", "Prediction", "TP green / FP red / FN blue", "Mean gate", "Logit correction"]
        for j, (panel, title) in enumerate(zip(panels, titles)):
            cmap, lo, hi = ("coolwarm", -2, 2) if j == 7 else ("gray", 0, 1)
            axes[i, j].imshow(panel, cmap=cmap, vmin=lo, vmax=hi)
            axes[i, j].set_title(f"{batch['case_id']} | {title}", fontsize=8)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class Perturbed(Dataset):
    def __init__(self, dataset, condition, seed):
        self.dataset, self.condition = dataset, condition
        # A randomized cycle ensures no sample receives its own OCT.
        order = np.random.default_rng(seed).permutation(len(dataset))
        self.mapping = np.empty(len(dataset), dtype=int)
        self.mapping[order] = np.roll(order, 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        batch = dict(self.dataset[index])
        x = batch["image"].clone()
        if self.condition == "zero":
            x[1].zero_()
        elif self.condition == "shuffle":
            x[1] = self.dataset[int(self.mapping[index])]["image"][1]
        batch["image"] = x
        return batch


@torch.no_grad()
def evaluate(model, loader, device, threshold, use_amp=False):
    model.eval()
    micro, records = BinarySegmentationMetrics(threshold), []
    for batch in loader:
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            z = model(batch["image"].to(device)).cpu()
        y = batch["mask"]
        micro.update(z, y)
        for i, case in enumerate(batch["case_id"]):
            meter = BinarySegmentationMetrics(threshold)
            meter.update(z[i:i+1], y[i:i+1])
            records.append({"case_id": case, **meter.compute()})
    return micro.compute(), records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use an empty evaluation output directory")
    state = torch.load(args.run_dir / "best.pt", map_location="cpu")
    config = state["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    dataset = PairedDataset(args.data_root or config["data_root"], args.split,
                            mode="fusion", target=config["target"], augment=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"split": args.split, "checkpoint": str(args.run_dir / "best.pt"), "best_epoch": state["epoch"],
              "note": "Zero OCT is distribution shift; shuffle probes pairing. Neither alone proves useful OCT contribution."}
    for condition, seed in (("normal", 0), ("zero", 0), ("shuffle", 42), ("shuffle", 43), ("shuffle", 44)):
        name = f"{condition}_{seed}" if condition == "shuffle" else condition
        loader = DataLoader(Perturbed(dataset, condition, seed), batch_size=8, num_workers=0)
        metrics, records = evaluate(model, loader, device, config["threshold"], config["amp"])
        report[name] = metrics
        with (args.output_dir / f"{name}_per_case.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write_json(args.output_dir / "ablation_metrics.json", report)
    save_preview(model, dataset, args.output_dir / "predictions_and_gates.png", device, config["threshold"])
    print(report)


if __name__ == "__main__":
    main()
