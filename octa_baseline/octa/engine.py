from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import BinarySegmentationMetrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    description: str = "",
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics = BinarySegmentationMetrics(threshold)
    loss_sum = 0.0
    sample_count = 0

    progress = tqdm(loader, desc=description, leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with torch.set_grad_enabled(training), amp_context:
            logits = model(images)
            loss = criterion(logits, masks)

        if training:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        metrics.update(logits.detach(), masks)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    result = metrics.compute()
    result["loss"] = loss_sum / max(sample_count, 1)
    return result

