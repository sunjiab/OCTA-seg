from __future__ import annotations

import torch


class BinarySegmentationMetrics:
    """Dataset-level metrics accumulated from a global confusion matrix."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = torch.sigmoid(logits) >= self.threshold
        targets = targets >= 0.5
        self.tp += torch.logical_and(predictions, targets).sum().item()
        self.fp += torch.logical_and(predictions, ~targets).sum().item()
        self.fn += torch.logical_and(~predictions, targets).sum().item()
        self.tn += torch.logical_and(~predictions, ~targets).sum().item()

    def compute(self) -> dict[str, float]:
        eps = 1e-8
        return {
            "dice": (2 * self.tp) / (2 * self.tp + self.fp + self.fn + eps),
            "iou": self.tp / (self.tp + self.fp + self.fn + eps),
            "precision": self.tp / (self.tp + self.fp + eps),
            "recall": self.tp / (self.tp + self.fn + eps),
            "specificity": self.tn / (self.tn + self.fp + eps),
        }

