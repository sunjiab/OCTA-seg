import torch

from octa.data import OCTA500Dataset, case_ids
from octa.losses import BCEDiceLoss
from octa.metrics import BinarySegmentationMetrics
from octa.model import UNet


def test_split_sizes() -> None:
    assert len(case_ids("3mm", "train")) == 140
    assert len(case_ids("3mm", "val")) == 10
    assert len(case_ids("3mm", "test")) == 50
    assert len(case_ids("6mm", "train")) == 180
    assert len(case_ids("6mm", "val")) == 20
    assert len(case_ids("6mm", "test")) == 100


def test_model_loss_and_backward() -> None:
    model = UNet(base=4)
    image = torch.rand(2, 1, 64, 64)
    mask = (torch.rand(2, 1, 64, 64) > 0.5).float()
    logits = model(image)
    assert logits.shape == mask.shape
    loss = BCEDiceLoss()(logits, mask)
    loss.backward()
    assert torch.isfinite(loss)


def test_perfect_metrics() -> None:
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    logits = torch.where(target > 0, torch.tensor(20.0), torch.tensor(-20.0))
    metrics = BinarySegmentationMetrics()
    metrics.update(logits, target)
    for value in metrics.compute().values():
        assert abs(value - 1.0) < 1e-6


def test_real_sample(dataset_root) -> None:
    dataset = OCTA500Dataset(dataset_root, "3mm", "val")
    sample = dataset[0]
    assert sample["image"].shape == (1, 304, 304)
    assert sample["mask"].shape == (1, 304, 304)
    assert 0 <= sample["image"].min() <= sample["image"].max() <= 1
    assert set(torch.unique(sample["mask"]).tolist()).issubset({0.0, 1.0})

