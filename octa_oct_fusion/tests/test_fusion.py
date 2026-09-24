import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import PairedDataset
from octa.model import UNet
from octa.losses import BCEDiceLoss


def make_data(tmp_path):
    a = np.zeros((32, 32), dtype=np.uint8)
    a[3:15, 5:9] = 255
    for folder, array in (("3mm/OCTA(ILM_OPL)", a), ("3mm/OCT(ILM_OPL)", 255-a), ("labels/GT_Artery", a)):
        directory = tmp_path / folder
        directory.mkdir(parents=True)
        for case in range(10441, 10451):
            Image.fromarray(array).save(directory / f"{case}.bmp")
    return tmp_path


def test_pair_and_synchronized_augmentation(tmp_path):
    root = make_data(tmp_path)
    dataset = PairedDataset(root, "val", target="GT_Artery", augment=True)
    for _ in range(8):
        sample = dataset[0]
        assert torch.equal(sample["image"][0:1], sample["mask"])
        assert torch.equal(sample["image"][0] + sample["image"][1], torch.ones(32, 32))
    for mode, index in (("octa", 0), ("oct", 1)):
        sample = PairedDataset(root, "val", mode=mode, target="GT_Artery")[0]
        assert torch.equal(sample["image"], sample["pair"][index:index+1])


def test_missing_and_mismatched_pairs(tmp_path):
    root = make_data(tmp_path)
    path = root / "3mm/OCT(ILM_OPL)/10441.bmp"
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(path)
    dataset = PairedDataset(root, "val", target="GT_Artery")
    with pytest.raises(ValueError, match="Shape mismatch"):
        dataset[0]
    path.unlink()
    with pytest.raises(FileNotFoundError):
        PairedDataset(root, "val", target="GT_Artery")


def test_fusion_backward():
    torch.set_num_threads(1)
    model = UNet(in_channels=2, base=4)
    prediction = model(torch.rand(2, 2, 32, 32))
    assert prediction.shape == (2, 1, 32, 32)
    loss = BCEDiceLoss()(prediction, torch.ones_like(prediction))
    loss.backward()
    assert torch.isfinite(loss)
    gradient = model.stem[0].weight.grad
    assert gradient[:, 0].abs().sum() > 0
    assert gradient[:, 1].abs().sum() > 0
