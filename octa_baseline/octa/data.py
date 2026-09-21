from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


Split = Literal["train", "val", "test"]

# Common OCTA-500 subject-level splits used by the vessel-segmentation literature.
SPLIT_RANGES: dict[str, dict[Split, tuple[int, int]]] = {
    "3mm": {
        "train": (10301, 10440),
        "val": (10441, 10450),
        "test": (10451, 10500),
    },
    "6mm": {
        "train": (10001, 10180),
        "val": (10181, 10200),
        "test": (10201, 10300),
    },
}


def case_ids(scan_size: str, split: Split) -> list[str]:
    if scan_size not in SPLIT_RANGES:
        raise ValueError(f"scan_size must be one of {tuple(SPLIT_RANGES)}, got {scan_size!r}")
    start, end = SPLIT_RANGES[scan_size][split]
    return [str(index) for index in range(start, end + 1)]


class RandomDihedral:
    """Apply paired 90-degree rotations and flips without mask interpolation."""

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = int(torch.randint(0, 4, ()).item())
        image = torch.rot90(image, k, dims=(-2, -1))
        mask = torch.rot90(mask, k, dims=(-2, -1))
        if bool(torch.rand(()) < 0.5):
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        if bool(torch.rand(()) < 0.5):
            image = torch.flip(image, dims=(-2,))
            mask = torch.flip(mask, dims=(-2,))
        return image.contiguous(), mask.contiguous()


class OCTA500Dataset(Dataset):
    """One OCTA projection and one binary BMP target for each subject."""

    def __init__(
        self,
        root: str | Path,
        scan_size: str,
        split: Split,
        projection: str = "OCTA(ILM_OPL)",
        target: str = "GT_Capillary",
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.scan_size = scan_size
        self.split = split
        self.projection = projection
        self.target = target
        self.ids = case_ids(scan_size, split)
        self.augment = RandomDihedral() if augment else None

        self.image_dir = self.root / scan_size / projection
        self.mask_dir = self.root / "labels" / target
        missing = [
            path
            for case_id in self.ids
            for path in (
                self.image_dir / f"{case_id}.bmp",
                self.mask_dir / f"{case_id}.bmp",
            )
            if not path.is_file()
        ]
        if missing:
            preview = "\n".join(str(path) for path in missing[:5])
            raise FileNotFoundError(
                f"Missing {len(missing)} required files. First missing paths:\n{preview}"
            )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        case_id = self.ids[index]
        with Image.open(self.image_dir / f"{case_id}.bmp") as source:
            image_array = np.asarray(source.convert("L"), dtype=np.float32).copy()
        with Image.open(self.mask_dir / f"{case_id}.bmp") as source:
            mask_array = np.asarray(source.convert("L"), dtype=np.uint8).copy()

        image = torch.from_numpy(image_array).unsqueeze(0).div_(255.0)
        mask = torch.from_numpy(mask_array > 0).unsqueeze(0).float()
        if self.augment is not None:
            image, mask = self.augment(image, mask)
        return {"image": image, "mask": mask, "case_id": case_id}


def make_loaders(
    root: str | Path,
    scan_size: str,
    projection: str,
    target: str,
    batch_size: int,
    workers: int,
    pin_memory: bool,
) -> dict[Split, DataLoader]:
    loaders: dict[Split, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = OCTA500Dataset(
            root=root,
            scan_size=scan_size,
            split=split,
            projection=projection,
            target=target,
            augment=split == "train",
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=workers > 0,
            drop_last=False,
        )
    return loaders

