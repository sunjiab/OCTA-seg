"""Paired en-face projections; reuse baseline splits and augmentation."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "octa_baseline"))
from octa.data import case_ids, RandomDihedral


class PairedDataset(Dataset):
    def __init__(self, root, split, mode="fusion", target="GT_Capillary",
                 scan_size="3mm", slab="ILM_OPL", augment=False):
        if mode not in ("octa", "oct", "fusion"):
            raise ValueError(mode)
        self.root = Path(root)
        self.ids = case_ids(scan_size, split)
        self.mode = mode
        self.folders = [self.root / scan_size / f"{m}({slab})" for m in ("OCTA", "OCT")]
        self.folders.append(self.root / "labels" / target)
        self.augment = RandomDihedral() if augment else None
        for case in self.ids:
            for folder in self.folders:
                path = folder / f"{case}.bmp"
                if not path.is_file():
                    raise FileNotFoundError(path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        case = self.ids[index]
        arrays = []
        for folder in self.folders:
            with Image.open(folder / f"{case}.bmp") as source:
                arrays.append(np.asarray(source.convert("L"), dtype=np.float32).copy())
        if len({a.shape for a in arrays}) != 1:
            raise ValueError(f"Shape mismatch for {case}: {[a.shape for a in arrays]}")
        pair = torch.from_numpy(np.stack(arrays[:2])) / 255.0
        mask = torch.from_numpy(arrays[2] > 0).unsqueeze(0).float()
        if self.augment:
            pair, mask = self.augment(pair, mask)
        image = pair if self.mode == "fusion" else pair[0:1] if self.mode == "octa" else pair[1:2]
        return {"image": image, "mask": mask, "case_id": case, "pair": pair}
