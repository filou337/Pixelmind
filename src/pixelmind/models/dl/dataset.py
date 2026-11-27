"""PyTorch dataset: random crops of T1/T2/mask with synchronized augmentations."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pixelmind.data.loader import load_pair


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 -> float32 3xHxW in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().div_(255)


def augment(arrays: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    """Same geometric transform for T1, T2 and mask (dossier §6.5): flips + 90° rotations."""
    k = int(rng.integers(4))
    flip = bool(rng.integers(2))
    out = []
    for a in arrays:
        a = np.rot90(a, k, axes=(0, 1))
        out.append(a[:, ::-1] if flip else a)
    return out


class CropDataset(Dataset):
    """`crops_per_image` random crops per pair and per epoch.

    cache=True keeps decoded images in RAM (~7 MB per pair, ~3.6 GB for train):
    use it on Kaggle/Colab, not on a small laptop.
    """

    def __init__(
        self,
        root: Path,
        records: list[dict],
        crop: int = 256,
        crops_per_image: int = 8,
        train: bool = True,
        cache: bool = False,
        seed: int = 0,
    ) -> None:
        self.root, self.records, self.crop = root, records, crop
        self.crops_per_image, self.train, self.seed = crops_per_image, train, seed
        # preloaded in the main process: DataLoader workers (fork) then share the pages
        # instead of each building its own copy
        self._cache = [load_pair(root, r) for r in records] if cache else None
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.records) * self.crops_per_image

    def _pair(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cache is None:
            return load_pair(self.root, self.records[idx])
        return self._cache[idx]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # deterministic per (seed, epoch, item): reproducible and different every epoch
        rng = np.random.default_rng([self.seed, self._epoch, i])
        t1, t2, mask = self._pair(i // self.crops_per_image)
        h, w = mask.shape
        c = self.crop
        y, x = int(rng.integers(h - c + 1)), int(rng.integers(w - c + 1))
        arrays = [a[y : y + c, x : x + c] for a in (t1, t2, mask)]
        if self.train:
            arrays = augment(arrays, rng)
        t1, t2, mask = arrays
        m = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return to_tensor(t1), to_tensor(t2), m
