"""Load T1/T2/mask pairs listed in the manifest."""

from pathlib import Path

import numpy as np
from PIL import Image

from pixelmind.data.quality import binarize_mask, normalize_mask


def usable(records: list[dict], split: str) -> list[dict]:
    """Pairs of a split that passed blocking quality rules (no FAIL)."""
    return [r for r in records if r["split"] == split and r["quality_status"] != "FAIL"]


def load_pair(root: Path, record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t1 uint8 HxWx3, t2 uint8 HxWx3, mask bool HxW)."""

    def read(role: str) -> np.ndarray:
        with Image.open(root / record["paths"][role]) as img:
            return np.asarray(img)

    mask, _ = normalize_mask(read("mask"))
    return read("t1"), read("t2"), binarize_mask(mask)
