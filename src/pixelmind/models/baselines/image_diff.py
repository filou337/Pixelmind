"""Baseline 0: absolute image difference, then a threshold (tuned on val).

Each scorer maps (t1, t2) uint8 HxWx3 -> uint8 HxW change score.
"""

import numpy as np
from scipy.ndimage import uniform_filter

# |z1 - z2| averaged over channels rarely exceeds ~4 std; map [0, 4] -> [0, 255]
_Z_RANGE = 4.0


def standardize(img: np.ndarray) -> np.ndarray:
    """Per-image, per-channel zero mean / unit std: removes global illumination shifts."""
    x = img.astype(np.float32)
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def to_uint8(x: np.ndarray, vmax: float) -> np.ndarray:
    return np.clip(x * (255.0 / vmax), 0, 255).astype(np.uint8)


def absdiff(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Raw RGB absolute difference, averaged over channels."""
    d = np.abs(t1.astype(np.int16) - t2.astype(np.int16)).mean(axis=2)
    return d.astype(np.uint8)


def normdiff(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Difference after per-image standardization (radiometric normalization)."""
    return to_uint8(np.abs(standardize(t1) - standardize(t2)).mean(axis=2), _Z_RANGE)


def normdiff_smooth(t1: np.ndarray, t2: np.ndarray, size: int = 7) -> np.ndarray:
    """normdiff + local averaging: suppresses isolated pixel noise and small misalignments."""
    d = np.abs(standardize(t1) - standardize(t2)).mean(axis=2)
    return to_uint8(uniform_filter(d, size=size), _Z_RANGE)


SCORERS = {"absdiff": absdiff, "normdiff": normdiff, "normdiff_smooth": normdiff_smooth}
