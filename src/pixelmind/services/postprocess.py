"""Post-processing of a change-score map: binary mask, connected regions, overlay."""

import numpy as np
from scipy import ndimage


def to_mask(scores: np.ndarray, threshold: int, min_region_px: int) -> np.ndarray:
    """scores >= threshold, without connected regions smaller than min_region_px."""
    mask = scores >= threshold
    if min_region_px <= 1 or not mask.any():
        return mask
    labels, n = ndimage.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= min_region_px
    keep[0] = False
    return keep[labels]


def regions(mask: np.ndarray, scores: np.ndarray, limit: int = 100) -> tuple[int, list[dict]]:
    """Connected changed regions, largest first: (count, up to `limit` regions)."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return 0, []
    idx = np.arange(1, n + 1)
    areas = ndimage.sum_labels(mask, labels, idx)
    mean_scores = ndimage.mean(scores, labels, idx)
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels)):
        out.append({
            "bbox": [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)],
            "area_px": int(areas[i]),
            "mean_score": round(float(mean_scores[i]) / 255, 3),
        })  # fmt: skip
    out.sort(key=lambda r: -r["area_px"])
    return n, out[:limit]


def overlay(image: np.ndarray, mask: np.ndarray, color=(255, 40, 40), alpha=0.5) -> np.ndarray:
    """RGB image with changed pixels tinted."""
    out = image.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, np.float32)
    return out.astype(np.uint8)
