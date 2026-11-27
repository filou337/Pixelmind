"""Data quality rules for T1/T2/mask pairs (dossier d'architecture §5.2–5.3).

Rules are plain functions over arrays so they are testable without files.
Status = worst severity found: PASS < WARN < REVIEW < FAIL.
"""

from dataclasses import asdict, dataclass
from enum import IntEnum

import numpy as np


class Status(IntEnum):
    PASS = 0
    WARN = 1
    REVIEW = 2
    FAIL = 3


@dataclass(frozen=True)
class Issue:
    status: Status
    code: str
    message: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.name
        return d


# LEVIR-CD has a few annotation artefacts (values 156, 254...) on <= 0.013 % of pixels.
MAX_STRAY_MASK_FRACTION = 0.001


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Single rule used everywhere: {0,1} masks -> > 0, {0..255} masks -> >= 128."""
    return mask > 0 if mask.max(initial=0) <= 1 else mask >= 128


def overall_status(issues: list[Issue]) -> Status:
    return max((i.status for i in issues), default=Status.PASS)


def normalize_mask(mask: np.ndarray) -> tuple[np.ndarray | None, list[Issue]]:
    """Return a 2D mask, or None if it cannot be interpreted as a single-band mask."""
    if mask.ndim == 2:
        return mask, []
    if mask.ndim == 3 and all(
        np.array_equal(mask[..., 0], mask[..., c]) for c in range(1, mask.shape[2])
    ):
        return mask[..., 0], [
            Issue(Status.WARN, "mask_multichannel", f"mask has {mask.shape[2]} identical channels")
        ]
    return None, [
        Issue(Status.FAIL, "mask_invalid_shape", f"mask shape {mask.shape} is not single-band")
    ]


def check_pair(t1: np.ndarray, t2: np.ndarray, mask: np.ndarray) -> list[Issue]:
    """Per-pair rules: validity (RGB uint8), consistency (same size), binary mask."""
    issues: list[Issue] = []

    for name, img in (("t1", t1), ("t2", t2)):
        if img.ndim != 3 or img.shape[2] != 3:
            issues.append(
                Issue(Status.FAIL, f"{name}_not_rgb", f"{name} shape {img.shape}, expected HxWx3")
            )
        if img.dtype != np.uint8:
            issues.append(
                Issue(Status.FAIL, f"{name}_dtype", f"{name} dtype {img.dtype}, expected uint8")
            )
        elif img.size and img.std() < 1.0:
            issues.append(Issue(Status.WARN, f"{name}_constant", f"{name} is (almost) constant"))

    m, mask_issues = normalize_mask(mask)
    issues += mask_issues

    sizes = {"t1": t1.shape[:2], "t2": t2.shape[:2], "mask": mask.shape[:2]}
    if len(set(sizes.values())) > 1:
        issues.append(Issue(Status.FAIL, "size_mismatch", f"sizes differ: {sizes}"))

    if m is not None:
        values = set(np.unique(m).tolist())
        if not values <= {0, 1} and not values <= {0, 255}:
            stray = float(np.isin(m, [0, 255], invert=True).mean())
            shown = sorted(values)[:10]
            msg = f"mask values {shown}, {stray:.4%} of pixels outside {{0, 255}}"
            if stray <= MAX_STRAY_MASK_FRACTION:
                issues.append(Issue(Status.WARN, "mask_near_binary", msg + " (binarized >= 128)"))
            else:
                issues.append(Issue(Status.FAIL, "mask_not_binary", msg))

    if t1.shape == t2.shape and np.array_equal(t1, t2):
        issues.append(Issue(Status.WARN, "t1_equals_t2", "T1 and T2 are identical"))

    return issues


def check_duplicates(records: list[dict]) -> dict[str, list[Issue]]:
    """Dataset-level unicity: same image file used by several pairs.

    Across splits it is a leakage risk (FAIL); within one split, a WARN.
    Returns issues keyed by pair_id.
    """
    seen: dict[str, list[tuple[str, str, str]]] = {}
    for r in records:
        for role in ("t1", "t2"):
            seen.setdefault(r["sha256"][role], []).append((r["pair_id"], r["split"], role))

    out: dict[str, list[Issue]] = {}
    for entries in seen.values():
        pairs = {p for p, _, _ in entries}
        if len(pairs) < 2:
            continue
        splits = {s for _, s, _ in entries}
        status = Status.FAIL if len(splits) > 1 else Status.WARN
        code = "duplicate_across_splits" if len(splits) > 1 else "duplicate_within_split"
        for pair_id, _, role in entries:
            others = sorted(pairs - {pair_id})
            out.setdefault(pair_id, []).append(
                Issue(status, code, f"{role} image also used by {', '.join(others)}")
            )
    return out
