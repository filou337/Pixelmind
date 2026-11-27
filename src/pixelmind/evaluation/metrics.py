"""Pixel-level change-detection metrics, streamed over images.

Scores are quantized to uint8 (0..255). We accumulate one histogram of scores for
changed pixels and one for unchanged pixels: every threshold's confusion matrix,
the PR curve and average precision then come out exactly, in O(256) memory,
whatever the number of pixels. Prediction rule everywhere: changed <=> score >= t.
"""

import numpy as np

LEVELS = 256


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Precision/recall/F1/IoU; None when undefined (e.g. no positives at all)."""

    def div(a: float, b: float) -> float | None:
        return float(a / b) if b else None

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "recall": recall,
        "f1": div(2 * tp, 2 * tp + fp + fn),  # == Dice at pixel level
        "iou": div(tp, tp + fp + fn),
        "fpr": div(fp, fp + tn),
    }


def counts_at(scores: np.ndarray, mask: np.ndarray, threshold: int) -> tuple[int, int, int, int]:
    pred = scores >= threshold
    tp = int(np.count_nonzero(pred & mask))
    fp = int(np.count_nonzero(pred & ~mask))
    fn = int(np.count_nonzero(~pred & mask))
    return tp, fp, fn, int(mask.size - tp - fp - fn)


class ScoreHistogram:
    def __init__(self) -> None:
        self.pos = np.zeros(LEVELS, dtype=np.int64)
        self.neg = np.zeros(LEVELS, dtype=np.int64)

    def add(self, scores: np.ndarray, mask: np.ndarray) -> None:
        if scores.dtype != np.uint8:
            raise TypeError(f"scores must be uint8, got {scores.dtype}")
        self.pos += np.bincount(scores[mask], minlength=LEVELS)
        self.neg += np.bincount(scores[~mask], minlength=LEVELS)

    def _cumulative(self) -> tuple[np.ndarray, np.ndarray]:
        """tp[t], fp[t] = pixels with score >= t, for t in 0..255."""
        tp = np.cumsum(self.pos[::-1])[::-1]
        fp = np.cumsum(self.neg[::-1])[::-1]
        return tp, fp

    def metrics_at(self, threshold: int) -> dict:
        tp, fp = self._cumulative()
        P, N = int(self.pos.sum()), int(self.neg.sum())
        t, f = int(tp[threshold]), int(fp[threshold])
        return metrics_from_counts(t, f, P - t, N - f)

    def best_threshold(self) -> int:
        """Threshold in 1..255 maximizing F1 (t=0 would flag every pixel)."""
        tp, fp = self._cumulative()
        P = self.pos.sum()
        f1 = 2 * tp / np.maximum(2 * tp + fp + (P - tp), 1)
        return int(np.argmax(f1[1:]) + 1)

    def pr_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """(precision, recall) for thresholds 255 -> 0, recall increasing."""
        tp, fp = self._cumulative()
        tp, fp = tp[::-1], fp[::-1]
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
        recall = tp / max(int(self.pos.sum()), 1)
        return precision, recall

    def average_precision(self) -> float:
        """AP = sum_n (R_n - R_{n-1}) P_n (same definition as scikit-learn)."""
        precision, recall = self.pr_curve()
        return float(np.sum(np.diff(recall, prepend=0.0) * precision))
