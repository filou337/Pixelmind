"""Baseline 1: hand-crafted per-pixel features + gradient boosting.

HistGradientBoostingClassifier is scikit-learn's histogram-based boosting (same
family as XGBoost/LightGBM): fast to train and, unlike a deep Random Forest,
fast to predict on 1M pixels per image on a CPU.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy.ndimage import sobel, uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier

from pixelmind.data.loader import load_pair
from pixelmind.models.baselines.image_diff import standardize

FEATURES = [
    "t1_r", "t1_g", "t1_b",
    "t2_r", "t2_g", "t2_b",
    "zdiff_r", "zdiff_g", "zdiff_b",
    "zdiff_mean",
    "zdiff_local_mean_5", "zdiff_local_std_5",
    "zdiff_local_mean_15", "zdiff_local_std_15",
    "grad_t1", "grad_t2", "grad_absdiff",
    "t2_minus_t1_gray",
]  # fmt: skip


def _grad(gray: np.ndarray) -> np.ndarray:
    return np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))


def _local_mean_std(x: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    m = uniform_filter(x, size=size)
    var = uniform_filter(x * x, size=size) - m * m
    return m, np.sqrt(np.maximum(var, 0))


def pixel_features(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """(H*W, len(FEATURES)) float32 feature matrix."""
    a = t1.astype(np.float32) / 255
    b = t2.astype(np.float32) / 255
    zd = np.abs(standardize(t1) - standardize(t2))
    zm = zd.mean(axis=2)
    m5, s5 = _local_mean_std(zm, 5)
    m15, s15 = _local_mean_std(zm, 15)
    ga, gb = _grad(a.mean(axis=2)), _grad(b.mean(axis=2))
    cols = [
        *np.moveaxis(a, 2, 0), *np.moveaxis(b, 2, 0), *np.moveaxis(zd, 2, 0),
        zm, m5, s5, m15, s15, ga, gb, np.abs(ga - gb),
        b.mean(axis=2) - a.mean(axis=2),
    ]  # fmt: skip
    return np.stack([c.ravel() for c in cols], axis=1).astype(np.float32)


def sample_training_set(
    root: Path, records: Iterable[dict], per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Up to `per_class` changed + `per_class` unchanged pixels per image.

    Balanced sampling shifts predicted probabilities upward; this is fine because
    the decision threshold is tuned on the (unbalanced) validation split.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for r in records:
        t1, t2, mask = load_pair(root, r)
        feats = pixel_features(t1, t2)
        y = mask.ravel()
        for label in (True, False):
            idx = np.flatnonzero(y == label)
            if idx.size:
                pick = rng.choice(idx, size=min(per_class, idx.size), replace=False)
                xs.append(feats[pick])
                ys.append(np.full(pick.size, label))
    return np.concatenate(xs), np.concatenate(ys)


def train(X: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True,
        validation_fraction=0.1, random_state=seed,
    )  # fmt: skip
    return model.fit(X, y)


def make_scorer(model: HistGradientBoostingClassifier, chunk: int = 262_144):
    """Scorer (t1, t2) -> uint8 HxW probability of change, predicted in chunks (low RAM)."""

    def score(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        feats = pixel_features(t1, t2)
        proba = np.concatenate(
            [model.predict_proba(feats[i : i + chunk])[:, 1] for i in range(0, len(feats), chunk)]
        )
        return (proba * 255).round().astype(np.uint8).reshape(t1.shape[:2])

    return score
