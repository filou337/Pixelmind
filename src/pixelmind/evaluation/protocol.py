"""Evaluation protocol shared by every model (dossier §7.2).

1. Threshold chosen on the validation split only (max F1).
2. Test metrics reported at that threshold, globally and per scene.
3. Error analysis by amount of change and on no-change scenes.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from pixelmind.data.loader import load_pair, usable
from pixelmind.evaluation.metrics import ScoreHistogram, counts_at, metrics_from_counts

Scorer = Callable[[np.ndarray, np.ndarray], np.ndarray]

# changed-pixel ratio buckets for error analysis
BUCKETS = [
    ("no change", 0.0, 0.0),
    ("< 1 %", 0.0, 0.01),
    ("1-5 %", 0.01, 0.05),
    ("> 5 %", 0.05, 1.0),
]


def bucket_of(ratio: float) -> str:
    if ratio == 0:
        return BUCKETS[0][0]
    return next(name for name, lo, hi in BUCKETS[1:] if lo < ratio <= hi)


def evaluate(scorer: Scorer, records: list[dict], root: Path) -> dict:
    val_hist = ScoreHistogram()
    for r in usable(records, "val"):
        t1, t2, mask = load_pair(root, r)
        val_hist.add(scorer(t1, t2), mask)
    threshold = val_hist.best_threshold()

    test_hist = ScoreHistogram()
    scenes = []
    for r in usable(records, "test"):
        t1, t2, mask = load_pair(root, r)
        scores = scorer(t1, t2)
        test_hist.add(scores, mask)
        m = metrics_from_counts(*counts_at(scores, mask, threshold))
        ratio = r["changed_pixel_ratio"]
        scenes.append(
            {"pair_id": r["pair_id"], "changed_ratio": ratio, "bucket": bucket_of(ratio), **m}
        )

    precision, recall = test_hist.pr_curve()
    return {
        "threshold": threshold,
        "val": {**val_hist.metrics_at(threshold), "ap": val_hist.average_precision()},
        "test": {**test_hist.metrics_at(threshold), "ap": test_hist.average_precision()},
        "test_pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
        "test_scenes": scenes,
        "error_analysis": error_analysis(scenes),
    }


def error_analysis(scenes: list[dict]) -> dict:
    by_bucket = {}
    for name, _, _ in BUCKETS:
        group = [s for s in scenes if s["bucket"] == name]
        if not group:
            continue
        tot = {k: sum(s[k] for s in group) for k in ("tp", "fp", "fn", "tn")}
        by_bucket[name] = {"scenes": len(group), **metrics_from_counts(**tot)}
    # F1 is only meaningful where something changed; no-change scenes are covered by
    # the "no change" bucket FPR (any false positive would give them F1 = 0)
    scored = [s for s in scenes if s["changed_ratio"] > 0 and s["f1"] is not None]
    worst = sorted(scored, key=lambda s: s["f1"])[:10]
    return {
        "by_change_bucket": by_bucket,
        "scene_f1_median": float(np.median([s["f1"] for s in scored])) if scored else None,
        "worst_scenes": [
            {k: s[k] for k in ("pair_id", "changed_ratio", "f1", "precision", "recall")}
            for s in worst
        ],
    }
