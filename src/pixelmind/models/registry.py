"""Model registry: the models the API may serve, discovered from evaluated results.

A model is servable only if it went through the benchmark protocol: its result JSON
(artifacts/experiments/levircd/{name}.json) provides the decision threshold chosen on
val and its test metrics. Weights are loaded lazily, on first use.
"""

import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pixelmind import __version__
from pixelmind.models.baselines import image_diff

Scorer = Callable[[np.ndarray, np.ndarray], np.ndarray]


def file_version(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]


@dataclass
class ModelEntry:
    name: str
    kind: str  # baseline | ml | dl
    version: str
    threshold: int  # on the uint8 score, chosen on the validation split
    test_metrics: dict
    loader: Callable[[], Scorer] = field(repr=False)
    _scorer: Scorer | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def scorer(self) -> Scorer:
        with self._lock:
            if self._scorer is None:
                self._scorer = self.loader()
            return self._scorer

    def summary(self) -> dict:
        keys = ("f1", "iou", "precision", "recall", "ap")
        return {
            "name": self.name,
            "kind": self.kind,
            "version": self.version,
            "threshold": self.threshold,
            "test_metrics": {k: self.test_metrics.get(k) for k in keys},
        }


def _gbm_loader(path: Path) -> Callable[[], Scorer]:
    def load() -> Scorer:
        import joblib

        from pixelmind.models.baselines import gbm

        return gbm.make_scorer(joblib.load(path))

    return load


def _dl_loader(path: Path) -> Callable[[], Scorer]:
    def load() -> Scorer:
        from pixelmind.models.dl.inference import load_checkpoint, make_scorer, pick_device

        device = pick_device()
        model, _ = load_checkpoint(path, device)
        return make_scorer(model, device)

    return load


class ModelRegistry:
    def __init__(self, entries: list[ModelEntry]) -> None:
        self.entries = {e.name: e for e in entries}

    @classmethod
    def discover(cls, artifacts_dir: Path) -> "ModelRegistry":
        results_dir = artifacts_dir / "experiments" / "levircd"
        entries = []
        for f in sorted(results_dir.glob("*.json")):
            r = json.loads(f.read_text())
            name, common = f.stem, {"threshold": r["threshold"], "test_metrics": r["test"]}
            if name in image_diff.SCORERS:
                scorer = image_diff.SCORERS[name]
                entries.append(
                    ModelEntry(name, "baseline", f"pixelmind-{__version__}", **common,
                               loader=lambda s=scorer: s)
                )  # fmt: skip
            elif name == "gbm" and (path := results_dir / "gbm.joblib").exists():
                entries.append(ModelEntry(name, "ml", file_version(path), **common,
                                          loader=_gbm_loader(path)))  # fmt: skip
            elif ckpt := (r.get("extra") or {}).get("checkpoint"):
                path = artifacts_dir / "models" / ckpt
                if path.exists():
                    entries.append(ModelEntry(name, "dl", file_version(path), **common,
                                              loader=_dl_loader(path)))  # fmt: skip
        return cls(entries)

    def default(self) -> ModelEntry | None:
        """Best test F1 among available models."""
        return max(self.entries.values(), key=lambda e: e.test_metrics.get("f1") or 0, default=None)

    def get(self, name: str | None) -> ModelEntry:
        entry = self.default() if name is None else self.entries.get(name)
        if entry is None:
            available = ", ".join(self.entries) or "none"
            raise KeyError(f"unknown model {name!r} (available: {available})")
        return entry
