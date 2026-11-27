"""Input drift (dossier §7.4, §11.4): per-image statistics compared with LEVIR-CD train.

Without immediate ground truth, we watch the input distribution. A drift raises an alert
and a review, never an automatic retraining.

python -m pixelmind.monitoring.drift   # (re)builds docs/data/levircd_reference_stats.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

FEATURES = {
    "t2_brightness": "luminosité moyenne de T2 (0-255)",
    "t2_contrast": "contraste de T2 (écart type des niveaux de gris)",
    "t2_red_share": "part du rouge dans T2",
    "t2_blue_share": "part du bleu dans T2",
    "pair_brightness_shift": "écart de luminosité moyenne T2 − T1",
    "pair_abs_diff": "différence absolue moyenne T1/T2 (niveaux de gris)",
}
N_BINS = 10
MIN_WINDOW = 30
# PSI measures the size of the shift, but on small windows it is inflated by sampling noise
# (≈ (bins - 1) / n under no drift: ~0.3 for 30 analyses). A level therefore needs both a
# large PSI and a significant chi-square test against the reference deciles.
LEVELS = ((0.25, 0.01, "significative"), (0.10, 0.05, "modérée"))


def image_stats(t1: np.ndarray, t2: np.ndarray) -> dict[str, float]:
    """Cheap statistics of a T1/T2 pair (uint8 HxWx3), stored with every analysis."""
    g1 = t1.astype(np.float32).mean(axis=2)
    g2 = t2.astype(np.float32).mean(axis=2)
    rgb = t2.reshape(-1, 3).astype(np.float64).mean(axis=0)
    total = max(rgb.sum(), 1e-6)
    stats = {
        "t2_brightness": float(g2.mean()),
        "t2_contrast": float(g2.std()),
        "t2_red_share": float(rgb[0] / total),
        "t2_blue_share": float(rgb[2] / total),
        "pair_brightness_shift": float(g2.mean() - g1.mean()),
        "pair_abs_diff": float(np.abs(g2 - g1).mean()) if g1.shape == g2.shape else float("nan"),
    }
    return {k: round(v, 4) for k, v in stats.items()}


def build_reference(samples: list[dict[str, float]], source: str) -> dict:
    """Decile bin edges per feature: each reference bin holds 10 % of the data."""
    ref = {"source": source, "n": len(samples), "features": {}}
    for f in FEATURES:
        x = np.array([s[f] for s in samples])
        edges = np.quantile(x, np.linspace(0, 1, N_BINS + 1))[1:-1]
        ref["features"][f] = {
            "edges": [round(float(e), 4) for e in edges],
            "mean": round(float(x.mean()), 4),
            "p01": round(float(np.quantile(x, 0.01)), 4),
            "p99": round(float(np.quantile(x, 0.99)), 4),
        }
    return ref


def _counts(reference: dict, values: list[float]) -> np.ndarray:
    edges = np.array(reference["edges"])
    return np.bincount(np.searchsorted(edges, values, side="right"), minlength=N_BINS)


def psi(reference: dict, values: list[float]) -> float:
    """Population Stability Index against the reference deciles."""
    cur = np.clip(_counts(reference, values) / max(len(values), 1), 1e-4, None)
    ref = np.full(N_BINS, 1 / N_BINS)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def chi2_pvalue(reference: dict, values: list[float], n_ref: int) -> float:
    """Two-sample chi-square test (reference deciles vs current bins).

    Two-sample rather than goodness of fit: the decile edges are themselves estimated on
    `n_ref` reference images (445 for LEVIR-CD train), which adds uncertainty.
    """
    from scipy.stats import chi2_contingency

    cur = _counts(reference, values)
    ref = np.full(N_BINS, n_ref / N_BINS)
    keep = (cur + ref) > 0
    return float(chi2_contingency(np.vstack([ref[keep], cur[keep]])).pvalue)


def drift_level(psi_value: float, p_value: float) -> str:
    for min_psi, max_p, name in LEVELS:
        if psi_value >= min_psi and p_value < max_p:
            return name
    return "stable"


def drift_report(reference: dict | None, samples: list[dict[str, float]]) -> dict:
    if reference is None:
        return {"status": "no_reference", "n": len(samples), "features": {}, "alert": False}
    if len(samples) < MIN_WINDOW:
        return {"status": "insufficient_data", "n": len(samples), "min_n": MIN_WINDOW,
                "features": {}, "alert": False}  # fmt: skip
    features = {}
    for f, label in FEATURES.items():
        vals = [s[f] for s in samples if f in s and np.isfinite(s[f])]
        if not vals:
            continue
        value = psi(reference["features"][f], vals)
        p_value = chi2_pvalue(reference["features"][f], vals, reference["n"])
        features[f] = {"label": label, "psi": round(value, 3), "p_value": round(p_value, 4),
                       "level": drift_level(value, p_value),
                       "reference_mean": reference["features"][f]["mean"],
                       "current_mean": round(float(np.mean(vals)), 4)}  # fmt: skip
    alert = any(v["level"] == "significative" for v in features.values())
    return {
        "status": "alert" if alert else "ok",
        "n": len(samples),
        "reference": {"source": reference["source"], "n": reference["n"]},
        "features": features,
        "alert": alert,
        "action": "Dérive significative : revue humaine des analyses récentes et vérification "
        "du domaine d'emploi du modèle. Pas de réentraînement automatique (dossier §11.4)."
        if alert
        else None,
    }


def main() -> None:
    from pixelmind.data.loader import load_pair, usable
    from pixelmind.data.manifest import read_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/levircd.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("docs/data/levircd_reference_stats.json"))
    args = parser.parse_args()

    records = usable(read_manifest(args.manifest), "train")
    samples = []
    for r in records:
        t1, t2, _ = load_pair(args.root, r)
        samples.append(image_stats(t1, t2))
    ref = build_reference(samples, f"LEVIR-CD train ({len(samples)} paires)")
    args.out.write_text(json.dumps(ref, indent=1, ensure_ascii=False))
    print(f"reference ({len(samples)} pairs) -> {args.out}")


if __name__ == "__main__":
    main()
