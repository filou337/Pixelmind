"""Sprint 2: run and compare baselines on LEVIR-CD with the shared protocol.

python -m pixelmind.experiments.baselines [--models absdiff normdiff normdiff_smooth gbm]

Outputs:
  artifacts/experiments/levircd/{model}.json   full results (+ gbm.joblib)
  docs/experiments/levircd_benchmark.md       comparison report (no imagery)
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from pixelmind.data.loader import usable
from pixelmind.data.manifest import read_manifest
from pixelmind.evaluation.protocol import evaluate
from pixelmind.experiments import report
from pixelmind.models.baselines import image_diff
from pixelmind.provenance import git_commit

ALL_MODELS = [*image_diff.SCORERS, "gbm"]


def run_gbm(records, root, out_dir, per_class, seed) -> tuple[dict, dict]:
    import joblib
    from sklearn.inspection import permutation_importance

    from pixelmind.models.baselines import gbm

    t0 = time.perf_counter()
    X, y = gbm.sample_training_set(root, usable(records, "train"), per_class, seed)
    model = gbm.train(X, y, seed)
    train_s = time.perf_counter() - t0
    joblib.dump(model, out_dir / "gbm.joblib")

    # permutation importance on a balanced validation sample (cheap, model-agnostic)
    Xv, yv = gbm.sample_training_set(root, usable(records, "val"), 500, seed + 1)
    imp = permutation_importance(model, Xv, yv, scoring="f1", n_repeats=3, random_state=seed)
    importance = dict(
        sorted(zip(gbm.FEATURES, imp.importances_mean.tolist(), strict=True), key=lambda kv: -kv[1])
    )
    extra = {
        "train_samples": int(len(y)),
        "train_positive_share": float(y.mean()),
        "n_iter": int(model.n_iter_),
        "train_seconds": round(train_s, 1),
        "feature_importance_f1_drop": importance,
        "params": {"per_class": per_class, **model.get_params()},
    }
    return gbm.make_scorer(model), extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/levircd.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=report.RESULTS_DIR)
    parser.add_argument("--report", type=Path, default=report.REPORT_DIR)
    parser.add_argument("--per-class", type=int, default=1000, help="gbm: pixels/class/image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    records = read_manifest(args.manifest)
    meta = {
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "git_commit": git_commit(),
        "seed": args.seed,
        "machine": f"{platform.processor() or platform.machine()}, CPU only",
    }

    for name in args.models:
        print(f"== {name}", flush=True)
        t0 = time.perf_counter()
        extra = {}
        if name == "gbm":
            scorer, extra = run_gbm(records, args.root, args.out, args.per_class, args.seed)
        else:
            scorer = image_diff.SCORERS[name]
        res = evaluate(scorer, records, args.root)
        res.update(model=name, seconds=time.perf_counter() - t0, extra=extra, meta=meta)
        (args.out / f"{name}.json").write_text(json.dumps(res, indent=1))
        t = res["test"]
        print(f"   threshold={res['threshold']} test F1={t['f1']:.3f} IoU={t['iou']:.3f} "
              f"AP={t['ap']:.3f} ({res['seconds']:.0f}s)", flush=True)  # fmt: skip

    print(f"report -> {report.rebuild(args.out, args.report)}")


if __name__ == "__main__":
    main()
