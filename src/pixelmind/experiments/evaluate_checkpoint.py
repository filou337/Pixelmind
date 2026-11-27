"""Evaluate a trained checkpoint with the shared protocol and add it to the benchmark.

python -m pixelmind.experiments.evaluate_checkpoint artifacts/models/siamese_unet_diff_best.pt
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

from pixelmind.data.manifest import read_manifest
from pixelmind.evaluation.protocol import evaluate
from pixelmind.experiments import report
from pixelmind.models.dl.inference import load_checkpoint, make_scorer, pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--name", help="model name in the report (default: from checkpoint)")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/levircd.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--out", type=Path, default=report.RESULTS_DIR)
    parser.add_argument("--report", type=Path, default=report.REPORT_DIR)
    args = parser.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    cfg = ckpt["config"]
    name = args.name or args.checkpoint.stem.removesuffix("_best")

    t0 = time.perf_counter()
    res = evaluate(
        make_scorer(model, device, tile=args.tile), read_manifest(args.manifest), args.root
    )
    res.update(
        model=name,
        seconds=time.perf_counter() - t0,
        meta={
            "manifest": str(args.manifest),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "git_commit": ckpt.get("git_commit"),
            "seed": cfg["seed"],
            "machine": f"{platform.processor() or platform.machine()}, {device}",
        },
        extra={
            "checkpoint": args.checkpoint.name,
            "best_epoch": ckpt["epoch"],
            "n_params": sum(p.numel() for p in model.parameters()),
            "config": cfg,
        },
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(json.dumps(res, indent=1))
    t = res["test"]
    print(
        f"{name}: threshold={res['threshold']} "
        f"test F1={t['f1']:.3f} IoU={t['iou']:.3f} AP={t['ap']:.3f}"
    )
    print(f"report -> {report.rebuild(args.out, args.report)}")


if __name__ == "__main__":
    main()
