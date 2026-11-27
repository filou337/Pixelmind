"""Build a versioned JSONL manifest of LEVIR-CD pairs, with quality status.

Expected layout: {root}/levircd/{split}/{A,B,label}/{scene}.png
Paths in the manifest are relative to {root} so the manifest is portable.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from pixelmind.data.quality import (
    Issue,
    Status,
    binarize_mask,
    check_duplicates,
    check_pair,
    normalize_mask,
)

DATASET = "levircd"
LICENCE = "LEVIR-CD: academic, non-commercial use only (Chen & Shi, 2020)"
SPLITS = ("train", "val", "test")
MANIFEST_VERSION = 1


def sha256sum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as img:
        return np.asarray(img), img.mode


def build_record(root: Path, split: str, scene: str) -> dict:
    base = root / DATASET / split
    paths = {"t1": base / "A" / scene, "t2": base / "B" / scene, "mask": base / "label" / scene}
    record: dict = {
        "pair_id": f"{DATASET}_{split}_{Path(scene).stem}",
        "dataset": DATASET,
        "split": split,
        "scene_id": Path(scene).stem,
        "licence": LICENCE,
        "paths": {k: str(p.relative_to(root)) for k, p in paths.items()},
        "sha256": {},
        "width": None,
        "height": None,
        "modes": {},
        "changed_pixel_ratio": None,
    }

    issues: list[Issue] = []
    arrays: dict[str, np.ndarray] = {}
    for role, p in paths.items():
        if not p.exists():
            issues.append(Issue(Status.FAIL, f"{role}_missing", f"{p.relative_to(root)} not found"))
            continue
        record["sha256"][role] = sha256sum(p)
        try:
            arrays[role], record["modes"][role] = _load(p)
        except Exception as exc:  # corrupted / unreadable file
            issues.append(Issue(Status.FAIL, f"{role}_unreadable", str(exc)))

    if len(arrays) == 3:
        issues += check_pair(arrays["t1"], arrays["t2"], arrays["mask"])
        record["height"], record["width"] = arrays["t1"].shape[:2]
        m, _ = normalize_mask(arrays["mask"])
        if m is not None:
            record["changed_pixel_ratio"] = round(float(binarize_mask(m).mean()), 6)

    record["quality_issues"] = [i.to_dict() for i in issues]
    return record


def build_manifest(root: Path, splits: tuple[str, ...] = SPLITS) -> list[dict]:
    records = []
    for split in splits:
        scenes: set[str] = set()
        for sub in ("A", "B", "label"):
            d = root / DATASET / split / sub
            scenes |= {p.name for p in d.glob("*.png")} if d.is_dir() else set()
        records += [build_record(root, split, s) for s in sorted(scenes)]

    complete = [r for r in records if {"t1", "t2"} <= r["sha256"].keys()]
    dup_issues = check_duplicates(complete)
    for r in records:
        r["quality_issues"] += [i.to_dict() for i in dup_issues.get(r["pair_id"], [])]
        statuses = [Status[i["status"]] for i in r["quality_issues"]]
        r["quality_status"] = max(statuses, default=Status.PASS).name
        r["manifest_version"] = MANIFEST_VERSION
    return records


def write_manifest(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/manifests/levircd.jsonl"))
    args = parser.parse_args()

    records = build_manifest(args.root)
    write_manifest(records, args.out)
    counts = Counter((r["split"], r["quality_status"]) for r in records)
    print(f"{len(records)} pairs -> {args.out}")
    for (split, status), n in sorted(counts.items()):
        print(f"  {split:5s} {status:6s} {n}")


if __name__ == "__main__":
    main()
