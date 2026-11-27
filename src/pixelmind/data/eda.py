"""EDA + quality report from a LEVIR-CD manifest.

Writes a markdown report and a histogram (statistics only, no imagery) to --out,
and a sample grid of T1/T2/mask (contains dataset imagery, licence-restricted)
to --samples-out, which should stay out of Git.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from pixelmind.data.manifest import SPLITS, read_manifest


def summarize(records: list[dict]) -> dict:
    by_split: dict[str, dict] = {}
    for split in SPLITS:
        rs = [r for r in records if r["split"] == split]
        if not rs:
            continue
        ratios = np.array(
            [r["changed_pixel_ratio"] for r in rs if r["changed_pixel_ratio"] is not None]
        )
        pixels = np.array(
            [r["width"] * r["height"] for r in rs if r["changed_pixel_ratio"] is not None]
        )
        by_split[split] = {
            "pairs": len(rs),
            "status": Counter(r["quality_status"] for r in rs),
            "sizes": Counter(f"{r['width']}x{r['height']}" for r in rs),
            "no_change_pairs": int((ratios == 0).sum()),
            "ratio_mean": float(ratios.mean()) if ratios.size else float("nan"),
            "ratio_median": float(np.median(ratios)) if ratios.size else float("nan"),
            "ratio_p95": float(np.percentile(ratios, 95)) if ratios.size else float("nan"),
            "ratio_max": float(ratios.max()) if ratios.size else float("nan"),
            # pixel-weighted share of changed pixels = class imbalance seen by the model
            "positive_pixel_share": float((ratios * pixels).sum() / pixels.sum())
            if ratios.size
            else float("nan"),
        }
    issues = Counter((i["status"], i["code"]) for r in records for i in r["quality_issues"])
    return {"by_split": by_split, "issues": issues, "total": len(records)}


def render_markdown(summary: dict, histogram_name: str | None) -> str:
    lines = [
        "# LEVIR-CD – rapport qualité et EDA",
        "",
        "Généré par `python -m pixelmind.data.eda` à partir du manifeste "
        "(`data/manifests/levircd.jsonl`).",
        "",
        f"Total : **{summary['total']} paires**.",
        "",
        "## Par split",
        "",
        "| Split | Paires | PASS | WARN | REVIEW | FAIL | Tailles | Paires sans changement "
        "| Ratio moyen | Médiane | P95 | Max | Part pixels positifs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for split, s in summary["by_split"].items():
        st = s["status"]
        sizes = ", ".join(f"{k} ({v})" for k, v in s["sizes"].items())
        lines.append(
            f"| {split} | {s['pairs']} "
            f"| {st['PASS']} | {st['WARN']} | {st['REVIEW']} | {st['FAIL']} "
            f"| {sizes} | {s['no_change_pairs']} ({s['no_change_pairs'] / s['pairs']:.0%}) "
            f"| {s['ratio_mean']:.2%} | {s['ratio_median']:.2%} | {s['ratio_p95']:.2%} "
            f"| {s['ratio_max']:.2%} | {s['positive_pixel_share']:.2%} |"
        )
    lines += ["", "## Anomalies qualité", ""]
    if summary["issues"]:
        lines += ["| Statut | Code | Occurrences |", "|---|---|---|"]
        lines += [f"| {st} | `{code}` | {n} |" for (st, code), n in summary["issues"].most_common()]
    else:
        lines.append("Aucune anomalie détectée : toutes les paires sont PASS.")
    if histogram_name:
        lines += [
            "",
            "## Distribution du ratio de pixels changés",
            "",
            f"![histogramme]({histogram_name})",
        ]
    return "\n".join(lines) + "\n"


def plot_histogram(records: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, max(r["changed_pixel_ratio"] or 0 for r in records) or 1, 40)
    for split in SPLITS:
        ratios = [
            r["changed_pixel_ratio"]
            for r in records
            if r["split"] == split and r["changed_pixel_ratio"] is not None
        ]
        if ratios:
            ax.hist(ratios, bins=bins, alpha=0.6, label=f"{split} (n={len(ratios)})", density=True)
    ax.set_xlabel("Ratio de pixels changés par paire")
    ax.set_ylabel("Densité")
    ax.set_title("LEVIR-CD – ratio de changement par split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_samples(records: list[dict], root: Path, path: Path, n: int = 6, seed: int = 0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = np.random.default_rng(seed)
    pool = [
        r for r in records if r["quality_status"] != "FAIL" and (r["changed_pixel_ratio"] or 0) > 0
    ]
    picks = [pool[i] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)), squeeze=False)
    for row, r in zip(axes, picks, strict=True):
        for ax, role in zip(row, ("t1", "t2", "mask"), strict=True):
            with Image.open(root / r["paths"][role]) as img:
                ax.imshow(np.asarray(img), cmap="gray" if role == "mask" else None)
            ax.set_title(f"{r['pair_id']} – {role}" if role == "t1" else role, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/levircd.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("docs/data"))
    parser.add_argument(
        "--samples-out", type=Path, default=Path("artifacts/eda/levircd_samples.png")
    )
    args = parser.parse_args()

    records = read_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    hist = "levircd_change_ratio.png"
    plot_histogram(records, args.out / hist)
    report = args.out / "levircd_quality_report.md"
    report.write_text(render_markdown(summarize(records), hist))
    plot_samples(records, args.root, args.samples_out)
    print(f"report -> {report}\nsamples -> {args.samples_out}")


if __name__ == "__main__":
    main()
