"""Comparative LEVIR-CD benchmark report, rebuilt from every result JSON.

python -m pixelmind.experiments.report
"""

import argparse
import json
from pathlib import Path

from pixelmind.evaluation.protocol import error_analysis

RESULTS_DIR = Path("artifacts/experiments/levircd")
REPORT_DIR = Path("docs/experiments")
REPORT_NAME = "levircd_benchmark.md"
FIGURE_NAME = "levircd_pr.png"
BASELINE_ORDER = ["absdiff", "normdiff", "normdiff_smooth", "gbm"]


def fmt(x: float | None, pct: bool = True) -> str:
    if x is None:
        return "–"
    return f"{x:.1%}" if pct else f"{x:.3f}"


def load_results(results_dir: Path) -> dict[str, dict]:
    found = {f.stem: json.loads(f.read_text()) for f in results_dir.glob("*.json")}
    for r in found.values():  # derived from per-scene results: keep it in sync with the code
        r["error_analysis"] = error_analysis(r["test_scenes"])
    order = [n for n in BASELINE_ORDER if n in found]
    order += sorted(
        (n for n in found if n not in BASELINE_ORDER), key=lambda n: found[n]["test"]["f1"] or 0
    )
    return {n: found[n] for n in order}


def render_report(results: dict[str, dict], figure: str) -> str:
    first = next(iter(results.values()))["meta"]
    L = [
        "# Benchmark LEVIR-CD",
        "",
        "Généré par `python -m pixelmind.experiments.report`. Protocole commun "
        "(`pixelmind.evaluation.protocol`) : seuil choisi sur **val** (F1 max), métriques "
        "sur **test** (128 paires) au niveau pixel, micro-moyennées sur toutes les images. "
        "F1 = Dice au niveau pixel.",
        "",
        f"Manifeste `{first['manifest']}` (sha256 `{first['manifest_sha256'][:12]}…`).",
        "",
        "## Résultats (test)",
        "",
        "| Modèle | Seuil | F1 val | Précision | Rappel | **F1** | IoU | PR-AUC (AP) | FPR "
        "| Éval. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        t = r["test"]
        L.append(
            f"| `{name}` | {r['threshold']} | {fmt(r['val']['f1'])} | {fmt(t['precision'])} "
            f"| {fmt(t['recall'])} | **{fmt(t['f1'])}** | {fmt(t['iou'])} | {fmt(t['ap'])} "
            f"| {fmt(t['fpr'], pct=False)} | {r['seconds']:.0f} s |"
        )
    L += ["", f"![Courbes précision-rappel (test)]({figure})", "", "## Analyse d'erreurs (test)"]
    L += ["", "F1 par quantité de changement dans la scène (pixels changés / total) :", ""]
    buckets = list(next(iter(results.values()))["error_analysis"]["by_change_bucket"])
    L.append("| Modèle | " + " | ".join(buckets) + " | F1 médian par scène |")
    L.append("|---|" + "---|" * (len(buckets) + 1))
    for name, r in results.items():
        ea = r["error_analysis"]
        cells = []
        for b in buckets:
            g = ea["by_change_bucket"][b]
            # no-change scenes have no F1: report the share of pixels wrongly flagged
            cells.append(f"FPR {fmt(g['fpr'])}" if b == "no change" else fmt(g["f1"]))
        L.append(f"| `{name}` | " + " | ".join(cells) + f" | {fmt(ea['scene_f1_median'])} |")
    counts = next(iter(results.values()))["error_analysis"]["by_change_bucket"]
    L += ["", "Scènes par groupe : " + ", ".join(f"{b} = {g['scenes']}" for b, g in counts.items())]

    best = max(results, key=lambda n: results[n]["test"]["f1"] or 0)
    L += ["", f"### 10 pires scènes – `{best}`", ""]
    L += ["| Scène | Changement | F1 | Précision | Rappel |", "|---|---|---|---|---|"]
    for s in results[best]["error_analysis"]["worst_scenes"]:
        L.append(
            f"| {s['pair_id']} | {fmt(s['changed_ratio'])} | {fmt(s['f1'])} "
            f"| {fmt(s['precision'])} | {fmt(s['recall'])} |"
        )

    if "gbm" in results:
        g = results["gbm"]["extra"]
        L += [
            "",
            "## Baseline 1 – features + gradient boosting",
            "",
            f"{g['train_samples']:,} pixels d'entraînement échantillonnés "
            f"(équilibrés, part positive {g['train_positive_share']:.0%}), "
            f"{g['n_iter']} itérations, {g['train_seconds']:.0f} s d'entraînement.",
            "",
            "Importance par permutation (baisse de F1 sur un échantillon val) :",
            "",
            "| Feature | Δ F1 |",
            "|---|---|",
        ]
        L += [f"| `{f}` | {v:.3f} |" for f, v in list(g["feature_importance_f1_drop"].items())[:10]]

    L += ["", "## Reproductibilité", "", "| Modèle | Commit | Seed | Machine | Détails |"]
    L.append("|---|---|---|---|---|")
    for name, r in results.items():
        m, e = r["meta"], r.get("extra") or {}
        details = ", ".join(
            f"{k}={e[k]}" for k in ("checkpoint", "best_epoch", "n_params", "device") if k in e
        )
        full = m.get("git_commit") or "?"
        commit = full[:8] + ("-dirty" if full.endswith("-dirty") else "")
        L.append(f"| `{name}` | `{commit}` | {m.get('seed', '–')} | {m['machine']} | {details} |")
    return "\n".join(L) + "\n"


def plot_pr(results: dict[str, dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        c, t = r["test_pr_curve"], r["test"]
        (line,) = ax.plot(c["recall"], c["precision"], label=f"{name} (AP {t['ap']:.2f})")
        ax.scatter([t["recall"]], [t["precision"]], s=20, color=line.get_color())
    ax.set(xlabel="Rappel", ylabel="Précision", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("LEVIR-CD test – précision/rappel\n(points : seuil choisi sur val)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def rebuild(results_dir: Path = RESULTS_DIR, report_dir: Path = REPORT_DIR) -> Path:
    results = load_results(results_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_pr(results, report_dir / FIGURE_NAME)
    path = report_dir / REPORT_NAME
    path.write_text(render_report(results, FIGURE_NAME))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    print(f"report -> {rebuild(args.results, args.report)}")


if __name__ == "__main__":
    main()
