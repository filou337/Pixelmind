"""RAG evaluation (dossier §8.4) on configs/rag_eval_questions.yaml.

python -m pixelmind.rag.evaluate                 # retrieval only: free, deterministic
python -m pixelmind.rag.evaluate --generate      # + answers with the configured LLM (costs tokens)

Retrieval: Recall@k and MRR for dense / sparse / hybrid, and the abstention threshold on
the best dense score. Generation: abstention accuracy, answers with a citation, citations
pointing to an expected document.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from pixelmind.rag.retriever import MODES, Retriever

KS = (1, 3, 5)


def is_relevant(chunk: dict, item: dict) -> bool:
    if chunk["doc_id"] not in item["docs"]:
        return False
    text = (chunk["section"] + " " + chunk["text"]).lower()
    return any(kw.lower() in text for kw in item.get("keywords", [""]))


def retrieval_metrics(retriever: Retriever, items: list[dict], mode: str, k: int = 5) -> dict:
    ranks = []
    for item in items:
        hits = retriever.search(item["q"], k=k, mode=mode).hits
        ranks.append(next((h.rank for h in hits if is_relevant(h.chunk, item)), None))
    out = {f"recall@{n}": float(np.mean([r is not None and r <= n for r in ranks])) for n in KS}
    out["mrr@5"] = float(np.mean([1 / r if r else 0.0 for r in ranks]))
    out["misses"] = [it["q"] for it, r in zip(items, ranks, strict=True) if r is None]
    return out


def abstention_threshold(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """Threshold on the best dense score maximizing balanced accuracy (midpoint of the gap)."""
    cands = sorted(set(pos + neg))
    best_t, best_acc = 0.0, -1.0
    for a, b in zip(cands, cands[1:], strict=False):
        t = (a + b) / 2
        acc = (np.mean([p >= t for p in pos]) + np.mean([n < t for n in neg])) / 2
        if acc > best_acc:
            best_t, best_acc = t, acc
    return round(best_t, 3), float(best_acc)


def main() -> None:
    from qdrant_client import QdrantClient

    from pixelmind.rag.index import RAG_DIR, FastEmbedder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("configs/rag_eval_questions.yaml"))
    parser.add_argument("--generate", action="store_true", help="also evaluate LLM answers")
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/rag_eval.md"))
    args = parser.parse_args()

    items = yaml.safe_load(args.questions.read_text())["questions"]
    answerable = [i for i in items if i.get("answerable", True)]
    unanswerable = [i for i in items if not i.get("answerable", True)]
    retriever = Retriever(
        QdrantClient(path=str(RAG_DIR / "qdrant")), FastEmbedder(RAG_DIR / "models")
    )
    manifest = json.loads((RAG_DIR / "index_manifest.json").read_text())

    per_mode = {m: retrieval_metrics(retriever, answerable, m) for m in MODES}
    pos = [retriever.search(i["q"], k=1, mode="dense").best_dense_score for i in answerable]
    neg = [retriever.search(i["q"], k=1, mode="dense").best_dense_score for i in unanswerable]
    threshold, bal_acc = abstention_threshold(pos, neg)

    L = [
        "# Évaluation du RAG documentaire",
        "",
        "Généré par `python -m pixelmind.rag.evaluate`. Jeu : `configs/rag_eval_questions.yaml` "
        f"({len(answerable)} questions avec réponse dans le corpus, {len(unanswerable)} hors "
        f"corpus). Index : {manifest['n_chunks']} passages, corpus sha256 "
        f"`{manifest['corpus_sha256'][:12]}…`, embeddings `{manifest['embedder']}`.",
        "",
        "Un passage est jugé pertinent s'il vient d'un document attendu et contient un mot-clé "
        "attendu.",
        "",
        "## Recherche (k = 5)",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 | MRR@5 |",
        "|---|---|---|---|---|",
    ]
    for m, r in per_mode.items():
        L.append(f"| {m} | {r['recall@1']:.0%} | {r['recall@3']:.0%} | {r['recall@5']:.0%} "
                 f"| {r['mrr@5']:.3f} |")  # fmt: skip
    for m, r in per_mode.items():
        if r["misses"]:
            L += ["", f"Questions sans passage pertinent dans le top 5 ({m}) :"]
            L += [f"- {q}" for q in r["misses"]]
    L += [
        "",
        "## Abstention (seuil sur le meilleur score cosinus dense)",
        "",
        f"- Questions avec réponse : score min {min(pos):.3f}, médiane {np.median(pos):.3f}",
        f"- Questions hors corpus : score max {max(neg):.3f}, médiane {np.median(neg):.3f}",
        f"- Seuil retenu : **{threshold}** (exactitude équilibrée {bal_acc:.0%} sur ce jeu). "
        "Avec seulement 31 questions, ce seuil est choisi et mesuré sur les mêmes données : "
        "c'est une estimation optimiste, à confirmer sur des questions nouvelles.",
    ]

    if args.generate:
        from pixelmind.rag.service import QALog, QAService
        from pixelmind.settings import Settings

        settings = Settings.from_env()
        from pixelmind.rag.generator import ClaudeGenerator

        gen = ClaudeGenerator(settings.llm_model, settings.llm_effort, settings.llm_fallbacks)
        qa = QAService(retriever, gen, QALog(RAG_DIR / "qa_eval_log.db"), manifest,
                       min_score=threshold)  # fmt: skip
        rows, tokens = [], [0, 0]
        for it in items:
            r = qa.ask(it["q"])
            ok_abst = r["abstained"] != it.get("answerable", True)
            cited = bool(r["citations"])
            valid = all(c.get("doc_id") in it.get("docs", []) for c in r["citations"])
            rows.append((it, r, ok_abst, cited, valid))
            tokens[0] += r["generation"]["usage"].get("input_tokens", 0)
            tokens[1] += r["generation"]["usage"].get("output_tokens", 0)
        ans = [x for x in rows if x[0].get("answerable", True)]
        L += [
            "",
            f"## Génération ({gen.model}, effort {gen.effort}, prompt `rag-v1`)",
            "",
            f"- Décision répondre/s'abstenir correcte : {np.mean([x[2] for x in rows]):.0%}",
            f"- Réponses (questions avec réponse) citant au moins une source : "
            f"{np.mean([x[3] for x in ans]):.0%}",
            f"- Réponses dont toutes les citations viennent d'un document attendu : "
            f"{np.mean([x[4] for x in ans if x[3]] or [0]):.0%}",
            f"- Tokens : {tokens[0]:,} en entrée, {tokens[1]:,} en sortie.",
            "",
            "La fidélité des réponses aux passages (groundedness) reste à vérifier "
            "par une relecture humaine.",
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
