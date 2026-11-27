"""Question answering over the documentation and, optionally, one analysis.

Flow: retrieve -> relevance gate (abstain without calling the LLM when nothing relevant
is found) -> generate with citations -> log everything needed to audit the answer.
"""

import json
import secrets
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from pixelmind.rag.generator import PROMPT_VERSION, Answer, Source
from pixelmind.rag.retriever import Hit, Retriever

EXTRACTIVE_MIN_SCORE = 0.45

NOT_FOUND = (
    "Je ne trouve pas cette information dans la documentation indexée de PIXELMIND. "
    "Reformulez la question ou consultez directement les documents."
)


def chunk_source(hit: Hit) -> Source:
    c = hit.chunk
    page = f", p. {c['page']}" if c.get("page") else ""
    return Source(
        source=f"{c['source']}#{c['section']}",
        title=f"{c['title']} — {c['section']}{page}",
        blocks=[c["text"]],
        ref={k: c.get(k) for k in ("chunk_id", "doc_id", "doc_type", "section", "page")},
    )


def analysis_source(result: dict) -> Source:
    """An analysis result as citable facts, one per block (dossier §8.3: the LLM must not
    invent a location, area, date or conclusion: it can only quote these)."""
    aid, p, m = result["analysis_id"], result.get("prediction"), result["model"]
    facts = [
        f"Analyse {aid} créée le {result['created_at']}, statut {result['status']}.",
        f"Statut qualité des données : {result['quality_status']}.",
        f"Modèle : {m['name']} ({m['kind']}, version {m['version']}), seuil {m['threshold']}/255.",
    ]
    f1 = (result.get("model_test_metrics") or {}).get("f1")
    if f1 is not None:
        facts.append(f"F1 de ce modèle sur le jeu de test LEVIR-CD : {f1:.1%}.")
    if p:
        facts.append(f"Part de pixels changés : {p['changed_pixel_ratio']:.2%} de l'image.")
        if p["changed_area_m2_estimate"] is not None:
            gsd = result["parameters"]["gsd_m"]
            facts.append(f"Surface changée estimée : {p['changed_area_m2_estimate']} m² "
                         f"(résolution déclarée {gsd} m/pixel).")  # fmt: skip
        else:
            facts.append("Surface en m² non calculée : résolution au sol (gsd_m) non fournie.")
        facts.append(f"Nombre de régions changées : {p['n_regions']}.")
        for r in p["regions"][:3]:
            facts.append(f"Région de {r['area_px']} pixels, boîte [x0, y0, x1, y1] = {r['bbox']}, "
                         f"score moyen {r['mean_score']}.")  # fmt: skip
    if result.get("metrics"):
        mt = result["metrics"]
        facts.append(f"Comparaison au masque de référence : F1 {mt['f1']}, IoU {mt['iou']}, "
                     f"précision {mt['precision']}, rappel {mt['recall']}.")  # fmt: skip
    facts += [f"Alerte : {w}" for w in result.get("warnings", [])]
    facts.append(
        "Revue humaine requise : " + ("oui." if result["human_review_required"] else "non.")
    )
    facts += [f"Limite : {lim}" for lim in result.get("limitations", [])]
    return Source(f"analysis:{aid}", f"Résultat de l'analyse {aid}", facts,
                  ref={"analysis_id": aid})  # fmt: skip


class QALog:
    """Audit log of every question (dossier §10.3–10.5)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as con, con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS questions (question_id TEXT PRIMARY KEY, "
                "created_at TEXT, analysis_id TEXT, abstained INTEGER, result TEXT)"
            )

    def save(self, r: dict) -> None:
        with closing(sqlite3.connect(self.db_path)) as con, con:
            con.execute(
                "INSERT INTO questions VALUES (?, ?, ?, ?, ?)",
                (r["question_id"], r["created_at"], r["analysis_id"], int(r["abstained"]),
                 json.dumps(r, ensure_ascii=False)),
            )  # fmt: skip


class QAService:
    def __init__(
        self, retriever: Retriever, generator, log: QALog, index_manifest: dict,
        min_score: float = 0.35, k: int = 5, mode: str = "hybrid",
    ) -> None:  # fmt: skip
        self.retriever, self.generator, self.log = retriever, generator, log
        self.index_manifest, self.min_score, self.k, self.mode = index_manifest, min_score, k, mode

    def ask(
        self, question: str, analysis: dict | None = None, mode: str | None = None,
        k: int | None = None, doc_types: list[str] | None = None,
    ) -> dict:  # fmt: skip
        t0 = time.perf_counter()
        mode, k = mode or self.mode, k or self.k
        r = self.retriever.search(question, k=k, mode=mode, doc_types=doc_types)
        gate = r.best_dense_score >= self.min_score

        sources = [analysis_source(analysis)] if analysis else []
        if gate:
            sources += [chunk_source(h) for h in r.hits]
        if sources:
            answer = self.generator.answer(question, sources)
        else:  # nothing relevant: abstain without spending an LLM call
            answer = Answer(NOT_FOUND, abstained=True, citations=[],
                            provider=self.generator.provider)  # fmt: skip

        result = {
            "question_id": f"q_{datetime.now(UTC):%Y%m%d}_{secrets.token_hex(4)}",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "question": question,
            "analysis_id": analysis["analysis_id"] if analysis else None,
            "answer": answer.text,
            "abstained": answer.abstained,
            "refused": answer.refused,
            "citations": answer.citations,
            "retrieval": {
                "mode": mode,
                "k": k,
                "doc_types": doc_types,
                "best_dense_score": round(r.best_dense_score, 4),
                "min_score": self.min_score,
                "relevant": gate,
                "hits": [
                    {
                        "rank": h.rank,
                        "score": round(h.score, 4),
                        **{f: h.chunk.get(f) for f in ("chunk_id", "doc_id", "section", "page")},
                    }
                    for h in r.hits
                ],
            },  # fmt: skip
            "generation": {
                "provider": answer.provider,
                "model": answer.model,
                "prompt_version": PROMPT_VERSION,
                "stop_reason": answer.stop_reason,
                "usage": answer.usage,
                "llm_called": bool(sources),
            },  # fmt: skip
            "index": {
                k: self.index_manifest.get(k)
                for k in ("corpus_sha256", "created_at", "embedder", "n_chunks")
            },  # fmt: skip
            "latency_s": round(time.perf_counter() - t0, 3),
        }
        self.log.save(result)
        return result


def build_qa_service(settings) -> QAService:
    """Production wiring: embedded Qdrant index + local embeddings + configured generator."""
    from qdrant_client import QdrantClient

    from pixelmind.rag.generator import ClaudeGenerator, ExtractiveGenerator
    from pixelmind.rag.index import FastEmbedder

    manifest_path = settings.rag_dir / "index_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("index RAG absent : lancer `python -m pixelmind.rag.index`")
    retriever = Retriever(
        QdrantClient(path=str(settings.rag_dir / "qdrant")),
        FastEmbedder(settings.rag_dir / "models"),
    )
    min_score = settings.rag_min_score
    if settings.llm_provider == "extractive":
        generator = ExtractiveGenerator()
        # no LLM to abstain on borderline cases: use the stricter threshold of rag_eval.md
        min_score = max(min_score, EXTRACTIVE_MIN_SCORE)
    else:
        generator = ClaudeGenerator(settings.llm_model, settings.llm_effort, settings.llm_fallbacks)
    return QAService(
        retriever, generator, QALog(settings.rag_dir / "qa_log.db"),
        json.loads(manifest_path.read_text()), min_score, settings.rag_k, settings.rag_mode,
    )  # fmt: skip
