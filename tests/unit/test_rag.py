import hashlib
import math
import re
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from pixelmind.rag import corpus
from pixelmind.rag.generator import (
    ABSTAIN,
    ClaudeGenerator,
    ExtractiveGenerator,
    GenerationError,
    Source,
)
from pixelmind.rag.index import SparseVec, build_index
from pixelmind.rag.retriever import Retriever
from pixelmind.rag.service import QALog, QAService, analysis_source


class FakeEmbedder:
    """Deterministic bag-of-words embeddings: no model download in tests."""

    name, dim = "fake", 64

    @staticmethod
    def _tokens(t):
        return re.findall(r"\w{3,}", t.lower())

    def dense(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in self._tokens(t):
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1
            out.append([x / n for x in v])
        return out

    def sparse(self, texts):
        out = []
        for t in texts:
            counts = {}
            for tok in self._tokens(t):
                i = int(hashlib.md5(tok.encode()).hexdigest(), 16) % 10_000
                counts[i] = counts.get(i, 0) + 1.0
            out.append(SparseVec(list(counts), list(counts.values())))
        return out


# --- corpus / chunking ------------------------------------------------------


def test_markdown_sections_ignore_code_comments_and_split_long_text():
    text = "# Titre\nintro\n## Partie A\n```bash\n# pas un titre\nls\n```\n\n" + "\n\n".join(
        f"Paragraphe {i} " + "mot " * 60 for i in range(6)
    )
    sections = corpus._sections_markdown(text)
    assert [s for s, _, _ in sections] == ["Titre", "Titre > Partie A"]
    doc = corpus.Document("d", "T", "readme", "lic", "README.md", "x")
    chunks = [c for c in corpus.chunk_document(doc, _write(text)) if c.section.endswith("A")]
    assert len(chunks) > 1 and all(len(c.text) <= corpus.MAX_CHARS for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def _write(text, suffix=".md"):
    import tempfile
    from pathlib import Path

    f = Path(tempfile.mkdtemp()) / f"doc{suffix}"
    f.write_text(text)
    return f


def test_pdf_headings_are_sequential_and_table_rows_rejected():
    pages = [
        "PIXELMIND | Dossier d'architecture technique Page 1\n1. Vision\ntexte un\n1.1 Problème\nb",
        "2. Architecture fonctionnelle\n1. Acquisition Import local, catalogue,\nligne\n"
        "3 Vision & ML Baselines, modèles,\n2.1 Parcours\nc",
    ]
    sections = corpus._sections_pdf(pages)
    assert [(s, p) for s, p, _ in sections] == [
        ("1 Vision", 1), ("1.1 Problème", 1), ("2 Architecture fonctionnelle", 2),
        ("2.1 Parcours", 2),
    ]  # fmt: skip
    assert "PIXELMIND |" not in sections[0][2]


def test_manifest_rejects_missing_files(tmp_path):
    (tmp_path / "c.yaml").write_text(
        "documents:\n  - {path: nope/*.md, doc_type: adr, licence: x}\n"
    )
    with pytest.raises(FileNotFoundError):
        corpus.load_manifest(tmp_path / "c.yaml", tmp_path)


# --- index + retrieval ------------------------------------------------------


def make_chunk(i, doc_id, doc_type, section, text):
    return corpus.Chunk(f"c{i}", doc_id, f"Doc {doc_id}", doc_type, "lic", f"{doc_id}.md",
                        section, None, text)  # fmt: skip


CHUNKS = [
    make_chunk(0, "adr3", "adr", "Décision", "Entraînement sur GPU Kaggle car pas de CUDA local"),
    make_chunk(1, "card", "dataset", "Licence", "LEVIR-CD est réservé à un usage académique"),
    make_chunk(2, "bench", "report", "Résultats", "Le gradient boosting obtient F1 43 sur test"),
]


@pytest.fixture
def retriever():
    client = QdrantClient(":memory:")
    build_index(client, CHUNKS, FakeEmbedder())
    return Retriever(client, FakeEmbedder())


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_retrieval_modes_find_the_right_chunk(retriever, mode):
    r = retriever.search("usage académique de LEVIR-CD licence", k=2, mode=mode)
    assert r.hits[0].chunk["doc_id"] == "card" and r.hits[0].rank == 1
    assert r.best_dense_score > 0.3


def test_retrieval_filter_by_doc_type(retriever):
    r = retriever.search("licence académique", k=3, mode="hybrid", doc_types=["adr"])
    assert {h.chunk["doc_id"] for h in r.hits} == {"adr3"}


# --- Claude generator (fake client: no network, no cost) --------------------


def text_block(text, citations=()):
    return SimpleNamespace(type="text", text=text, citations=list(citations))


def cite(idx, text):
    return SimpleNamespace(type="search_result_location", search_result_index=idx,
                           cited_text=text, source="s", title="t")  # fmt: skip


class FakeClaude:
    def __init__(self, content=None, stop_reason="end_turn", error=None):
        self.calls = []
        self.content, self.stop_reason, self.error = content or [], stop_reason, error
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        usage = SimpleNamespace(input_tokens=120, output_tokens=30)
        return SimpleNamespace(content=self.content, stop_reason=self.stop_reason,
                               model="claude-opus-5-5", usage=usage)  # fmt: skip


SOURCES = [Source("a.md#x", "A", ["bloc A"], {"chunk_id": "c0", "doc_id": "adr3"}),
           Source("b.md#y", "B", ["bloc B"], {"chunk_id": "c1", "doc_id": "card"})]  # fmt: skip


def test_claude_request_shape_and_citation_mapping():
    fake = FakeClaude([
        text_block("Le projet entraîne sur Kaggle", [cite(0, "bloc A")]),
        text_block(", et LEVIR-CD est académique.", [cite(1, "bloc B"), cite(1, "bloc B")]),
    ])  # fmt: skip
    a = ClaudeGenerator(client=fake).answer("Pourquoi Kaggle ?", SOURCES)
    assert a.text == "Le projet entraîne sur Kaggle [1], et LEVIR-CD est académique. [2]"
    assert [(c["n"], c["chunk_id"], c["cited_text"]) for c in a.citations] == [
        (1, "c0", "bloc A"), (2, "c1", "bloc B")]  # fmt: skip
    assert not a.abstained and a.usage == {"input_tokens": 120, "output_tokens": 30}

    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"] == {"effort": "medium"}
    blocks = call["messages"][0]["content"]
    assert blocks[0] == {"type": "search_result", "source": "a.md#x", "title": "A",
                         "content": [{"type": "text", "text": "bloc A"}],
                         "citations": {"enabled": True}}  # fmt: skip
    assert blocks[-1] == {"type": "text", "text": "Pourquoi Kaggle ?"}


def test_claude_without_fallbacks_uses_plain_endpoint():
    fake = FakeClaude([text_block("ok")])
    ClaudeGenerator(client=fake, fallbacks=False).answer("q", SOURCES)
    assert "fallbacks" not in fake.calls[0] and "betas" not in fake.calls[0]


def test_claude_abstention_and_refusal():
    a = ClaudeGenerator(client=FakeClaude([text_block(f"{ABSTAIN} : rien")])).answer("q", SOURCES)
    assert a.abstained and not a.citations
    r = ClaudeGenerator(client=FakeClaude([], stop_reason="refusal")).answer("q", SOURCES)
    assert r.refused and r.abstained and r.text == ""


def test_claude_errors_are_mapped_to_http_statuses():
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    errors = {
        503: anthropic.APIConnectionError(request=req),
        429: anthropic.RateLimitError(
            "slow down", response=httpx2.Response(429, request=req), body=None
        ),  # fmt: skip
    }
    for status, err in errors.items():
        with pytest.raises(GenerationError) as exc:
            ClaudeGenerator(client=FakeClaude(error=err)).answer("q", SOURCES)
        assert exc.value.status_code == status


# --- QA service + API -------------------------------------------------------


class SpyGenerator(ExtractiveGenerator):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def answer(self, question, sources):
        self.calls += 1
        return super().answer(question, sources)


ANALYSIS = {
    "analysis_id": "pm_20261127_0000abcd", "created_at": "2026-11-27T00:00:00+00:00",
    "status": "completed_with_warnings", "quality_status": "PASS",
    "model": {"name": "gbm", "kind": "ml", "version": "sha256:x", "threshold": 200},
    "model_test_metrics": {"f1": 0.431},
    "parameters": {"min_region_px": 16, "gsd_m": None},
    "prediction": {"changed_pixel_ratio": 0.0973, "changed_area_m2_estimate": None,
                   "n_regions": 2, "regions": [{"bbox": [1, 2, 3, 4], "area_px": 50,
                                                "mean_score": 0.9}]},
    "metrics": None, "warnings": ["gsd_m non fourni"], "human_review_required": False,
    "limitations": ["à vérifier par un humain"],
}  # fmt: skip


def test_analysis_source_states_only_measured_facts():
    facts = analysis_source(ANALYSIS).blocks
    assert "Part de pixels changés : 9.73% de l'image." in facts
    assert any("non calculée" in f for f in facts)  # no area without gsd_m
    assert not any("m² (" in f for f in facts)


def test_service_abstains_without_llm_call_when_nothing_relevant(retriever, tmp_path):
    gen = SpyGenerator()
    qa = QAService(retriever, gen, QALog(tmp_path / "log.db"), {"corpus_sha256": "abc"},
                   min_score=0.99)  # fmt: skip
    r = qa.ask("recette de la tarte tatin")
    assert r["abstained"] and gen.calls == 0 and not r["generation"]["llm_called"]
    # with an analysis, its facts are still available to answer
    r = qa.ask("combien de régions ?", analysis=ANALYSIS)
    assert gen.calls == 1 and r["citations"][0]["source"] == "analysis:pm_20261127_0000abcd"


@pytest.fixture
def api(retriever, tmp_path):
    from pixelmind.api.main import create_app
    from pixelmind.models.registry import ModelRegistry
    from pixelmind.services.analysis import AnalysisService
    from pixelmind.services.store import AnalysisStore
    from pixelmind.settings import Settings

    settings = Settings(artifacts_dir=tmp_path)
    store = AnalysisStore(settings.analyses_dir / "analyses.db")
    store.save({**ANALYSIS, "quality_issues": []})
    svc = AnalysisService(settings, ModelRegistry([]), store)
    qa = QAService(retriever, ExtractiveGenerator(), QALog(tmp_path / "qa.db"),
                   {"corpus_sha256": "abc", "n_chunks": 3}, min_score=0.1)  # fmt: skip
    with TestClient(create_app(svc, qa)) as c:
        yield c


def test_api_docs_question(api):
    r = api.post("/api/v1/questions", json={"question": "Quelle est la licence de LEVIR-CD ?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["citations"][0]["doc_id"] == "card" and "[1]" in body["answer"]
    assert body["retrieval"]["mode"] == "hybrid" and body["index"]["corpus_sha256"] == "abc"


def test_api_analysis_question_and_validation(api):
    aid = ANALYSIS["analysis_id"]
    r = api.post(f"/api/v1/analyses/{aid}/questions", json={"question": "Combien de régions ?"})
    assert r.status_code == 200 and r.json()["analysis_id"] == aid
    assert api.post("/api/v1/analyses/pm_x/questions", json={"question": "abc"}).status_code == 404
    assert api.post("/api/v1/questions", json={"question": "a"}).status_code == 422
