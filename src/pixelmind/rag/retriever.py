"""Retrieval: dense, sparse (BM25) or hybrid (Reciprocal Rank Fusion), with filters."""

from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from pixelmind.rag.index import COLLECTION, Embedder

MODES = ("dense", "sparse", "hybrid")


@dataclass
class Hit:
    chunk: dict  # Chunk payload
    score: float  # mode-dependent: cosine, BM25 or RRF score
    rank: int


@dataclass
class Retrieval:
    hits: list[Hit]
    best_dense_score: float  # cosine of the best chunk: relevance signal for abstention
    mode: str


class Retriever:
    def __init__(self, client: QdrantClient, embedder: Embedder) -> None:
        self.client, self.embedder = client, embedder

    def search(
        self, question: str, k: int = 5, mode: str = "hybrid",
        doc_types: list[str] | None = None,
    ) -> Retrieval:  # fmt: skip
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        flt = None
        if doc_types:
            flt = models.Filter(must=[models.FieldCondition(
                key="doc_type", match=models.MatchAny(any=doc_types))])  # fmt: skip
        dense = self.embedder.dense([question])[0]
        s = self.embedder.sparse([question])[0]
        sparse = models.SparseVector(indices=s.indices, values=s.values)

        top_dense = self.client.query_points(
            COLLECTION, query=dense, using="dense", limit=max(k, 1), query_filter=flt
        ).points
        best = top_dense[0].score if top_dense else 0.0
        if mode == "dense":
            points = top_dense[:k]
        elif mode == "sparse":
            points = self.client.query_points(
                COLLECTION, query=sparse, using="bm25", limit=k, query_filter=flt
            ).points
        else:
            points = self.client.query_points(
                COLLECTION,
                prefetch=[
                    models.Prefetch(query=dense, using="dense", limit=4 * k, filter=flt),
                    models.Prefetch(query=sparse, using="bm25", limit=4 * k, filter=flt),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
            ).points
        hits = [Hit(p.payload, float(p.score), i) for i, p in enumerate(points, start=1)]
        return Retrieval(hits, float(best), mode)
