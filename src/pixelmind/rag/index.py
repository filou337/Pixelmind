"""Embeddings and Qdrant index (dense + BM25 sparse vectors).

python -m pixelmind.rag.index      # (re)builds artifacts/rag/qdrant from configs/rag_corpus.yaml

Qdrant runs embedded (local path): no server for the MVP. The index manifest records the
corpus, chunking and embedding versions so every answer can be traced (dossier §10.3).
"""

import argparse
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient, models

from pixelmind.rag.corpus import MAX_CHARS, Chunk, build_corpus

COLLECTION = "pixelmind_docs"
DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SPARSE_MODEL = "Qdrant/bm25"
RAG_DIR = Path("artifacts/rag")


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


class Embedder(Protocol):
    name: str
    dim: int

    def dense(self, texts: list[str]) -> list[list[float]]: ...
    def sparse(self, texts: list[str]) -> list[SparseVec]: ...


class FastEmbedder:
    """Local CPU embeddings (ONNX): multilingual MiniLM (dense) + BM25 with French stemming."""

    dim = 384

    def __init__(self, cache_dir: Path = RAG_DIR / "models") -> None:
        from fastembed import SparseTextEmbedding, TextEmbedding

        self.name = f"{DENSE_MODEL} + {SPARSE_MODEL}(french)"
        self._dense = TextEmbedding(DENSE_MODEL, cache_dir=str(cache_dir))
        self._sparse = SparseTextEmbedding(
            SPARSE_MODEL, cache_dir=str(cache_dir), language="french"
        )

    # small batches: fastembed's default (256) needs >1 GB of RAM on long chunks
    BATCH = 8

    def dense(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._dense.embed(texts, batch_size=self.BATCH)]

    def sparse(self, texts: list[str]) -> list[SparseVec]:
        return [SparseVec(e.indices.tolist(), e.values.tolist())
                for e in self._sparse.embed(texts, batch_size=self.BATCH)]  # fmt: skip


def embed_text(c: Chunk) -> str:
    """What gets embedded: the text with its title and section for context."""
    return f"{c.title} — {c.section}\n{c.text}"


def point_id(chunk_id: str) -> str:
    return str(uuid.UUID(hex=hashlib.sha256(chunk_id.encode()).hexdigest()[:32]))


def build_index(client: QdrantClient, chunks: list[Chunk], embedder: Embedder) -> None:
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=embedder.dim,
                                                     distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )  # fmt: skip
    texts = [embed_text(c) for c in chunks]
    dense, sparse = embedder.dense(texts), embedder.sparse(texts)
    client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector={
                    "dense": d,
                    "bm25": models.SparseVector(indices=s.indices, values=s.values),
                },
                payload=c.to_payload(),
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ],
    )


def corpus_sha(docs) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda d: d.path):
        h.update(f"{d.path}:{d.sha256}\n".encode())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("configs/rag_corpus.yaml"))
    parser.add_argument("--out", type=Path, default=RAG_DIR)
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    args = parser.parse_args()

    docs, chunks = build_corpus(args.corpus, Path("."), args.max_chars)
    embedder = FastEmbedder(args.out / "models")
    client = QdrantClient(path=str(args.out / "qdrant"))
    build_index(client, chunks, embedder)
    client.close()
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "collection": COLLECTION,
        "corpus_sha256": corpus_sha(docs),
        "embedder": embedder.name,
        "max_chars": args.max_chars,
        "n_chunks": len(chunks),
        "documents": [
            {
                "doc_id": d.doc_id,
                "path": d.path,
                "doc_type": d.doc_type,
                "licence": d.licence,
                "sha256": d.sha256,
            }
            for d in docs
        ],  # fmt: skip
    }
    (args.out / "index_manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False)
    )
    print(f"{len(docs)} documents, {len(chunks)} chunks -> {args.out / 'qdrant'}")


if __name__ == "__main__":
    main()
