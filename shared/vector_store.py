"""Thin wrapper around ChromaDB (local) / pgvector (prod).

For the scaffold, only the ChromaDB path is implemented. The pgvector path is
added in Lab 06 when the staging Pulumi stack provisions a real DB.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import chromadb


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """Minimal vector store interface used across labs."""

    def __init__(
        self,
        collection: str | None = None,
        persist_dir: str = "./chroma_db",
    ) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        name = collection or os.environ.get("VECTOR_DB_COLLECTION", "lab-corpus")
        # Cosine space so `score = 1 - distance` (see `search`) is true cosine
        # similarity. ChromaDB defaults to squared-L2; for the normalized vectors
        # NV-EmbedQA and the mock embedder return, that would not be cosine.
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self._collection.upsert(
            ids=list(ids),
            documents=list(texts),
            embeddings=[list(e) for e in embeddings],
            metadatas=list(metadatas) if metadatas else None,
        )

    def search(
        self, query_embedding: Sequence[float], top_k: int = 20
    ) -> list[SearchResult]:
        res = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
        )
        out: list[SearchResult] = []
        docs = res["documents"][0] if res.get("documents") else []
        dists = res["distances"][0] if res.get("distances") else []
        metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(docs)
        for text, dist, meta in zip(docs, dists, metas, strict=False):
            score = 1.0 - float(dist)  # cosine distance -> similarity
            out.append(SearchResult(text=text, score=score, metadata=meta or {}))
        return out

    def count(self) -> int:
        return self._collection.count()
