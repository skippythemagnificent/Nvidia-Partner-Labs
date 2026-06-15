"""Labs 01-02 — RAG retrieval quality (self-contained, offline).

The labs run retrieval through a NIM (or the mock NIM server). To keep NeMoClaw's MCP
server self-contained — no running mock server, no NVIDIA_API_KEY — these tools embed
and rerank with the *same local models the mock NIM wraps* (all-MiniLM-L6-v2 and
cross-encoder/ms-marco-MiniLM-L-6-v2), loaded lazily on first use. Cosine similarity
is computed in-memory; the chunkers and hit-rate / MRR metrics are faithful ports of
``labs/01-rag-fundamentals`` and ``labs/02-rag-reranking``.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # mock NV-EmbedQA stand-in
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # mock NV-RerankQA stand-in
FIXED_CHUNK_CHARS = 140  # Lab 01 naive baseline
SENTENCE_MAX_CHARS = 320  # Lab 01/02 sentence-aware budget


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


# ── Chunking (ports of the lab chunkers) ─────────────────────────────────────
def fixed_size_chunks(
    text: str, size: int = FIXED_CHUNK_CHARS, overlap: int = 0
) -> list[str]:
    chunks: list[str] = []
    step = size - overlap
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += step
    return chunks


def sentence_chunks(text: str, max_chars: int = SENTENCE_MAX_CHARS) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            chunks.append(current.strip())
            current = s
        else:
            current = (current + " " + s).strip()
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_corpus(corpus: list[dict], chunker) -> list[dict]:
    out: list[dict] = []
    for d in corpus:
        for j, piece in enumerate(chunker(d["text"])):
            out.append({"id": f"{d['id']}#{j}", "doc_id": d["id"], "text": piece})
    return out


def _make_chunker(kind: str):
    if kind == "fixed":
        return lambda t: fixed_size_chunks(t, FIXED_CHUNK_CHARS)
    if kind == "sentence":
        return lambda t: sentence_chunks(t, SENTENCE_MAX_CHARS)
    raise ValueError(f"unknown chunker {kind!r}; use 'fixed' or 'sentence'")


# ── Index + retrieve (in-memory cosine) ──────────────────────────────────────
class _Index:
    def __init__(self, chunks: list[dict], matrix: np.ndarray):
        self.chunks = chunks
        self.matrix = matrix


def build_index(chunks: list[dict]) -> _Index:
    embs = _embedder().encode(
        [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False
    )
    return _Index(chunks, np.asarray(embs, dtype=np.float32))


def retrieve(index: _Index, query: str, k: int) -> list[dict]:
    q = _embedder().encode([query], normalize_embeddings=True, show_progress_bar=False)[
        0
    ]
    scores = index.matrix @ np.asarray(q, dtype=np.float32)  # cosine (normalized)
    order = np.argsort(-scores)[:k]
    return [{**index.chunks[i], "score": float(scores[i])} for i in order]


# ── Metrics (ports of hit_rate / mrr) ────────────────────────────────────────
def hit_rate(index: _Index, eval_items: list[dict], k: int) -> tuple[float, list[str]]:
    """Return (hit_rate, list_of_missed_question_ids)."""
    hits = 0
    misses: list[str] = []
    for it in eval_items:
        results = retrieve(index, it["question"], k)
        context = " ".join(r["text"] for r in results).lower()
        if it["answer_span"].lower() in context:
            hits += 1
        else:
            misses.append(it["id"])
    return hits / len(eval_items), misses


def rerank_candidates(query: str, passages: list[str], top_n: int) -> list[dict]:
    """[{index, logit}, ...] for `passages`, best first (local cross-encoder)."""
    scores = _cross_encoder().predict([(query, p) for p in passages])
    order = sorted(range(len(passages)), key=lambda i: -float(scores[i]))[:top_n]
    return [{"index": i, "logit": float(scores[i])} for i in order]


def two_stage(index: _Index, query: str, k1: int, k2: int) -> list[dict]:
    """Bi-encoder retrieve top-k1, cross-encoder rerank to top-k2, with rank deltas."""
    candidates = retrieve(index, query, k1)
    rankings = rerank_candidates(query, [c["text"] for c in candidates], top_n=k2)
    out: list[dict] = []
    for new_rank, row in enumerate(rankings, start=1):
        c = candidates[row["index"]]
        embed_rank = row["index"] + 1
        out.append(
            {
                "doc_id": c["doc_id"],
                "text": c["text"],
                "embed_rank": embed_rank,
                "rerank_rank": new_rank,
                "rerank_logit": row["logit"],
                "rank_delta": embed_rank - new_rank,
            }
        )
    return out
