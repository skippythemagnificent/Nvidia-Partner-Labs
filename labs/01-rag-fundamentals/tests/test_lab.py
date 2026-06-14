"""Validation for Lab 01.

Two kinds of checks:
  - data invariants the notebook and its narrative depend on (corpus shape,
    answer spans, the fixed-vs-sentence chunking claim, cosine-space scores);
  - that the *solution* notebook executes top to bottom against the mock NIM.

The learner `lab.ipynb` is intentionally NOT executed here — its TODO stubs fail
loudly until completed, which is the point. `task lab:test` runs nbmake on it so a
learner can check their own filled-in copy.
"""

from __future__ import annotations

import re
import uuid

import pytest

from shared.vector_store import VectorStore

EXPECTED_DOCS = 50
RETRIEVAL_TOP_K = 3
FIXED_CHUNK_CHARS = 140
SENTENCE_MAX_CHARS = 320


# ── reference chunkers (mirror the notebook exercises) ───────────────────────


def fixed_size_chunks(text: str, size: int, overlap: int = 0) -> list[str]:
    out: list[str] = []
    step = size - overlap
    i = 0
    while i < len(text):
        out.append(text[i : i + size])
        i += step
    return out


def sentence_chunks(text: str, max_chars: int) -> list[str]:
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


# ── data invariants ──────────────────────────────────────────────────────────


def test_corpus_shape(corpus):
    assert len(corpus) == EXPECTED_DOCS
    ids = [d["id"] for d in corpus]
    assert len(ids) == len(set(ids)), "duplicate doc ids"
    for d in corpus:
        assert {"id", "title", "text", "source_url"} <= set(d)
        assert len(d["text"]) > 80, f"{d['id']} text is suspiciously short"


def test_eval_spans_are_verbatim(corpus, eval_set):
    by_id = {d["id"]: d for d in corpus}
    assert len(eval_set) == 12
    for item in eval_set:
        doc = by_id[item["answer_doc_id"]]
        assert item["answer_span"].lower() in doc["text"].lower(), (
            f"{item['id']}: answer_span not found verbatim in {doc['id']}"
        )


# ── retrieval behaviour (needs the mock embedder) ────────────────────────────


def _embed(client, model, texts, input_type):
    res = client.embeddings.create(
        model=model, input=texts, extra_body={"input_type": input_type}
    )
    return [d.embedding for d in res.data]


def _build(client, corpus, chunker, tmp_path):
    name = f"t-{uuid.uuid4().hex[:8]}"
    store = VectorStore(collection=name, persist_dir=str(tmp_path / name))
    rows = [
        {"id": f"{d['id']}#{j}", "doc_id": d["id"], "text": piece}
        for d in corpus
        for j, piece in enumerate(chunker(d["text"]))
    ]
    embs = _embed(
        client, "nvidia/nv-embedqa-e5-v5", [r["text"] for r in rows], "passage"
    )
    store.upsert(
        ids=[r["id"] for r in rows],
        texts=[r["text"] for r in rows],
        embeddings=embs,
        metadatas=[{"doc_id": r["doc_id"]} for r in rows],
    )
    return store


def _hit_rate(client, store, eval_set, k):
    hits, misses = 0, []
    for it in eval_set:
        q_emb = _embed(client, "nvidia/nv-embedqa-e5-v5", [it["question"]], "query")[0]
        ctx = " ".join(r.text for r in store.search(q_emb, top_k=k)).lower()
        if it["answer_span"].lower() in ctx:
            hits += 1
        else:
            misses.append(it["id"])
    return hits / len(eval_set), misses


def test_cosine_space_scores(mock_nim, tmp_path):
    """Self-similarity ~1 and all scores in [-1, 1] — confirms the store is in
    cosine space, not the Euclidean default."""
    from shared.nim_client import get_embed_client

    client = get_embed_client()
    texts = [
        "NIM exposes an OpenAI-compatible API for chat and embeddings.",
        "DCGM Exporter surfaces GPU-level Prometheus metrics.",
    ]
    embs = _embed(client, "nvidia/nv-embedqa-e5-v5", texts, "passage")
    store = VectorStore(collection="cos-check", persist_dir=str(tmp_path / "cos"))
    store.upsert(ids=["a", "b"], texts=texts, embeddings=embs)
    results = store.search(embs[0], top_k=2)
    assert all(-1.0001 <= r.score <= 1.0001 for r in results)
    assert results[0].score == pytest.approx(1.0, abs=1e-3), (
        "self-similarity should be ~1.0"
    )


def test_sentence_chunking_beats_fixed(mock_nim, corpus, eval_set, tmp_path):
    from shared.nim_client import get_embed_client

    client = get_embed_client()
    fixed = _build(
        client, corpus, lambda t: fixed_size_chunks(t, FIXED_CHUNK_CHARS), tmp_path
    )
    sent = _build(
        client, corpus, lambda t: sentence_chunks(t, SENTENCE_MAX_CHARS), tmp_path
    )

    fixed_hr, fixed_misses = _hit_rate(client, fixed, eval_set, RETRIEVAL_TOP_K)
    sent_hr, sent_misses = _hit_rate(client, sent, eval_set, RETRIEVAL_TOP_K)

    assert sent_hr > fixed_hr, "sentence chunking should raise retrieval hit-rate"
    # The boundary-split teaching case: fixed splits q08's answer, sentence keeps it.
    assert "q08" in fixed_misses, (
        "q08 should miss under fixed chunking (boundary split)"
    )
    assert "q08" not in sent_misses, "q08 should hit under sentence chunking"


@pytest.mark.slow
def test_solution_notebook_executes(mock_nim, solution_nb):
    """Execute the completed solution end to end; no cell may raise."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    nb = nbformat.read(str(solution_nb), as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(solution_nb.parent)}},
    )
    client.execute()  # raises CellExecutionError on any unhandled exception
