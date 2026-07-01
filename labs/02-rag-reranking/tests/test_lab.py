"""Validation for Lab 02.

Data invariants, the two-stage reranking pipeline (reuses the shared rerank()
helper + mock cross-encoder), and end-to-end execution of the lab notebook.
"""
from __future__ import annotations

import re
import uuid

import pytest

from shared.nim_client import get_embed_client, rerank
from shared.vector_store import VectorStore

EXPECTED_DOCS = 10
SENTENCE_MAX_CHARS = 320
OVER_RETRIEVE_K1 = 20
RERANK_K2 = 5
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
RERANK_MODEL = "nvidia/rerank-qa-mistral-4b"


# ── reference pipeline (mirrors the notebook) ────────────────────────────────


def sentence_chunks(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            chunks.append(current.strip())
            current = s
        else:
            current = (current + " " + s).strip()
    if current:
        chunks.append(current.strip())
    return chunks


def _embed(client, texts, input_type):
    res = client.embeddings.create(
        model=EMBED_MODEL, input=texts, extra_body={"input_type": input_type}
    )
    return [d.embedding for d in res.data]


@pytest.fixture(scope="module")
def index(mock_nim, corpus, tmp_path_factory):
    client = get_embed_client()
    persist = tmp_path_factory.mktemp("chroma02")
    rows = [
        {"id": f'{d["id"]}#{j}', "doc_id": d["id"], "text": piece}
        for d in corpus
        for j, piece in enumerate(sentence_chunks(d["text"], SENTENCE_MAX_CHARS))
    ]
    store = VectorStore(collection=f"t-{uuid.uuid4().hex[:8]}", persist_dir=str(persist))
    store.upsert(
        ids=[r["id"] for r in rows],
        texts=[r["text"] for r in rows],
        embeddings=_embed(client, [r["text"] for r in rows], "passage"),
        metadatas=[{"doc_id": r["doc_id"]} for r in rows],
    )
    return store, client, len(rows)


def _two_stage(index, query, k1=OVER_RETRIEVE_K1, k2=RERANK_K2):
    store, client, _ = index
    q_emb = _embed(client, [query], "query")[0]
    cands = store.search(q_emb, top_k=k1)
    rankings = rerank(query, [c.text for c in cands], model=RERANK_MODEL, top_n=k1)
    out = []
    for j, entry in enumerate(rankings, start=1):
        c = cands[entry["index"]]
        out.append({
            "text": c.text, "doc_id": c.metadata["doc_id"],
            "embed_rank": entry["index"] + 1, "rerank_rank": j,
            "rerank_score": entry["logit"], "rank_delta": (entry["index"] + 1) - j,
        })
    return out[:k2]


# ── data invariants ──────────────────────────────────────────────────────────


def test_corpus_shape(corpus):
    assert len(corpus) == EXPECTED_DOCS
    ids = [d["id"] for d in corpus]
    assert len(ids) == len(set(ids))
    for d in corpus:
        assert {"id", "title", "text"} <= set(d)


def test_eval_spans_are_verbatim(corpus, eval_set):
    by_id = {d["id"]: d for d in corpus}
    assert len(eval_set) == 12
    for item in eval_set:
        doc = by_id[item["answer_doc_id"]]
        assert item["answer_span"].lower() in doc["text"].lower(), item["id"]


# ── reranking behaviour ──────────────────────────────────────────────────────


def test_reranking_changes_order(index, eval_set):
    """At least one query must have a chunk whose rank the reranker changed."""
    moved = [
        c["rank_delta"]
        for it in eval_set
        for c in _two_stage(index, it["question"])
        if c["rank_delta"]
    ]
    assert moved and max(abs(d) for d in moved) > 0, "reranking returned identical order"


def test_top_rerank_score_above_threshold(index):
    """A clearly-relevant query's top reranked chunk has a positive logit."""
    q = "What is the difference between the APR and the interest rate on a personal loan?"
    top = _two_stage(index, q)[0]
    assert top["rerank_score"] > 0.75


def test_reranking_improves_mrr(index, eval_set):
    def mrr(rank_fn):
        total = 0.0
        for it in eval_set:
            for rank, r in enumerate(rank_fn(it["question"]), start=1):
                if it["answer_span"].lower() in r.lower():
                    total += 1.0 / rank
                    break
        return total / len(eval_set)

    store, client, _ = index
    def bi_only(q):
        return [r.text for r in store.search(_embed(client, [q], "query")[0], top_k=RERANK_K2)]
    def two_stage(q):
        return [c["text"] for c in _two_stage(index, q)]

    assert mrr(two_stage) >= mrr(bi_only), "two-stage should not reduce MRR"


def test_over_retrieve_floor(index, eval_set):
    """k1 too small drops answers the bi-encoder ranked just out of the window."""
    q04 = next(it for it in eval_set if it["id"] == "q04")
    span = q04["answer_span"].lower()
    small = any(span in c["text"].lower() for c in _two_stage(index, q04["question"], k1=3))
    big = any(span in c["text"].lower() for c in _two_stage(index, q04["question"], k1=20))
    assert big and not small, "expected k1=3 to lose q04 and k1=20 to recover it"


def test_nim_latency_in_mock_mode(index):
    """Each NIM call returns quickly in mock mode (MOCK_LATENCY_MS=0)."""
    from shared.utils import timed_call

    store, client, _ = index
    timings = []
    _, ms = timed_call(_embed, client, ["a quick latency probe"], "query")
    timings.append(ms)
    _, ms = timed_call(rerank, "probe", ["alpha", "beta", "gamma"], model=RERANK_MODEL)
    timings.append(ms)
    assert all(t < 500 for t in timings), f"a mock NIM call exceeded 500ms: {timings}"


# ── notebook execution ─────────────────────────────────────


@pytest.mark.slow
def test_lab_notebook_executes(mock_nim, lab_nb):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    nb = nbformat.read(str(lab_nb), as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(lab_nb.parent)}},
    )
    client.execute()
