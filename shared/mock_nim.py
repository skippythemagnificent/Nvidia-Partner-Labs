"""Mock NIM server — OpenAI-compatible endpoints backed by real local models.

Why real models instead of stubs: rank deltas, similarity ranges, and rerank
score distributions must be meaningful so the labs teach correct intuition.

  - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
  - Reranking:  cross-encoder/ms-marco-MiniLM-L-6-v2
  - LLM:        proxied to NVIDIA API Catalog (requires NVIDIA_API_KEY)

Run with: `task mock:start` (binds to :8099).
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

MOCK_LATENCY_MS = int(os.environ.get("MOCK_LATENCY_MS", "120"))

app = FastAPI(title="Mock NIM", version="0.1.0")

_embed_model = None
_rerank_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embed_model


def _get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder

        _rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _rerank_model


async def _simulate_latency() -> None:
    await asyncio.sleep(MOCK_LATENCY_MS / 1000.0)


# ── Embeddings ───────────────────────────────────────────────────────────────


class EmbedRequest(BaseModel):
    model: str
    input: str | list[str]
    input_type: str | None = None  # NV-EmbedQA: "query" | "passage"


@app.post("/v1/embeddings")
async def embeddings(req: EmbedRequest) -> dict[str, Any]:
    await _simulate_latency()
    texts = [req.input] if isinstance(req.input, str) else req.input
    vectors = _get_embed_model().encode(texts, normalize_embeddings=True).tolist()
    return {
        "object": "list",
        "model": req.model,
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": 0},
    }


# ── Reranking (NIM-style) ────────────────────────────────────────────────────


class RerankPassage(BaseModel):
    text: str


class RerankRequest(BaseModel):
    model: str
    query: RerankPassage | str
    passages: list[RerankPassage] | list[str]
    top_n: int | None = None


@app.post("/v1/ranking")
async def rerank(req: RerankRequest) -> dict[str, Any]:
    await _simulate_latency()
    q = req.query.text if isinstance(req.query, RerankPassage) else req.query
    passages = [
        p.text if isinstance(p, RerankPassage) else p for p in req.passages
    ]
    pairs = [(q, p) for p in passages]
    scores = _get_rerank_model().predict(pairs).tolist()
    ranked = sorted(
        ({"index": i, "logit": float(s)} for i, s in enumerate(scores)),
        key=lambda x: x["logit"],
        reverse=True,
    )
    if req.top_n:
        ranked = ranked[: req.top_n]
    return {"model": req.model, "rankings": ranked}


# ── LLM (proxy to API Catalog) ───────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise HTTPException(500, "NVIDIA_API_KEY not set — mock LLM proxies to API Catalog")

    import httpx

    await _simulate_latency()
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
    return r.json()


# ── NIM-style health endpoints ───────────────────────────────────────────────


@app.get("/v2/health/ready")
def health_ready() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/v2/models")
def models() -> dict[str, Any]:
    return {
        "models": [
            {"name": "mock/embed-minilm-l6", "platform": "sentence-transformers"},
            {"name": "mock/rerank-msmarco-minilm-l6", "platform": "sentence-transformers"},
            {"name": "mock/llm-proxy", "platform": "api-catalog"},
        ]
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "mock-nim",
        "started": time.time(),
        "latency_ms": MOCK_LATENCY_MS,
    }
