"""NIM client factories + health check.

All labs must use these factories instead of constructing `openai.OpenAI` directly.
Endpoint URLs are read from the environment (`.env`), which is populated either by
`task setup` or by `task infra:env STACK=...` after a Pulumi deploy.
"""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"


def _base_url(env_key: str) -> str:
    return os.environ.get(env_key, _DEFAULT_BASE)


def _api_key() -> str:
    # Treat an empty value like a missing one: the OpenAI client rejects "", and the
    # mock NIM ignores auth anyway, so fall back to a harmless placeholder.
    return os.environ.get("NVIDIA_API_KEY") or "not-used"


def get_embed_client() -> OpenAI:
    return OpenAI(base_url=_base_url("NIM_EMBED_URL"), api_key=_api_key())


def get_rerank_client() -> OpenAI:
    return OpenAI(base_url=_base_url("NIM_RERANK_URL"), api_key=_api_key())


def rerank(
    query: str,
    passages: list[str],
    model: str = "nvidia/nv-rerankqa-mistral-4b-v3",
    top_n: int | None = None,
    timeout: float = 30.0,
) -> list[dict]:
    """Score `passages` against `query` with the reranking NIM.

    Reranking is not part of the OpenAI schema, so it is a plain POST to the NIM's
    ``/ranking`` endpoint rather than a method on the OpenAI client. Returns the
    ``rankings`` list — dicts of ``{"index": <position in passages>, "logit":
    <relevance score>}`` sorted from most to least relevant. Higher logit = more
    relevant; the index maps each ranking back to the input `passages`.
    """
    url = _base_url("NIM_RERANK_URL").rstrip("/") + "/ranking"
    payload: dict = {
        "model": model,
        "query": {"text": query},
        "passages": [{"text": p} for p in passages],
    }
    if top_n is not None:
        payload["top_n"] = top_n
    r = httpx.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {_api_key()}"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["rankings"]


def get_llm_client() -> OpenAI:
    return OpenAI(base_url=_base_url("NIM_LLM_URL"), api_key=_api_key())


def health_check() -> dict[str, bool]:
    """Probe `/v2/health/ready` on each configured NIM endpoint."""
    results: dict[str, bool] = {}
    for name, key in (
        ("embed", "NIM_EMBED_URL"),
        ("rerank", "NIM_RERANK_URL"),
        ("llm", "NIM_LLM_URL"),
    ):
        url = _base_url(key).replace("/v1", "/v2/health/ready")
        try:
            r = httpx.get(url, timeout=5)
            results[name] = r.status_code == 200
        except Exception:
            results[name] = False
    return results
