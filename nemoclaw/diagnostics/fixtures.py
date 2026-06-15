"""Single source of truth for locating and loading the lab data fixtures.

Every diagnostic tool operates on the *same committed fixtures* the labs ship
(`labs/NN-.../data/`), so NeMoClaw's answers match the labs' verified expected
outputs exactly. Fixtures are loaded lazily and cached so the MCP server starts fast
and only reads what a given tool needs.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# nemoclaw/diagnostics/fixtures.py -> parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
LABS = REPO_ROOT / "labs"


def _load_json(rel: str):
    return json.loads((LABS / rel).read_text())


# ── Lab 01/02 — RAG corpora ──────────────────────────────────────────────────
@lru_cache(maxsize=None)
def rag_corpus(lab: str) -> list[dict]:
    """`lab` is "01" or "02". Returns the list of {id,title,text} docs."""
    folder = "01-rag-fundamentals" if lab == "01" else "02-rag-reranking"
    return _load_json(f"{folder}/data/corpus.json")


@lru_cache(maxsize=None)
def rag_eval(lab: str) -> list[dict]:
    folder = "01-rag-fundamentals" if lab == "01" else "02-rag-reranking"
    return _load_json(f"{folder}/data/eval.json")


# ── Lab 03 — NIM deployment ──────────────────────────────────────────────────
@lru_cache(maxsize=None)
def nim_logs() -> dict[str, str]:
    """{log_name: text} for every sample log in labs/03/data/logs/."""
    log_dir = LABS / "03-nim-deployment/data/logs"
    return {p.stem: p.read_text() for p in sorted(log_dir.glob("*.log"))}


@lru_cache(maxsize=None)
def nim_profiles() -> list[dict]:
    return _load_json("03-nim-deployment/data/profiles.json")["profiles"]


@lru_cache(maxsize=None)
def nim_deployment() -> dict:
    return _load_json("03-nim-deployment/data/deployment.json")


@lru_cache(maxsize=None)
def nim_cache_manifest() -> dict:
    return _load_json("03-nim-deployment/data/cache_manifest.json")


# ── Lab 04 — GPU architecture ────────────────────────────────────────────────
@lru_cache(maxsize=None)
def gpus() -> dict[str, dict]:
    return {
        g["id"]: g for g in _load_json("04-gpu-architecture/data/gpus.json")["gpus"]
    }


@lru_cache(maxsize=None)
def models() -> dict[str, dict]:
    return {
        m["id"]: m for m in _load_json("04-gpu-architecture/data/models.json")["models"]
    }


@lru_cache(maxsize=None)
def workloads() -> dict[str, dict]:
    return {
        w["id"]: w
        for w in _load_json("04-gpu-architecture/data/workloads.json")["workloads"]
    }


# ── Lab 05 — agents ──────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def tickets() -> list[dict]:
    return _load_json("05-agents-orchestration/data/tickets.json")


@lru_cache(maxsize=None)
def agent_backend() -> dict:
    return _load_json("05-agents-orchestration/data/backend.json")


@lru_cache(maxsize=None)
def adversarial() -> list[dict]:
    return _load_json("05-agents-orchestration/data/adversarial.json")


# ── Lab 06 — MLOps ───────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def metrics() -> dict:
    """{model, step_s, slo, samples} — the 30-min simulated Prometheus scrape."""
    return _load_json("06-mlops-platform/data/metrics.json")


@lru_cache(maxsize=None)
def metrics_raw() -> str:
    return (LABS / "06-mlops-platform/data/metrics_raw.txt").read_text()


@lru_cache(maxsize=None)
def eval_runs() -> dict:
    return _load_json("06-mlops-platform/data/eval_runs.json")
