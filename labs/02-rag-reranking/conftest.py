"""Fixtures for Lab 02 tests.

Mirrors Lab 01: corpus/eval data plus a session-scoped mock NIM so the bi-encoder
and reranker run with no GPU and no NVIDIA_API_KEY. Also pins NVIDIA_API_KEY empty
so the notebook's optional generation cell skips and the solution-execution test
stays offline and deterministic.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = next(
    p
    for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
    if (p / "shared").is_dir()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "labs/02-rag-reranking/data"
SOLUTION_NB = REPO_ROOT / "solutions/02-rag-reranking/lab.ipynb"
MOCK_PORT = 8099
MOCK_URL = f"http://localhost:{MOCK_PORT}/v1"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow end-to-end notebook execution")


@pytest.fixture(scope="session")
def solution_nb() -> Path:
    return SOLUTION_NB


@pytest.fixture(scope="session")
def corpus() -> list[dict]:
    return json.loads((DATA / "corpus.json").read_text())


@pytest.fixture(scope="session")
def eval_set() -> list[dict]:
    return json.loads((DATA / "eval.json").read_text())


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


def _wait_ready(timeout_s: float = 60.0) -> bool:
    import httpx

    ready = MOCK_URL.replace("/v1", "/v2/health/ready")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(ready, timeout=1).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def mock_nim() -> str:
    """Ensure a mock NIM is reachable on :8099 and the clients point at it."""
    os.environ.update(
        USE_MOCK_NIM="true",
        NIM_EMBED_URL=MOCK_URL,
        NIM_RERANK_URL=MOCK_URL,
        NIM_LLM_URL=MOCK_URL,
        MOCK_LATENCY_MS="0",
        NVIDIA_API_KEY="",  # keep the optional generation cell skipped (offline test)
    )
    if not _port_open("localhost", MOCK_PORT):
        import uvicorn

        from shared.mock_nim import app

        config = uvicorn.Config(app, host="127.0.0.1", port=MOCK_PORT, log_level="warning")
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, daemon=True).start()

    if not _wait_ready():
        pytest.skip("mock NIM did not become ready on :8099")
    return MOCK_URL
