"""Fixtures for Lab 01 tests.

Lives at the lab root (not under tests/) so it also applies when the notebook
itself is collected by nbmake. Provides the corpus/eval data and a session-scoped
mock NIM so the tests run with no GPU and no NVIDIA_API_KEY.
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

# Repo root = nearest ancestor holding `shared/`. Put it on the path so tests can
# import the shared package regardless of pytest's rootdir.
REPO_ROOT = next(
    p
    for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
    if (p / "shared").is_dir()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "labs/01-rag-fundamentals/data"
SOLUTION_NB = REPO_ROOT / "solutions/01-rag-fundamentals/lab.ipynb"
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
    """Ensure a mock NIM is reachable on :8099 and env points the clients at it.

    Reuses an already-running server (e.g. `task mock:start`) if present;
    otherwise boots one in a background thread for the test session.
    """
    os.environ.update(
        USE_MOCK_NIM="true",
        NIM_EMBED_URL=MOCK_URL,
        NIM_RERANK_URL=MOCK_URL,
        NIM_LLM_URL=MOCK_URL,
        MOCK_LATENCY_MS="0",
    )

    if not _port_open("localhost", MOCK_PORT):
        import uvicorn

        from shared.mock_nim import app

        config = uvicorn.Config(
            app, host="127.0.0.1", port=MOCK_PORT, log_level="warning"
        )
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, daemon=True).start()

    if not _wait_ready():
        pytest.skip("mock NIM did not become ready on :8099")
    return MOCK_URL
