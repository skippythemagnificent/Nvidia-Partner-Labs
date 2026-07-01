"""Fixtures for Lab 05 tests.

Lab 05 runs fully offline against the seeded simulation in `shared/agent_sim.py` —
no GPU, no NIM, no NVIDIA_API_KEY — so (like Labs 03–04) there is no mock-NIM server.
These fixtures load the ticket / backend / adversarial data and expose the lab notebook
notebook path for the execution test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = next(
    p
    for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
    if (p / "shared").is_dir()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "labs/05-agents-orchestration/data"
LAB_NB = REPO_ROOT / "labs/05-agents-orchestration/lab.ipynb"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow end-to-end notebook execution")


@pytest.fixture(scope="session")
def lab_nb() -> Path:
    return LAB_NB


@pytest.fixture(scope="session")
def tickets() -> list[dict]:
    return json.loads((DATA / "tickets.json").read_text())


@pytest.fixture(scope="session")
def backend() -> dict:
    return json.loads((DATA / "backend.json").read_text())


@pytest.fixture(scope="session")
def adversarial() -> list[dict]:
    return json.loads((DATA / "adversarial.json").read_text())
