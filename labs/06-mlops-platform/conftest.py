"""Fixtures for Lab 06 tests.

Lab 06 is fully analytical — no GPU, no NIM, no NVIDIA_API_KEY — so (like Labs 03–05)
there is no mock-NIM server. These fixtures load the raw scrape, the time series, and
the RAGAS runs, and expose the lab notebook path for the execution test.
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

DATA = REPO_ROOT / "labs/06-mlops-platform/data"
LAB_NB = REPO_ROOT / "labs/06-mlops-platform/lab.ipynb"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow end-to-end notebook execution")


@pytest.fixture(scope="session")
def lab_nb() -> Path:
    return LAB_NB


@pytest.fixture(scope="session")
def raw_scrape() -> str:
    return (DATA / "metrics_raw.txt").read_text()


@pytest.fixture(scope="session")
def metrics() -> dict:
    return json.loads((DATA / "metrics.json").read_text())


@pytest.fixture(scope="session")
def eval_runs() -> dict:
    return json.loads((DATA / "eval_runs.json").read_text())
