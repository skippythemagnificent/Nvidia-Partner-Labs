"""Fixtures for Lab 04 tests.

Lab 04 is fully analytical — no GPU, no NIM, no NVIDIA_API_KEY — so (like Lab 03)
there is no mock-NIM server. These fixtures load the GPU/model/workload tables and
expose the lab notebook path for the execution test.
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

DATA = REPO_ROOT / "labs/04-gpu-architecture/data"
LAB_NB = REPO_ROOT / "labs/04-gpu-architecture/lab.ipynb"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow end-to-end notebook execution")


@pytest.fixture(scope="session")
def lab_nb() -> Path:
    return LAB_NB


@pytest.fixture(scope="session")
def gpus() -> dict[str, dict]:
    return {g["id"]: g for g in json.loads((DATA / "gpus.json").read_text())["gpus"]}


@pytest.fixture(scope="session")
def models() -> dict[str, dict]:
    return {m["id"]: m for m in json.loads((DATA / "models.json").read_text())["models"]}


@pytest.fixture(scope="session")
def workloads() -> dict[str, dict]:
    return {w["id"]: w for w in json.loads((DATA / "workloads.json").read_text())["workloads"]}
