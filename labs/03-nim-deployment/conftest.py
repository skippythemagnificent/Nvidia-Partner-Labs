"""Fixtures for Lab 03 tests.

Lab 03 is fully offline — no GPU, no NIM, no NVIDIA_API_KEY — so unlike Labs 01–02
there is no mock-NIM server to stand up. These fixtures just load the generated
deployment artifacts (log samples + profile manifest) and expose the solution
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

DATA = REPO_ROOT / "labs/03-nim-deployment/data"
SOLUTION_NB = REPO_ROOT / "solutions/03-nim-deployment/lab.ipynb"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow end-to-end notebook execution")


@pytest.fixture(scope="session")
def solution_nb() -> Path:
    return SOLUTION_NB


@pytest.fixture(scope="session")
def logs() -> dict[str, str]:
    return {p.stem: p.read_text() for p in sorted((DATA / "logs").glob("*.log"))}


@pytest.fixture(scope="session")
def profiles() -> list[dict]:
    return json.loads((DATA / "profiles.json").read_text())["profiles"]


@pytest.fixture(scope="session")
def cache_manifest() -> dict:
    return json.loads((DATA / "cache_manifest.json").read_text())


@pytest.fixture(scope="session")
def deployment() -> dict:
    return json.loads((DATA / "deployment.json").read_text())
