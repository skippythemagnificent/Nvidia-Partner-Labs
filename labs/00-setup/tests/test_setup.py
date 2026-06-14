"""Setup validation — run with: `task lab:test LAB=00-setup`."""
from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_shared_package_importable() -> None:
    importlib.import_module("shared.nim_client")
    importlib.import_module("shared.utils")
    importlib.import_module("shared.vector_store")


def test_env_file_present() -> None:
    assert (ROOT / ".env").exists() or (ROOT / ".env.example").exists(), (
        "Expected .env or .env.example at project root"
    )


def test_nvidia_api_key_or_mock() -> None:
    use_mock = os.environ.get("USE_MOCK_NIM", "false").lower() == "true"
    has_key = bool(os.environ.get("NVIDIA_API_KEY", "").strip())
    assert use_mock or has_key, (
        "Set NVIDIA_API_KEY in .env, or set USE_MOCK_NIM=true to use the local mock."
    )


@pytest.mark.skipif(shutil.which("pulumi") is None, reason="pulumi not installed")
def test_infra_dir_present() -> None:
    assert (ROOT / "infra" / "pyproject.toml").exists()
    assert (ROOT / "infra" / "__main__.py").exists()
