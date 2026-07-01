"""Validation for Lab 03.

Lab 03 is an offline log-analysis lab, so these tests reimplement the reference
parsing/selection/diagnosis logic (mirroring the notebook) and assert the canonical
numbers, plus data invariants and end-to-end execution of the lab notebook.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import pytest

FP8_MIN_COMPUTE = 8.9
MAX_MODEL_LEN = 4096
EXPECTED_TTR_S = 148.043
EXPECTED_KV_TOKENS = 117440
H100_PROFILE = "8835c31752fd"
A100_PROFILE = "6f1ac2d40b77"

_TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")


# ── reference logic (mirrors the notebook) ───────────────────────────────────


def _ts(line: str) -> datetime | None:
    m = _TS.search(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f") if m else None


def parse_startup(log: str) -> dict:
    lines = log.splitlines()
    gpu = re.search(r"GPU 0: ([^|]+)\| compute capability ([\d.]+)", log)
    prof = re.search(r"Selected profile: (\w+) \(([^)]+)\)", log)
    attrs = re.search(r"precision=(\w+) tp=(\d+)", log)
    kv = re.search(r"max (\d+) tokens", log)
    start = _ts(lines[0])
    ready = _ts(next(line for line in lines if "Uvicorn running" in line))
    return {
        "gpu": gpu.group(1).strip(),
        "compute_capability": gpu.group(2),
        "profile_id": prof.group(1),
        "precision": attrs.group(1),
        "tp": int(attrs.group(2)),
        "kv_cache_tokens": int(kv.group(1)),
        "time_to_ready_s": round((ready - start).total_seconds(), 3),
    }


def _fp8_ok(cc):
    return cc is not None and float(cc) >= FP8_MIN_COMPUTE


def _compatible(p, det):
    if p["gpu"] not in (det["gpu"], "any"):
        return False
    if p["tp"] > det["count"]:
        return False
    if p["precision"] == "fp8" and not _fp8_ok(det["compute_capability"]):
        return False
    return True


def _score(p):
    return (1 if p["backend"] == "tensorrt_llm" else 0, 1 if p["precision"] == "fp8" else 0, p["tp"])


def select_profile(det, profiles):
    cands = [p for p in profiles if _compatible(p, det)]
    return max(cands, key=_score) if cands else None


def diagnose(log: str) -> str:
    if "401 Unauthorized" in log or "ImagePullBackOff" in log:
        return "ngc_auth"
    if "no available kv cache blocks" in log or "0 free KV cache blocks" in log:
        return "kv_cache_exhaustion"
    if "Cannot materialize engine offline" in log or "NGC is unreachable" in log:
        return "airgap_cache_miss"
    return "unknown"


# ── data invariants ──────────────────────────────────────────────────────────


def test_all_logs_present(logs):
    assert set(logs) == {
        "airgapped_offline", "kv_cache_exhaustion", "ngc_auth_failure", "startup_healthy"
    }
    assert all(text.strip() for text in logs.values())


def test_profile_manifest_shape(profiles):
    assert len(profiles) == 6
    ids = [p["id"] for p in profiles]
    assert len(ids) == len(set(ids))
    for p in profiles:
        assert {"id", "backend", "gpu", "precision", "tp", "artifacts"} <= set(p)
    assert any(p["backend"] == "vllm" and p["gpu"] == "any" for p in profiles), "need a portable fallback"


# ── startup parsing ──────────────────────────────────────────────────────────


def test_parse_startup_time_to_ready(logs):
    r = parse_startup(logs["startup_healthy"])
    assert r["profile_id"] == H100_PROFILE
    assert r["precision"] == "fp8"
    assert r["kv_cache_tokens"] == EXPECTED_KV_TOKENS
    assert round(r["time_to_ready_s"], 1) == round(EXPECTED_TTR_S, 1)


# ── profile selection ────────────────────────────────────────────────────────


def test_select_profile_matches_healthy_log(logs, profiles, deployment):
    """Auto-selection reproduces the profile the healthy H100 log actually picked."""
    chosen = select_profile(deployment["dev_box_gpu"], profiles)
    assert chosen["id"] == parse_startup(logs["startup_healthy"])["profile_id"] == H100_PROFILE


def test_select_profile_a100_drops_fp8(profiles, deployment):
    """A100 (sm80) cannot run fp8, so it falls to the fp16 profile."""
    chosen = select_profile(deployment["airgapped_target_gpu"], profiles)
    assert chosen["id"] == A100_PROFILE
    assert chosen["precision"] == "fp16"


def test_select_profile_uses_extra_gpus(profiles):
    """With 2 A100s, the tp=2 throughput profile beats tp=1."""
    chosen = select_profile({"gpu": "a100", "compute_capability": "8.0", "count": 2}, profiles)
    assert chosen["tp"] == 2


def test_select_profile_unknown_gpu_falls_back(profiles):
    """An unsupported GPU resolves only to the portable vLLM 'any' profile."""
    chosen = select_profile({"gpu": "v100", "compute_capability": "7.0", "count": 1}, profiles)
    assert chosen["backend"] == "vllm" and chosen["gpu"] == "any"


# ── diagnosis ────────────────────────────────────────────────────────────────


def test_diagnose_classifies_every_log(logs):
    assert diagnose(logs["ngc_auth_failure"]) == "ngc_auth"
    assert diagnose(logs["kv_cache_exhaustion"]) == "kv_cache_exhaustion"
    assert diagnose(logs["airgapped_offline"]) == "airgap_cache_miss"
    assert diagnose(logs["startup_healthy"]) == "unknown"


# ── capacity math ────────────────────────────────────────────────────────────


def test_kv_cache_concurrency_ceiling(logs):
    """The KV budget caps full-context concurrency below the offered load."""
    kv = parse_startup(logs["startup_healthy"])["kv_cache_tokens"]
    ceiling = kv // MAX_MODEL_LEN
    assert ceiling == 28
    assert ceiling < 64, "offered load (64) must exceed the ceiling to cause 503s"


# ── air-gapped cache gap ─────────────────────────────────────────────────────


def test_missing_artifacts_for_target_profile(profiles, cache_manifest, deployment):
    """The cache (prepped on H100) is missing the A100 profile's engine files."""
    target = select_profile(deployment["airgapped_target_gpu"], profiles)
    required = next(p["artifacts"] for p in profiles if p["id"] == target["id"])
    missing = set(required) - set(cache_manifest["present_artifacts"])
    assert missing == {
        f"models/{A100_PROFILE}/config.json",
        f"models/{A100_PROFILE}/rank0.engine",
    }


# ── notebook execution ─────────────────────────────────────


@pytest.mark.slow
def test_lab_notebook_executes(lab_nb):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    nb = nbformat.read(str(lab_nb), as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(lab_nb.parent)}},
    )
    client.execute()
