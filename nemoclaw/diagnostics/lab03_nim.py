"""Lab 03 — NIM deployment & troubleshooting (offline log/artifact diagnosis).

Faithful port of the verified solution logic in
``labs/03-nim-deployment/lab.ipynb``. Pinned by ``nemoclaw/tests/test_tools.py``
against the labs' canonical numbers (e.g. time-to-ready 148.043s, KV ceiling 28,
H100->fp8 / A100->fp16 profile split, air-gap missing artifacts).
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel

# Runtime knobs the healthy NIM reported (see startup_healthy.log).
MAX_MODEL_LEN = 4096  # tokens reserved per in-flight request
FP8_MIN_COMPUTE = 8.9  # fp8 needs Hopper/Ada (sm89+); A100 is sm80

_TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")


def parse_ts(line: str) -> datetime | None:
    """Pull the leading timestamp out of a NIM log line, if present."""
    m = _TS.search(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f") if m else None


class StartupReport(BaseModel):
    gpu: str
    compute_capability: str
    profile_id: str
    profile_name: str
    precision: str
    tp: int
    kv_cache_tokens: int
    time_to_ready_s: float
    ttft_ms: int | None = None


class ProfileChoice(BaseModel):
    profile_id: str
    backend: str
    precision: str
    tp: int
    reason: str


class Diagnosis(BaseModel):
    failure: str  # short machine key, e.g. "ngc_auth"
    root_cause: str
    fix: str
    evidence: str  # the log line that gave it away


def parse_startup(log: str) -> StartupReport:
    """Extract GPU, selected profile, KV-cache budget, and time-to-ready."""
    lines = log.splitlines()
    gpu = re.search(r"GPU 0: ([^|]+)\| compute capability ([\d.]+)", log)
    prof = re.search(r"Selected profile: (\w+) \(([^)]+)\)", log)
    attrs = re.search(r"precision=(\w+) tp=(\d+)", log)
    kv = re.search(r"max (\d+) tokens", log)
    ttft = re.search(r"TTFT=(\d+) ms", log)

    start = parse_ts(lines[0])
    ready = parse_ts(next(line for line in lines if "Uvicorn running" in line))
    time_to_ready_s = round((ready - start).total_seconds(), 3)

    return StartupReport(
        gpu=gpu.group(1).strip(),
        compute_capability=gpu.group(2),
        profile_id=prof.group(1),
        profile_name=prof.group(2),
        precision=attrs.group(1),
        tp=int(attrs.group(2)),
        kv_cache_tokens=int(kv.group(1)),
        time_to_ready_s=time_to_ready_s,
        ttft_ms=int(ttft.group(1)) if ttft else None,
    )


# ── Profile auto-selection (reproduces the ngc_injector) ─────────────────────
def fp8_supported(compute_capability: str | None) -> bool:
    return (
        compute_capability is not None and float(compute_capability) >= FP8_MIN_COMPUTE
    )


def compatible(profile: dict, detected: dict) -> bool:
    """Can this profile actually run on the detected hardware?"""
    if profile["gpu"] not in (detected["gpu"], "any"):
        return False
    if profile["tp"] > detected["count"]:
        return False
    if profile["precision"] == "fp8" and not fp8_supported(
        detected["compute_capability"]
    ):
        return False
    return True


def profile_score(profile: dict) -> tuple:
    """Rank compatible profiles: optimized backend > fp8 > more tensor-parallelism."""
    return (
        1 if profile["backend"] == "tensorrt_llm" else 0,
        1 if profile["precision"] == "fp8" else 0,
        profile["tp"],
    )


def select_profile(detected: dict, profiles: list[dict]) -> ProfileChoice | None:
    """Reproduce the ngc_injector auto-selection for the detected GPU."""
    candidates = [p for p in profiles if compatible(p, detected)]
    if not candidates:
        return None
    best = max(candidates, key=profile_score)
    reason = (
        f"most-optimized profile compatible with {detected['count']}x "
        f"{detected['gpu'].upper()} (cc {detected['compute_capability']}, "
        f"fp8={'yes' if fp8_supported(detected['compute_capability']) else 'no'})"
    )
    return ProfileChoice(
        profile_id=best["id"],
        backend=best["backend"],
        precision=best["precision"],
        tp=best["tp"],
        reason=reason,
    )


# ── Failure playbook ─────────────────────────────────────────────────────────
KNOWN_FAILURES = [
    {
        "failure": "ngc_auth",
        "signatures": ["401 Unauthorized", "ImagePullBackOff"],
        "root_cause": "The image pull secret (ngc-secret) is missing or holds an "
        "invalid NGC key, so the kubelet cannot pull from nvcr.io.",
        "fix": "Recreate the dockerconfigjson secret with username '$oauthtoken' and a "
        "valid NGC key, and reference it in the pod's imagePullSecrets: "
        "`kubectl create secret docker-registry ngc-secret "
        "--docker-server=nvcr.io --docker-username='$oauthtoken' "
        "--docker-password=$NGC_API_KEY`.",
    },
    {
        "failure": "kv_cache_exhaustion",
        "signatures": ["no available kv cache blocks", "0 free KV cache blocks"],
        "root_cause": "Concurrent requests reserved every KV-cache block; new "
        "requests are rejected with HTTP 503 once the token budget is full.",
        "fix": "Cap client concurrency to what the KV cache holds, lower "
        "max_model_len, raise gpu_memory_utilization, or add GPUs / "
        "tensor-parallelism to enlarge the cache.",
    },
    {
        "failure": "airgap_cache_miss",
        "signatures": ["Cannot materialize engine offline", "NGC is unreachable"],
        "root_cause": "Offline mode needs the selected profile's engine in the local "
        "cache, but the cache lacks artifacts for the profile this GPU resolves to.",
        "fix": "Run `nim download-to-cache --profile <id>` for the TARGET GPU's "
        "profile on a connected host, copy the cache to the air-gapped node, "
        "or pin NIM_MODEL_PROFILE to a profile that is actually cached.",
    },
]


def _first_line_with(log: str, needle: str) -> str:
    return next((ln.strip() for ln in log.splitlines() if needle in ln), needle)


def diagnose(log: str) -> Diagnosis:
    """Match a log against KNOWN_FAILURES and return a structured diagnosis."""
    for rule in KNOWN_FAILURES:
        hit = next((s for s in rule["signatures"] if s in log), None)
        if hit is not None:
            return Diagnosis(
                failure=rule["failure"],
                root_cause=rule["root_cause"],
                fix=rule["fix"],
                evidence=_first_line_with(log, hit),
            )
    return Diagnosis(
        failure="unknown",
        root_cause="No known signature matched.",
        fix="Inspect the log manually.",
        evidence="",
    )


def max_concurrent_requests(kv_cache_tokens: int, tokens_per_request: int) -> int:
    """How many full-context requests fit in the KV cache at once."""
    return kv_cache_tokens // tokens_per_request


def missing_artifacts(required: list[str], present: list[str]) -> list[str]:
    """Artifacts the selected profile needs that aren't in the offline cache."""
    return sorted(set(required) - set(present))
