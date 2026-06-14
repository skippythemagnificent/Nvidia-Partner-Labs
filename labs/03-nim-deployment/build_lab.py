"""Author the Lab 03 notebooks from a single source.

Emits two notebooks that stay in sync:
  - labs/03-nim-deployment/lab.ipynb        — learner copy (exercise cells stubbed)
  - solutions/03-nim-deployment/lab.ipynb   — completed, nbmake-clean copy

Same pattern as Labs 01–02: each `code(solution, stub)` cell carries both forms,
markdown is shared, and you edit THIS file (never the .ipynb) then regenerate:

    uv run python labs/03-nim-deployment/build_lab.py

Lab 03 is an offline troubleshooting lab: no GPU, no NIM calls. Every cell parses
or reasons about the artifacts in data/ produced by data/generate.py, so the
solution notebook executes anywhere with no network and no NVIDIA_API_KEY.
"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/03-nim-deployment/lab.ipynb"
SOL = ROOT / "solutions/03-nim-deployment/lab.ipynb"

CELLS = []
def md(s): CELLS.append(("md", s))
def code(sol, stub=None): CELLS.append(("code", (sol, stub)))

# ── Scenario ─────────────────────────────────────────────────────────────────
md("""# Lab 03 — NIM Deployment & Troubleshooting

## Scenario

An ISV is shipping its first **on-prem NIM** to an air-gapped federal customer.
The customer's inference node has no internet, no NGC access at runtime, and a hard
go-live date on Friday. The team has a Helm chart and an NVIDIA API key — that's the
easy part. What sinks deployments like this is everything *around* the chart: a pull
secret that 401s, a server that looks healthy until real traffic exhausts its KV
cache, and an offline model cache that was prepped on the wrong GPU.

This lab is **offline by design** — no GPU, no live NIM. You'll work the way an
on-call solutions engineer actually does: reading real-shaped NIM, Triton, and
Kubernetes logs, and writing small functions that turn those logs into a diagnosis.
You'll learn to read a NIM's **TRT-LLM profile auto-selection**, measure
**time-to-ready**, and root-cause the three failures above from their log
signatures. Everything loads from `data/` — produced by `data/generate.py`.""")

# ── Setup ────────────────────────────────────────────────────────────────────
md("""## Setup

Loads the four log samples, the model's profile manifest (`nim list-model-profiles`
output), the air-gapped cache manifest, and the deployment description. Defines the
three Pydantic models your exercises return — `StartupReport`, `ProfileChoice`, and
`Diagnosis` — and the timestamp helper used to measure time-to-ready.""")

code(r'''import json
import re
import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from rich import print as rprint

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "shared").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.utils import display_metrics_table

DATA = REPO_ROOT / "labs/03-nim-deployment/data"
LOGS = {p.stem: p.read_text() for p in sorted((DATA / "logs").glob("*.log"))}
MANIFEST = json.loads((DATA / "profiles.json").read_text())
PROFILES = MANIFEST["profiles"]
CACHE = json.loads((DATA / "cache_manifest.json").read_text())
DEPLOYMENT = json.loads((DATA / "deployment.json").read_text())

# Runtime knobs the healthy NIM reported (see startup_healthy.log).
MAX_MODEL_LEN = 4096          # tokens reserved per in-flight request
FP8_MIN_COMPUTE = 8.9         # fp8 needs Hopper/Ada (sm89+); A100 is sm80

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
    failure: str        # short machine key, e.g. "ngc_auth"
    root_cause: str
    fix: str
    evidence: str       # the log line that gave it away


print(f"logs: {sorted(LOGS)}")
print(f"profiles: {len(PROFILES)} for {MANIFEST['model']} | mock=offline (no GPU, no NIM)")''')

md("""**Expected output:**
```
logs: ['airgapped_offline', 'kv_cache_exhaustion', 'ngc_auth_failure', 'startup_healthy']
profiles: 6 for meta/llama-3.1-8b-instruct | mock=offline (no GPU, no NIM)
```""")

# ── 1. Anatomy of a healthy startup ──────────────────────────────────────────
md("""## 1 · What a healthy NIM startup looks like

### Concept

Before you can recognize a *broken* NIM you need to know what a clean one says on the
way up. A NIM's startup log narrates four things in order:

1. **GPU detection** — which device(s) it found and their compute capability.
2. **Profile auto-selection** — the `ngc_injector` inspects the model's profile
   manifest and picks the single most-optimized TRT-LLM profile your hardware can
   run (and tells you *why*). This is the line that decides your throughput.
3. **Engine + KV-cache allocation** — it loads the prebuilt engine and reserves the
   KV cache, reporting the total token budget (the number that caps concurrency).
4. **Ready** — Triton comes up, then `Uvicorn running` means the OpenAI-compatible
   endpoint is live. The clock from the first line to *this* line is your
   **time-to-ready**, the metric you quote in a go-live runbook.

### Your task — `parse_startup`

Read `startup_healthy.log` and produce a `StartupReport`. The field regexes are
written for you; the one thing left is the **measure that matters**: time-to-ready.

**Walkthrough.** `parse_ts(line)` turns a log line into a `datetime`. The first log
line is when the process started; the line containing `"Uvicorn running"` is when it
began serving. Subtract the two and take `.total_seconds()`.

**Step by step:**

1. `start = parse_ts(lines[0])`.
2. Find the ready line: `next(l for l in lines if "Uvicorn running" in l)`, and
   `parse_ts` it.
3. `time_to_ready_s = round((ready - start).total_seconds(), 3)`.""")

code(r'''def parse_startup(log: str) -> StartupReport:
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


report = parse_startup(LOGS["startup_healthy"])
rprint(report)
display_metrics_table(
    {"time-to-ready (s)": report.time_to_ready_s, "first-token latency (ms)": float(report.ttft_ms)},
    title=f"Healthy startup — {report.gpu}",
)''',
r'''def parse_startup(log: str) -> StartupReport:
    """Extract GPU, selected profile, KV-cache budget, and time-to-ready."""
    lines = log.splitlines()
    gpu = re.search(r"GPU 0: ([^|]+)\| compute capability ([\d.]+)", log)
    prof = re.search(r"Selected profile: (\w+) \(([^)]+)\)", log)
    attrs = re.search(r"precision=(\w+) tp=(\d+)", log)
    kv = re.search(r"max (\d+) tokens", log)
    ttft = re.search(r"TTFT=(\d+) ms", log)

    # TODO: measure time-to-ready. start = parse_ts(lines[0]); find the line
    # containing "Uvicorn running" and parse_ts it; subtract for total_seconds()
    # and round(..., 3).
    time_to_ready_s = None  # replace this line
    assert time_to_ready_s is not None, "Complete the time-to-ready measure before continuing"

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


report = parse_startup(LOGS["startup_healthy"])
rprint(report)
display_metrics_table(
    {"time-to-ready (s)": report.time_to_ready_s, "first-token latency (ms)": float(report.ttft_ms)},
    title=f"Healthy startup — {report.gpu}",
)''')

md("""**Expected output** (the report fields, then the metrics table):
```
StartupReport(gpu='NVIDIA H100 80GB HBM3', compute_capability='9.0',
    profile_id='8835c31752fd', profile_name='tensorrt_llm-h100-fp8-tp1-throughput',
    precision='fp8', tp=1, kv_cache_tokens=117440, time_to_ready_s=148.043, ttft_ms=243)
        Healthy startup — NVIDIA H100 80GB HBM3
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric                     ┃    Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ time-to-ready (s)          │ 148.0430 │
│ first-token latency (ms)   │ 243.0000 │
└────────────────────────────┴──────────┘
```
148 s is mostly engine load and KV-cache allocation. Note the selected profile —
**fp8 on the H100** — and hold onto `kv_cache_tokens=117440`; both come back below.""")

# ── 2. Profile selection ─────────────────────────────────────────────────────
md("""## 2 · Why *that* profile? Reproducing the auto-selection

### Concept

A NIM ships one engine per **profile**: a (GPU family × precision × tensor-parallel
size × tuning goal) combination. `nim list-model-profiles` prints the manifest; at
startup the `ngc_injector` picks the single most-optimized profile your hardware can
actually run. The rules it applies:

- **GPU family must match** (an H100 engine won't load on an A100), or fall back to
  the portable `any`/vLLM profile.
- **Tensor-parallel size must fit**: a `tp=2` profile needs ≥ 2 GPUs.
- **Precision must be supported**: **fp8 requires compute capability ≥ 8.9**
  (Hopper/Ada). An A100 (sm80) can't run fp8 and drops to fp16.
- Among what's left, prefer the **optimized TRT-LLM backend over the generic vLLM
  fallback**, then **fp8 over fp16** (faster), then a **larger tp** when extra GPUs
  are available (more throughput).

Knowing this is what lets you answer "why did it pick fp16 when I paid for fp8?" and
when to override with `NIM_MODEL_PROFILE`.

### Your task — `select_profile`

The two helpers below encode the rules: `compatible(p, detected)` is the hard filter,
`profile_score(p)` ranks the survivors (higher is more optimized). Implement the
selection itself.

**Step by step:**

1. Keep only `compatible(p, detected)` profiles → `candidates`.
2. If none are compatible, return `None`.
3. Otherwise pick `max(candidates, key=profile_score)` and wrap it in a
   `ProfileChoice` (a `reason` string is built for you).""")

code(r'''def fp8_supported(compute_capability: str | None) -> bool:
    return compute_capability is not None and float(compute_capability) >= FP8_MIN_COMPUTE


def compatible(profile: dict, detected: dict) -> bool:
    """Can this profile actually run on the detected hardware?"""
    if profile["gpu"] not in (detected["gpu"], "any"):
        return False
    if profile["tp"] > detected["count"]:
        return False
    if profile["precision"] == "fp8" and not fp8_supported(detected["compute_capability"]):
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
        profile_id=best["id"], backend=best["backend"],
        precision=best["precision"], tp=best["tp"], reason=reason,
    )


for label, det in [("dev box", DEPLOYMENT["dev_box_gpu"]),
                   ("air-gapped target", DEPLOYMENT["airgapped_target_gpu"])]:
    choice = select_profile(det, PROFILES)
    print(f"{label:>18} ({det['gpu']} x{det['count']}): {choice.profile_id} "
          f"| {choice.backend} {choice.precision} tp{choice.tp}")''',
r'''def fp8_supported(compute_capability: str | None) -> bool:
    return compute_capability is not None and float(compute_capability) >= FP8_MIN_COMPUTE


def compatible(profile: dict, detected: dict) -> bool:
    """Can this profile actually run on the detected hardware?"""
    if profile["gpu"] not in (detected["gpu"], "any"):
        return False
    if profile["tp"] > detected["count"]:
        return False
    if profile["precision"] == "fp8" and not fp8_supported(detected["compute_capability"]):
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
    # TODO: keep the compatible(p, detected) profiles; if none, return None;
    # otherwise pick max(..., key=profile_score) and return the ProfileChoice below.
    best = None  # replace this line
    assert best is not None, "Complete select_profile before continuing"
    reason = (
        f"most-optimized profile compatible with {detected['count']}x "
        f"{detected['gpu'].upper()} (cc {detected['compute_capability']}, "
        f"fp8={'yes' if fp8_supported(detected['compute_capability']) else 'no'})"
    )
    return ProfileChoice(
        profile_id=best["id"], backend=best["backend"],
        precision=best["precision"], tp=best["tp"], reason=reason,
    )


for label, det in [("dev box", DEPLOYMENT["dev_box_gpu"]),
                   ("air-gapped target", DEPLOYMENT["airgapped_target_gpu"])]:
    choice = select_profile(det, PROFILES)
    print(f"{label:>18} ({det['gpu']} x{det['count']}): {choice.profile_id} "
          f"| {choice.backend} {choice.precision} tp{choice.tp}")''')

md("""**Expected output** (the dev box and the target resolve to *different* profiles
— remember that):
```
           dev box (h100 x1): 8835c31752fd | tensorrt_llm fp8 tp1
 air-gapped target (a100 x1): 6f1ac2d40b77 | tensorrt_llm fp16 tp1
```
The H100 dev box gets the fp8 engine; the A100 target gets fp16, because sm80 can't
run fp8. They need **different cache artifacts** — the seed of Failure 3 below. (Sanity
check: `select_profile` reproduces the healthy log's `8835c31752fd` for the H100.)""")

# ── 3. Diagnosis framework ───────────────────────────────────────────────────
md("""## 3 · From log to diagnosis

### Concept

On-call work is pattern matching: a failure leaves a **signature** in the log, and
each signature maps to a root cause and a fix. We'll encode that as a small rule
table, `KNOWN_FAILURES`, and write one `diagnose(log)` that returns a structured
`Diagnosis` — the same shape whether the problem is auth, capacity, or a bad cache.
You'll point it at all three broken logs in the sections that follow.

### Your task — `diagnose`

`KNOWN_FAILURES` (given) is a list of rules, each with a list of `signatures`
(substrings that, if any appears in the log, identify that failure) plus its
`root_cause` and `fix`. Implement the matcher.

**Step by step:**

1. For each `rule` in `KNOWN_FAILURES`, check whether **any** of its `signatures`
   appears in `log`.
2. On the first match, grab the **evidence**: the first log line that contains the
   matching signature (`_first_line_with` is provided).
3. Return a `Diagnosis(failure=rule["failure"], root_cause=..., fix=..., evidence=...)`.
4. If nothing matches, return a `Diagnosis` with `failure="unknown"`.""")

code(r'''KNOWN_FAILURES = [
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


# Sanity check: the healthy log is not a failure.
print("startup_healthy ->", diagnose(LOGS["startup_healthy"]).failure)''',
r'''KNOWN_FAILURES = [
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
    # TODO: for each rule, if any of rule["signatures"] is in `log`, return a
    # Diagnosis carrying that rule's failure/root_cause/fix plus the evidence line
    # (_first_line_with(log, <the matching signature>)). If nothing matches, return
    # a Diagnosis with failure="unknown".
    raise NotImplementedError("Complete diagnose before continuing")


# Sanity check: the healthy log is not a failure.
print("startup_healthy ->", diagnose(LOGS["startup_healthy"]).failure)''')

md("""**Expected output:**
```
startup_healthy -> unknown
```""")

# ── Failure 1: NGC auth ──────────────────────────────────────────────────────
md("""### ⚠️ Failure 1 — the pod that won't pull (`ImagePullBackOff`)

The chart deploys, but the pod is stuck `0/1`. This is the most common first-day NIM
failure and it has nothing to do with the model: the node can't even pull the image
from `nvcr.io`. Run the diagnosis on the `kubectl describe` output.""")

code(r'''dx = diagnose(LOGS["ngc_auth_failure"])
rprint(dx)''')

md("""**Expected output:**
```
Diagnosis(failure='ngc_auth',
    root_cause='The image pull secret (ngc-secret) is missing or holds an invalid NGC key, ...',
    fix="Recreate the dockerconfigjson secret with username '$oauthtoken' ...",
    evidence='Warning  Failed  ... pulling from host nvcr.io failed with status code [manifests 1.3.0]: 401 Unauthorized')
```
**Root cause.** The `401 Unauthorized` from `nvcr.io` is decisive: the kubelet has no
valid credential for the NIM registry. The pull secret is missing, references the
wrong name, or was built with a stale NGC key. **Fix:** recreate the
`docker-registry` secret with username `$oauthtoken` and a current NGC key, and make
sure the pod's `imagePullSecrets` points at it. (This is the `ngc-secret` the Pulumi
`NimCluster` component wires up — verify it exists in the right namespace.)""")

# ── Failure 2: KV cache exhaustion ───────────────────────────────────────────
md("""### ⚠️ Failure 2 — healthy until it isn't (KV-cache exhaustion)

This NIM started cleanly and served the demo. Under the customer's real concurrency it
began returning **HTTP 503**. The log shows KV-cache utilization pinned at 1.00 and
requests rejected with *"0 free KV cache blocks"*. First diagnose it, then prove *why*
with the capacity math.

### Concept — the KV cache caps concurrency

Each in-flight request holds a slice of the **KV cache** for every token of its
context. The cache is a fixed token budget set at startup (the
`kv_cache_tokens=117440` you parsed in §1). If each request can use up to
`max_model_len` tokens, the server can hold at most
`kv_cache_tokens // max_model_len` requests at full context — beyond that, new
requests are rejected. `max_num_seqs=48` was *configured*, but the real ceiling is
set by the token budget, not that number.

### Your task — `max_concurrent_requests`

Return how many full-context requests fit in the KV cache.

**Step by step:** integer-divide the cache token budget by the per-request token
reservation: `kv_cache_tokens // tokens_per_request`.""")

code(r'''def max_concurrent_requests(kv_cache_tokens: int, tokens_per_request: int) -> int:
    """How many full-context requests fit in the KV cache at once."""
    return kv_cache_tokens // tokens_per_request


dx = diagnose(LOGS["kv_cache_exhaustion"])
print("diagnosis:", dx.failure)
print("evidence :", dx.evidence)

OFFERED_CONCURRENCY = 64  # the load ramp in the log
ceiling = max_concurrent_requests(report.kv_cache_tokens, MAX_MODEL_LEN)
print(f"\nKV budget {report.kv_cache_tokens} tokens / {MAX_MODEL_LEN} per req "
      f"-> {ceiling} concurrent requests at full context")
print(f"offered load = {OFFERED_CONCURRENCY} concurrent -> "
      f"{OFFERED_CONCURRENCY - ceiling} requests over the ceiling => 503s")''',
r'''def max_concurrent_requests(kv_cache_tokens: int, tokens_per_request: int) -> int:
    """How many full-context requests fit in the KV cache at once."""
    # TODO: integer-divide the KV cache token budget by the per-request reservation.
    raise NotImplementedError("Complete max_concurrent_requests before continuing")


dx = diagnose(LOGS["kv_cache_exhaustion"])
print("diagnosis:", dx.failure)
print("evidence :", dx.evidence)

OFFERED_CONCURRENCY = 64  # the load ramp in the log
ceiling = max_concurrent_requests(report.kv_cache_tokens, MAX_MODEL_LEN)
print(f"\nKV budget {report.kv_cache_tokens} tokens / {MAX_MODEL_LEN} per req "
      f"-> {ceiling} concurrent requests at full context")
print(f"offered load = {OFFERED_CONCURRENCY} concurrent -> "
      f"{OFFERED_CONCURRENCY - ceiling} requests over the ceiling => 503s")''')

md("""**Expected output:**
```
diagnosis: kv_cache_exhaustion
evidence : ERROR ... Request req-3f2a91 rejected: needs 4096 tokens but 0 free KV cache blocks (max_num_seqs=48 in flight, max_model_len=4096)
KV budget 117440 tokens / 4096 per req -> 28 concurrent requests at full context
offered load = 64 concurrent -> 36 requests over the ceiling => 503s
```
**Root cause.** The server can hold **28** full-context requests; the customer drove
**64**. `max_num_seqs=48` is a red herring — the token budget binds first. **Fixes,
cheapest first:** cap client concurrency (or add a queue with backpressure); lower
`max_model_len` if requests don't need 4096 tokens (smaller reservation → more
sequences); raise `gpu_memory_utilization`; or add a GPU and run `tp=2` for a bigger
cache. This capacity-vs-load tradeoff is exactly what Lab 04 models in depth.""")

# ── Failure 3: air-gapped cache ──────────────────────────────────────────────
md("""### ⚠️ Failure 3 — the air-gapped cache that loads the wrong engine

Go-live day. The cache was prepared on the team's H100 dev box, burned to media, and
copied to the customer's offline A100 node. The NIM detects the GPU, selects a
profile — then **aborts**: the engine it needs isn't in the cache. Diagnose it, then
pinpoint exactly which files are missing.

### Concept — offline mode can't fall back to NGC

In `NIM_OFFLINE=1` mode the server must materialize its engine from the local cache;
there is no download fallback. The trap from §2: the H100 dev box resolves to the
**fp8** profile, but the A100 target resolves to a **different fp16 profile** — so
the cache prepared on the dev box is missing the target's engine artifacts entirely.

### Your task — `missing_artifacts`

Compute which of the target profile's required artifacts are absent from the cache.

**Step by step:** return the sorted **set difference** `required − present` —
artifacts the profile needs that the cache doesn't have.""")

code(r'''def missing_artifacts(required: list[str], present: list[str]) -> list[str]:
    """Artifacts the selected profile needs that aren't in the offline cache."""
    return sorted(set(required) - set(present))


dx = diagnose(LOGS["airgapped_offline"])
print("diagnosis:", dx.failure)

target = select_profile(DEPLOYMENT["airgapped_target_gpu"], PROFILES)
required = next(p["artifacts"] for p in PROFILES if p["id"] == target.profile_id)
gap = missing_artifacts(required, CACHE["present_artifacts"])

print(f"cache prepared on : {CACHE['prepared_on']}")
print(f"target needs profile {target.profile_id} ({target.precision} tp{target.tp})")
print(f"missing artifacts : {gap}")''',
r'''def missing_artifacts(required: list[str], present: list[str]) -> list[str]:
    """Artifacts the selected profile needs that aren't in the offline cache."""
    # TODO: return the sorted set difference of required minus present.
    raise NotImplementedError("Complete missing_artifacts before continuing")


dx = diagnose(LOGS["airgapped_offline"])
print("diagnosis:", dx.failure)

target = select_profile(DEPLOYMENT["airgapped_target_gpu"], PROFILES)
required = next(p["artifacts"] for p in PROFILES if p["id"] == target.profile_id)
gap = missing_artifacts(required, CACHE["present_artifacts"])

print(f"cache prepared on : {CACHE['prepared_on']}")
print(f"target needs profile {target.profile_id} ({target.precision} tp{target.tp})")
print(f"missing artifacts : {gap}")''')

md("""**Expected output:**
```
diagnosis: airgap_cache_miss
cache prepared on : NVIDIA H100 80GB HBM3 (dev workstation)
target needs profile 6f1ac2d40b77 (fp16 tp1)
missing artifacts : ['models/6f1ac2d40b77/config.json', 'models/6f1ac2d40b77/rank0.engine']
```
**Root cause.** The cache holds the **H100/fp8** engine; the A100 target resolves to
the **fp16** profile `6f1ac2d40b77`, whose engine files were never on the media.
**Fix:** on a connected host run `nim download-to-cache --profile 6f1ac2d40b77` (the
*target's* profile, not the dev box's), re-copy the cache, and re-deploy. Always prep
the offline cache against the **customer's** GPU SKU — or pin `NIM_MODEL_PROFILE` so
selection can't surprise you. This is the line item that ships the federal deal.""")

# ── Challenge ────────────────────────────────────────────────────────────────
md("""## Challenge

1. **A new SKU walks in.** The customer swaps in **2× L40S** at the last minute. Call
   `select_profile` for it. Which profile wins, and is fp8 on the table? What would
   you have had to pre-stage in the cache?
2. **Right-size the concurrency cap.** Using `max_concurrent_requests`, find the
   largest `max_model_len` that still lets the node serve 50 concurrent requests from
   the same 117,440-token budget. Is shrinking context or adding a GPU the better
   lever for this customer?
3. **Extend the playbook.** Add a fourth rule to `KNOWN_FAILURES` for a CUDA
   out-of-memory crash on engine load (signature: `CUDA out of memory`). Write a
   one-line log sample, confirm `diagnose` classifies it, and state the fix.""")

# ── Key takeaways ────────────────────────────────────────────────────────────
md("""## Key takeaways

- **Read the startup log top to bottom.** GPU → profile selection → KV-cache
  allocation → `Uvicorn running`. Time-to-ready is the gap from the first line to
  *ready*, and the selected profile decides your throughput.
- **Profile auto-selection is deterministic.** GPU family, tp ≤ GPU count, and
  precision support (fp8 needs cc ≥ 8.9) filter the manifest; the most-optimized
  survivor wins. Override with `NIM_MODEL_PROFILE` when you must.
- **The KV cache caps concurrency.** Full-context capacity is
  `kv_cache_tokens // max_model_len`. Exceed it and you get HTTP 503, no matter what
  `max_num_seqs` says.
- **Air-gapped caches are GPU-specific.** Prep `download-to-cache` against the
  *target* SKU's profile; a cache built on a different GPU is missing the engine the
  target needs.

**References**
- NIM deployment (Helm): https://docs.nvidia.com/nim/large-language-models/latest/
- NIM model profiles & `download-to-cache`: https://docs.nvidia.com/nim/large-language-models/latest/utilities.html
- Air-gapped / offline cache: https://docs.nvidia.com/nim/large-language-models/latest/deployment-guide.html
- Triton Inference Server: https://docs.nvidia.com/deeplearning/triton-inference-server/""")

# ── build ─────────────────────────────────────────────────────────────────────
def make(use_stub: bool):
    nb = new_notebook()
    cells = []
    for kind, payload in CELLS:
        if kind == "md":
            cells.append(new_markdown_cell(payload))
        else:
            sol, stub = payload
            cells.append(new_code_cell(stub if (use_stub and stub) else sol))
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


LAB.parent.mkdir(parents=True, exist_ok=True)
SOL.parent.mkdir(parents=True, exist_ok=True)
nbf.write(make(use_stub=True), str(LAB))
nbf.write(make(use_stub=False), str(SOL))
print(f"wrote {LAB}")
print(f"wrote {SOL}")
