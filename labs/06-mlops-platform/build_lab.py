"""Author the Lab 06 notebooks from a single source.

Emits two notebooks that stay in sync:
  - labs/06-mlops-platform/lab.ipynb        — learner copy (exercise cells stubbed)
  - solutions/06-mlops-platform/lab.ipynb   — completed, nbmake-clean copy

Same committed-builder pattern as Labs 01–05: edit THIS file (never the .ipynb), then:

    uv run python labs/06-mlops-platform/build_lab.py

Lab 06 is analytical: no GPU, no NIM calls. Every cell parses or computes over the
sample scrape + time series in data/, so the solution notebook runs anywhere offline.
"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/06-mlops-platform/lab.ipynb"
SOL = ROOT / "solutions/06-mlops-platform/lab.ipynb"

CELLS = []
def md(s): CELLS.append(("md", s))
def code(sol, stub=None): CELLS.append(("code", (sol, stub)))

# ── Scenario ─────────────────────────────────────────────────────────────────
md("""# Lab 06 — MLOps & Platform

## Scenario

An MLOps team runs a NIM cluster under production load. At the Kubernetes layer
everything is green — pods `Running`, no restarts — right up until p99 latency
explodes and support lights up. The signal was there 2 minutes earlier, in metrics
nobody had on a board: the **KV cache** saturated, the request queue backed up, and
*then* time-to-first-token blew the SLO. Separately, a slower problem is brewing —
nightly **RAGAS** faithfulness has been drifting down as the corpus changes, and no
one has wired that to a retraining trigger.

This capstone is the observability and quality loop that catches both. You'll parse a
real Triton `:8002/metrics` scrape (plus DCGM GPU metrics), turn counters into rates
and latencies, compute tail percentiles, detect the **SLO-breach window**, and — the
payoff — measure the **alert lead time** a leading indicator buys you. Then you'll
close the quality loop: a RAGAS **regression gate** that fires a NeMo Customizer
retrain when faithfulness drops. Fully offline — it replays `data/`, produced by
`data/generate.py`; with live infra, point it at the real `PROMETHEUS_URL`.""")

# ── Setup ────────────────────────────────────────────────────────────────────
md("""## Setup

Loads the raw Prometheus scrape (`metrics_raw.txt`), the 30-minute time series
(`metrics.json`, 61 samples at 30s spacing with the SLOs attached), and the nightly
RAGAS runs (`eval_runs.json`).""")

code(r'''import json
import re
import sys
from pathlib import Path

from rich import print as rprint
from rich.table import Table
from rich.console import Console

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "shared").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.utils import display_metrics_table

DATA = REPO_ROOT / "labs/06-mlops-platform/data"
RAW_SCRAPE = (DATA / "metrics_raw.txt").read_text()
METRICS = json.loads((DATA / "metrics.json").read_text())
SAMPLES = METRICS["samples"]
SLO = METRICS["slo"]
STEP_S = METRICS["step_s"]
EVAL = json.loads((DATA / "eval_runs.json").read_text())

_console = Console()
SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values) -> str:
    """Tiny inline chart so a metric's shape is visible without a plotting lib."""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    return "".join(SPARK[min(len(SPARK) - 1, int((v - lo) / span * (len(SPARK) - 1)))] for v in values)


# Live-infra swap (Option B): instead of the sample file, scrape the real endpoint —
#   import httpx
#   RAW_SCRAPE = httpx.get(f"{os.environ['PROMETHEUS_URL']}/api/v1/.../metrics").text
#   (or query Prometheus directly with PromQL: rate(request_success_total[1m]))

print(f"scrape: {len(RAW_SCRAPE.splitlines())} lines | series: {len(SAMPLES)} samples "
      f"@ {STEP_S}s | SLOs: {SLO}")''')

md("""**Expected output:**
```
scrape: 39 lines | series: 61 samples @ 30s | SLOs: {'ttft_p99_ms': 500, 'itl_p99_ms': 50, 'kv_warn': 0.9}
```""")

# ── 1. Parse the scrape ──────────────────────────────────────────────────────
md("""## 1 · Read the `:8002/metrics` endpoint

### Concept

Triton (and the NIM around it) and the DCGM Exporter publish metrics in the
**Prometheus exposition format**: `# HELP`/`# TYPE` comment lines, then one sample per
line as `metric_name{label="v",...} value`. Prometheus scrapes this text on an
interval; before you build a dashboard you need to turn one scrape into numbers. The
metrics that matter for a NIM:

- `request_success_total` / `request_failure_total` — **counters** (cumulative).
- `gpu_cache_usage_perc`, `num_requests_running`, `num_requests_waiting` — Triton/NIM
  **gauges** (the KV-cache and queue state from Lab 03).
- `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_PROF_SM_ACTIVE`, `DCGM_FI_DEV_FB_USED` — GPU-level
  gauges from DCGM (the roofline signals from Lab 04).

### Your task — `parse_prometheus`

Parse the exposition text into a `{metric_name: float}` dict (one series per name in
this scrape; drop the labels).

**Step by step:**

1. For each line: strip it; skip blanks and lines starting with `#`.
2. Match `name`, an optional `{...}` label block, and the trailing `value` — e.g. with
   `re.match(r"^([a-zA-Z_:][\\w:]*)(\\{[^}]*\\})?\\s+([-\\d.eE+]+)$", line)`.
3. Store `out[name] = float(value)`.""")

code(r'''def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition text into {metric_name: value} (labels dropped)."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(3))
    return out


scrape = parse_prometheus(RAW_SCRAPE)
print(f"parsed {len(scrape)} metrics from the scrape\n")
display_metrics_table({
    "KV cache utilization": scrape["gpu_cache_usage_perc"],
    "requests waiting (queue)": scrape["num_requests_waiting"],
    "TTFT p99 (s)": scrape["time_to_first_token_seconds_p99"],
    "GPU utilization (%)": scrape["DCGM_FI_DEV_GPU_UTIL"],
    "SM active (frac)": scrape["DCGM_FI_PROF_SM_ACTIVE"],
}, title="Single scrape at the incident peak (t=1200s)")''',
r'''def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition text into {metric_name: value} (labels dropped)."""
    out = {}
    # TODO: for each line, skip blanks and '#' comments; match name + optional {labels}
    # + trailing value with the regex in the walkthrough; store out[name] = float(value).
    raise NotImplementedError("Complete parse_prometheus before continuing")


scrape = parse_prometheus(RAW_SCRAPE)
print(f"parsed {len(scrape)} metrics from the scrape\n")
display_metrics_table({
    "KV cache utilization": scrape["gpu_cache_usage_perc"],
    "requests waiting (queue)": scrape["num_requests_waiting"],
    "TTFT p99 (s)": scrape["time_to_first_token_seconds_p99"],
    "GPU utilization (%)": scrape["DCGM_FI_DEV_GPU_UTIL"],
    "SM active (frac)": scrape["DCGM_FI_PROF_SM_ACTIVE"],
}, title="Single scrape at the incident peak (t=1200s)")''')

md("""**Expected output** (the peak scrape is already alarming — KV cache pinned, 41
requests queued, TTFT p99 at 880 ms):
```
parsed 13 metrics from the scrape

         Single scrape at the incident peak (t=1200s)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric                     ┃    Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ KV cache utilization       │   0.9900 │
│ requests waiting (queue)   │  41.0000 │
│ TTFT p99 (s)               │   0.8800 │
│ GPU utilization (%)        │  99.0000 │
│ SM active (frac)           │   0.9300 │
└────────────────────────────┴──────────┘
```
A single scrape is a snapshot. To see the *incident*, you need the rate of change over
time — next.""")

# ── 2. Counters -> rates ─────────────────────────────────────────────────────
md("""## 2 · Counters → rates (what PromQL `rate()` does)

### Concept

Counters only ever go up; their **value** is meaningless, their **slope** is
everything. `request_success_total` reading 184,223 tells you nothing — but
*increasing by 1,635 over 30 seconds* tells you you're serving ~55 req/s. That slope
is what PromQL's `rate()` computes, and it's how you recover requests/sec and
tokens/sec from raw counters.

### Your task — `counter_rate`

Return the per-interval rate of a cumulative counter across the series: for each
consecutive pair of samples, `(curr - prev) / step_seconds`.

**Step by step:** for `i` in `1..len(samples)-1`, append
`(samples[i][key] - samples[i-1][key]) / step_s`; return the list (length n-1).""")

code(r'''def counter_rate(samples: list[dict], key: str, step_s: int) -> list[float]:
    """Per-interval rate of a cumulative counter (the PromQL rate() idea)."""
    return [(samples[i][key] - samples[i - 1][key]) / step_s for i in range(1, len(samples))]


rps = counter_rate(SAMPLES, "request_success_total", STEP_S)
tok = counter_rate(SAMPLES, "generation_tokens_total", STEP_S)
fail = counter_rate(SAMPLES, "request_failure_total", STEP_S)

print(f"successful RPS : baseline {rps[0]:.0f}  ->  peak {max(rps):.0f}")
print(f"output tokens/s: baseline {tok[0]:.0f}  ->  peak {max(tok):.0f}")
print(f"failures/s (503s on a full KV cache): peak {max(fail):.1f}\n")
print("RPS over 30 min:        ", sparkline(rps))
print("KV cache util over 30m: ", sparkline([s["kv_cache_util"] for s in SAMPLES]))
print("TTFT p99 over 30 min:   ", sparkline([s["ttft_p99_ms"] for s in SAMPLES]))''',
r'''def counter_rate(samples: list[dict], key: str, step_s: int) -> list[float]:
    """Per-interval rate of a cumulative counter (the PromQL rate() idea)."""
    # TODO: return [(samples[i][key] - samples[i-1][key]) / step_s for i in range(1, len(samples))]
    raise NotImplementedError("Complete counter_rate before continuing")


rps = counter_rate(SAMPLES, "request_success_total", STEP_S)
tok = counter_rate(SAMPLES, "generation_tokens_total", STEP_S)
fail = counter_rate(SAMPLES, "request_failure_total", STEP_S)

print(f"successful RPS : baseline {rps[0]:.0f}  ->  peak {max(rps):.0f}")
print(f"output tokens/s: baseline {tok[0]:.0f}  ->  peak {max(tok):.0f}")
print(f"failures/s (503s on a full KV cache): peak {max(fail):.1f}\n")
print("RPS over 30 min:        ", sparkline(rps))
print("KV cache util over 30m: ", sparkline([s["kv_cache_util"] for s in SAMPLES]))
print("TTFT p99 over 30 min:   ", sparkline([s["ttft_p99_ms"] for s in SAMPLES]))''')

md("""**Expected output** (the three sparklines tell the whole story — load rises, KV
cache saturates, then TTFT spikes):
```
successful RPS : baseline 20  ->  peak 55
output tokens/s: baseline 4800  ->  peak 13085
failures/s (503s on a full KV cache): peak 3.5

RPS over 30 min:         ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▃▄▅▆██▇█▇██▇█▇██▇▇▆▅▄▂▁▁▁▁▁▁▁
KV cache util over 30m:  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅▇▇▇▇█████████████▆▅▄▃▂▁▁▁▁▁▁▁
TTFT p99 over 30 min:    ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▃▃▄▇▇▇▇█▇▇▇▇▇▇▆▆▅▃▂▂▁▁▁▁▁▁▁▁
```
Notice the **KV-cache curve rises before the TTFT curve** — that lead is the alert you
want (§5). The 503 failures appear only while the cache is pinned, exactly the
exhaustion you diagnosed in Lab 03.""")

# ── 3. Percentiles ───────────────────────────────────────────────────────────
md("""## 3 · Tail latency — percentiles

### Concept

Averages hide incidents; **percentiles** expose them. p50 is the typical request, but
your SLA lives in the tail — p95, p99 — where the angry users are. Over this window,
summarizing the distribution of the per-scrape `ttft_p99` readings tells you both the
healthy baseline *and* how bad the worst sustained latency got.

### Your task — `percentile`

Implement a linear-interpolated percentile over a list of values.

**Step by step:**

1. `v = sorted(values)`; `k = (len(v) - 1) * p` (p in [0,1]).
2. `f = floor(k)`, `c = min(f + 1, len(v) - 1)`.
3. Interpolate: `v[f] + (v[c] - v[f]) * (k - f)`.""")

code(r'''def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated p-th percentile (p in [0, 1])."""
    v = sorted(values)
    k = (len(v) - 1) * p
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


ttft99 = [s["ttft_p99_ms"] for s in SAMPLES]
display_metrics_table({
    "TTFT p99 — median reading (p50)": percentile(ttft99, 0.50),
    "TTFT p99 — p95 of readings": percentile(ttft99, 0.95),
    "TTFT p99 — worst (p99 of readings)": percentile(ttft99, 0.99),
    "SLO": float(SLO["ttft_p99_ms"]),
}, title="Distribution of TTFT p99 across the 30-min window")''',
r'''def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated p-th percentile (p in [0, 1])."""
    # TODO: v = sorted(values); k = (len(v)-1)*p; f = int(k); c = min(f+1, len(v)-1);
    # return v[f] + (v[c]-v[f])*(k-f).
    raise NotImplementedError("Complete percentile before continuing")


ttft99 = [s["ttft_p99_ms"] for s in SAMPLES]
display_metrics_table({
    "TTFT p99 — median reading (p50)": percentile(ttft99, 0.50),
    "TTFT p99 — p95 of readings": percentile(ttft99, 0.95),
    "TTFT p99 — worst (p99 of readings)": percentile(ttft99, 0.99),
    "SLO": float(SLO["ttft_p99_ms"]),
}, title="Distribution of TTFT p99 across the 30-min window")''')

md("""**Expected output** (most of the window is healthy at ~180 ms, but the tail of
the distribution is ~870 ms — well past the 500 ms SLO):
```
       Distribution of TTFT p99 across the 30-min window
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric                                ┃    Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ TTFT p99 — median reading (p50)        │ 180.0000 │
│ TTFT p99 — p95 of readings             │ 850.0000 │
│ TTFT p99 — worst (p99 of readings)     │ 871.0000 │
│ SLO                                    │ 500.0000 │
└───────────────────────────────────────┴──────────┘
```
The median reading is fine, which is exactly why averaging would have missed this. The
tail is where the SLO breaks — so the tail is what you alert on.""")

# ── 4. SLO breach window ─────────────────────────────────────────────────────
md("""## 4 · Find the breach window

### Concept

An alert isn't "p99 is high right now" — it's "for how long, and starting when". You
want the contiguous **window** during which a metric violated its SLO, so the incident
timeline is unambiguous in the postmortem.

### Your task — `slo_breach_window`

Return `(start_t, end_t)` — the first and last timestamps where `samples[i][key]`
exceeds `threshold` — or `None` if the SLO held.

**Step by step:**

1. `breaching = [s["t_s"] for s in samples if s[key] > threshold]`.
2. Return `None` if it's empty, else `(min(breaching), max(breaching))`.""")

code(r'''def slo_breach_window(samples: list[dict], key: str, threshold: float) -> tuple[int, int] | None:
    """First and last timestamps where `key` exceeds `threshold` (None if never)."""
    breaching = [s["t_s"] for s in samples if s[key] > threshold]
    if not breaching:
        return None
    return min(breaching), max(breaching)


window = slo_breach_window(SAMPLES, "ttft_p99_ms", SLO["ttft_p99_ms"])
start, end = window
print(f"TTFT p99 breached {SLO['ttft_p99_ms']} ms from t={start}s to t={end}s "
      f"({(end - start) // 60} min {(end - start) % 60}s)")
print(f"peak TTFT p99 in window: {max(s['ttft_p99_ms'] for s in SAMPLES)} ms")
print(f"requests dropped (503s) during window: "
      f"{SAMPLES[-1]['request_failure_total'] - SAMPLES[0]['request_failure_total']}")''',
r'''def slo_breach_window(samples: list[dict], key: str, threshold: float) -> tuple[int, int] | None:
    """First and last timestamps where `key` exceeds `threshold` (None if never)."""
    # TODO: collect t_s where samples[i][key] > threshold; return None if empty else
    # (min, max) of those timestamps.
    raise NotImplementedError("Complete slo_breach_window before continuing")


window = slo_breach_window(SAMPLES, "ttft_p99_ms", SLO["ttft_p99_ms"])
start, end = window
print(f"TTFT p99 breached {SLO['ttft_p99_ms']} ms from t={start}s to t={end}s "
      f"({(end - start) // 60} min {(end - start) % 60}s)")
print(f"peak TTFT p99 in window: {max(s['ttft_p99_ms'] for s in SAMPLES)} ms")
print(f"requests dropped (503s) during window: "
      f"{SAMPLES[-1]['request_failure_total'] - SAMPLES[0]['request_failure_total']}")''')

md("""**Expected output:**
```
TTFT p99 breached 500 ms from t=1080s to t=1470s (6 min 30s)
peak TTFT p99 in window: 880 ms
requests dropped (503s) during window: 1532
```
A 6½-minute breach starting at minute 18, with ~1,500 requests turned away. Now the
question every postmortem asks: **could we have known sooner?**""")

# ── 5. Alert lead time (capstone) ────────────────────────────────────────────
md("""## 5 · Capstone — alert lead time from a leading indicator

### Concept

The latency breach is a **lagging** signal — by the time TTFT blows the SLO, users are
already hurting. A good alert fires on a **leading** indicator that *predicts* the
breach. Here the KV-cache utilization (and the queue behind it) saturates *before*
latency climbs — exactly the causal chain from Labs 03–04: cache fills → requests
queue → first-token latency rises. The **lead time** between the leading-indicator
warning and the SLO breach is how much runway your on-call gets to act (scale out,
shed load) before the SLA breaks.

### Your task — `alert_lead_time`

Find when the leading indicator first crosses its warning level and when the SLO first
breaches, and return the gap in seconds (and both timestamps).

**Step by step:**

1. `warn_t` = first `s["t_s"]` where `s[lead_key] >= lead_warn`.
2. `breach_t` = first `s["t_s"]` where `s[slo_key] > slo_threshold`.
3. Return `(warn_t, breach_t, breach_t - warn_t)`.""")

code(r'''def alert_lead_time(samples, lead_key, lead_warn, slo_key, slo_threshold):
    """Seconds of warning a leading indicator gives before the SLO breaches."""
    warn_t = next(s["t_s"] for s in samples if s[lead_key] >= lead_warn)
    breach_t = next(s["t_s"] for s in samples if s[slo_key] > slo_threshold)
    return warn_t, breach_t, breach_t - warn_t


warn_t, breach_t, lead = alert_lead_time(
    SAMPLES, "kv_cache_util", SLO["kv_warn"], "ttft_p99_ms", SLO["ttft_p99_ms"])
print(f"KV-cache warning (>= {SLO['kv_warn']}) first fired at t={warn_t}s")
print(f"TTFT p99 SLO breached at                t={breach_t}s")
print(f"=> alert lead time: {lead}s ({lead // 60} min) of runway before the SLA broke\n")
print("Recommended alert: page when kv_cache_util >= 0.90 for 1m — it predicts the "
      "TTFT breach with ~2 min of lead, vs alerting on TTFT itself (already too late).")''',
r'''def alert_lead_time(samples, lead_key, lead_warn, slo_key, slo_threshold):
    """Seconds of warning a leading indicator gives before the SLO breaches."""
    # TODO: warn_t = first t_s where samples[i][lead_key] >= lead_warn;
    #       breach_t = first t_s where samples[i][slo_key] > slo_threshold;
    #       return (warn_t, breach_t, breach_t - warn_t).
    raise NotImplementedError("Complete alert_lead_time before continuing")


warn_t, breach_t, lead = alert_lead_time(
    SAMPLES, "kv_cache_util", SLO["kv_warn"], "ttft_p99_ms", SLO["ttft_p99_ms"])
print(f"KV-cache warning (>= {SLO['kv_warn']}) first fired at t={warn_t}s")
print(f"TTFT p99 SLO breached at                t={breach_t}s")
print(f"=> alert lead time: {lead}s ({lead // 60} min) of runway before the SLA broke\n")
print("Recommended alert: page when kv_cache_util >= 0.90 for 1m — it predicts the "
      "TTFT breach with ~2 min of lead, vs alerting on TTFT itself (already too late).")''')

md("""**Expected output:**
```
KV-cache warning (>= 0.9) first fired at t=960s
TTFT p99 SLO breached at                t=1080s
=> alert lead time: 120s (2 min) of runway before the SLA broke

Recommended alert: page when kv_cache_util >= 0.90 for 1m — it predicts the TTFT breach with ~2 min of lead, vs alerting on TTFT itself (already too late).
```
**120 seconds of warning** — enough to autoscale or shed load before users notice.
This is the difference between a dashboard that *describes* an outage and one that
*prevents* it. Wire `kv_cache_util` and `num_requests_waiting` onto the Grafana board
beside latency, and alert on the leading one.""")

# ── 6. Close the loop: RAGAS regression gate ─────────────────────────────────
md("""## 6 · Close the loop — RAGAS regression gate → NeMo Customizer

### Concept

Latency isn't the only thing that regresses; **answer quality** drifts too. A
continuous-eval job runs RAGAS nightly against a golden set and tracks
**faithfulness** (is the answer grounded in the retrieved context?). When it crosses a
gate, that's the signal to **retrain/customize** — in the NVIDIA stack, kicking off a
**NeMo Customizer** job on a curated set. This turns evaluation from a dashboard you
glance at into a closed control loop.

### Your task — `regression_gate`

Return the first nightly run whose `faithfulness` falls below the gate (or `None` if
all passed).

**Step by step:** iterate `runs` in order; return the first whose
`run["faithfulness"] < gate`, else `None`.""")

code(r'''def regression_gate(runs: list[dict], gate: float) -> dict | None:
    """First eval run that drops below the faithfulness gate (triggers a retrain)."""
    return next((r for r in runs if r["faithfulness"] < gate), None)


gate = EVAL["faithfulness_gate"]
runs = EVAL["runs"]
print("faithfulness over nightly runs:", sparkline([r["faithfulness"] for r in runs]))
failing = regression_gate(runs, gate)
if failing:
    print(f"\n⚠️  gate {gate}: run #{failing['run']} ({failing['date']}) faithfulness "
          f"{failing['faithfulness']} < {gate}")
    print(f"=> trigger NeMo Customizer retrain on the curated set "
          f"(run drifted {runs[0]['faithfulness'] - failing['faithfulness']:.2f} from baseline "
          f"{runs[0]['faithfulness']})")
else:
    print(f"\nall runs >= {gate} — no retrain needed")''',
r'''def regression_gate(runs: list[dict], gate: float) -> dict | None:
    """First eval run that drops below the faithfulness gate (triggers a retrain)."""
    # TODO: return the first run whose run["faithfulness"] < gate, else None
    # (hint: next((r for r in runs if ...), None)).
    raise NotImplementedError("Complete regression_gate before continuing")


gate = EVAL["faithfulness_gate"]
runs = EVAL["runs"]
print("faithfulness over nightly runs:", sparkline([r["faithfulness"] for r in runs]))
failing = regression_gate(runs, gate)
if failing:
    print(f"\n⚠️  gate {gate}: run #{failing['run']} ({failing['date']}) faithfulness "
          f"{failing['faithfulness']} < {gate}")
    print(f"=> trigger NeMo Customizer retrain on the curated set "
          f"(run drifted {runs[0]['faithfulness'] - failing['faithfulness']:.2f} from baseline "
          f"{runs[0]['faithfulness']})")
else:
    print(f"\nall runs >= {gate} — no retrain needed")''')

md("""**Expected output:**
```
faithfulness over nightly runs: █▇█▇▆▆▅▄▃▂▁▂▁▁

⚠️  gate 0.85: run #10 (2026-05-10) faithfulness 0.83 < 0.85
=> trigger NeMo Customizer retrain on the curated set (run drifted 0.11 from baseline 0.94)
```
Faithfulness slid from 0.94 to below the 0.85 gate by night 10 — caught automatically,
not by a user complaint. The gate is the trigger that closes the MLOps loop:
**observe → evaluate → retrain → redeploy**.""")

# ── Challenge ────────────────────────────────────────────────────────────────
md("""## Challenge

1. **Multi-SLO board.** Add `itl_p99_ms` (SLO 50 ms) to the breach analysis. Does the
   inter-token-latency SLO break *before* or *after* TTFT, and what does that ordering
   say about where the bottleneck is (prefill queue vs decode)?
2. **Burn-rate alert.** Compute an error-budget burn rate: over a rolling 5-sample
   window, what fraction of time was TTFT in breach? Page when the budget burns faster
   than, say, 10% per window. How does its lead time compare to the KV-cache alert?
3. **Cost of the incident.** Using the 503 count and the failures sparkline, estimate
   requests lost. If autoscaling had triggered at the KV-cache warning (t=960) instead
   of the breach (t=1080), how many of those 626 failures might you have avoided?""")

# ── Key takeaways ────────────────────────────────────────────────────────────
md("""## Key takeaways

- **Counters need rates.** A counter's value is noise; its slope (`rate()`) is the
  signal. Recover RPS and tokens/sec from `*_total` counters.
- **Alert on the tail, page on the leader.** p50 hides incidents — watch p95/p99. But
  latency is *lagging*; the **leading** indicator (KV-cache saturation, queue depth)
  bought 2 minutes of runway here.
- **One board, three layers.** Triton (KV cache, queue), DCGM (GPU/SM util, memory),
  and RAGAS (quality) each show one slice; the incident is only legible with all three
  — and they trace straight back to Labs 03 (KV exhaustion) and 04 (the roofline).
- **Close the loop.** A RAGAS regression gate turns evaluation into a trigger:
  observe → evaluate → **NeMo Customizer retrain** → redeploy.

**References**
- Triton metrics: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html
- DCGM Exporter: https://github.com/NVIDIA/dcgm-exporter
- Prometheus exposition format: https://prometheus.io/docs/instrumenting/exposition_formats/
- RAGAS: https://docs.ragas.io/
- NeMo Customizer: https://docs.nvidia.com/nemo/customizer/
- kube-prometheus-stack + Grafana: https://github.com/prometheus-community/helm-charts""")

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
