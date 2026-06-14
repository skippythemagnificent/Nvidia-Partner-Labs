# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Generate the Lab 06 observability inputs: a raw Triton/DCGM scrape, a 30-minute
time series with a built-in incident, and a run of nightly RAGAS evaluations.

Run with:  uv run python labs/06-mlops-platform/data/generate.py
       or:  task lab:data:generate LAB=06-mlops-platform

  metrics_raw.txt   One scrape of a NIM's Triton `:8002/metrics` endpoint plus DCGM
                    Exporter, in Prometheus exposition format (HELP/TYPE comments +
                    `name{labels} value` samples). Captured at the incident peak, so
                    the numbers are alarming on purpose. The parsing exercise turns
                    this text into a metric dict.
  metrics.json      61 samples at 30s spacing over 30 minutes (t=0..1800s). Each holds
                    cumulative counters (request_success_total, request_failure_total,
                    generation_tokens_total) and instantaneous gauges (kv_cache_util,
                    gpu_util, sm_active, num_running, num_waiting, ttft/itl p50/p99).
                    A load spike from ~18 to ~24 minutes saturates the KV cache and
                    breaches the latency SLOs, then autoscaling recovers it.
  eval_runs.json    14 nightly RAGAS continuous-eval runs. Faithfulness drifts down as
                    the corpus changes and crosses the 0.85 gate — the signal that
                    closes the loop back to NeMo Customizer retraining.

The incident is engineered so a *leading* indicator (KV-cache saturation) crosses its
warning line before the latency SLO breaches — that gap is the alert lead time the
capstone computes. All values are deterministic (anchor interpolation, no RNG).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

MODEL = "llama-3.1-8b-instruct"
STEP_S = 30
DURATION_S = 1800

# SLOs the dashboard alerts on.
TTFT_P99_SLO_MS = 500
ITL_P99_SLO_MS = 50
KV_WARN = 0.90              # KV-cache utilization warning line (leading indicator)


def _interp(anchors: dict[int, float], t: int) -> float:
    """Piecewise-linear value at time t from sorted (t -> value) anchor points."""
    ts = sorted(anchors)
    if t <= ts[0]:
        return anchors[ts[0]]
    if t >= ts[-1]:
        return anchors[ts[-1]]
    for a, b in zip(ts, ts[1:]):
        if a <= t <= b:
            frac = (t - a) / (b - a)
            return anchors[a] + frac * (anchors[b] - anchors[a])
    return anchors[ts[-1]]


# Anchor curves. The load spike runs ~900s (15m) to ~1500s, peaking 1080–1200s.
RPS = {0: 20, 900: 20, 1080: 58, 1440: 58, 1620: 24, 1800: 24}
KV = {0: 0.55, 900: 0.55, 930: 0.78, 960: 0.92, 1020: 0.98, 1080: 0.99,
      1440: 0.99, 1500: 0.80, 1620: 0.55, 1800: 0.50}
TTFT99 = {0: 180, 900: 180, 930: 240, 960: 320, 990: 410, 1020: 470, 1050: 480,
          1080: 780, 1200: 880, 1440: 760, 1500: 420, 1620: 200, 1800: 185}
ITL99 = {0: 22, 900: 22, 1020: 40, 1080: 62, 1200: 70, 1440: 60, 1620: 26, 1800: 23}
WAIT = {0: 1, 900: 1, 960: 6, 1020: 14, 1080: 28, 1200: 41, 1440: 30, 1620: 3, 1800: 1}
GPU = {0: 0.60, 900: 0.62, 1020: 0.95, 1080: 0.99, 1200: 0.99, 1440: 0.97, 1620: 0.64, 1800: 0.58}


def _series() -> list[dict]:
    samples = []
    success_total = 120_000.0
    failure_total = 480.0
    gen_tokens_total = 38_500_000.0
    for t in range(0, DURATION_S + 1, STEP_S):
        rps = _interp(RPS, t)
        kv = round(_interp(KV, t), 3)
        ttft99 = round(_interp(TTFT99, t))
        itl99 = round(_interp(ITL99, t))
        waiting = round(_interp(WAIT, t))
        gpu = round(_interp(GPU, t), 3)
        # Failures appear only when the KV cache is effectively full (503s, as in Lab 03).
        fail_rps = rps * 0.06 if kv >= 0.97 else 0.0
        # Accumulate counters across this 30s interval.
        success_total += (rps - fail_rps) * STEP_S
        failure_total += fail_rps * STEP_S
        gen_tokens_total += (rps - fail_rps) * STEP_S * 240   # ~240 output tokens/req
        samples.append({
            "t_s": t,
            "request_success_total": round(success_total),
            "request_failure_total": round(failure_total),
            "generation_tokens_total": round(gen_tokens_total),
            "num_running": min(48, round(rps * 0.8)),
            "num_waiting": waiting,
            "kv_cache_util": kv,
            "gpu_util": gpu,
            "sm_active": round(gpu * 0.94, 3),
            "ttft_p50_ms": round(ttft99 * 0.45),
            "ttft_p99_ms": ttft99,
            "itl_p50_ms": round(itl99 * 0.6),
            "itl_p99_ms": itl99,
        })
    return samples


# One raw scrape at the incident peak (t=1200s) in Prometheus exposition format.
def _raw_scrape(peak: dict) -> str:
    g = "GPU-3a1f7c0e-1b2d-4e5a-9f00-2c4b6d8e0a11"
    return f"""\
# HELP request_success_total Successful inference requests (cumulative).
# TYPE request_success_total counter
request_success_total{{model="{MODEL}",version="1"}} {peak["request_success_total"]}
# HELP request_failure_total Failed inference requests (cumulative).
# TYPE request_failure_total counter
request_failure_total{{model="{MODEL}",version="1"}} {peak["request_failure_total"]}
# HELP generation_tokens_total Output tokens generated (cumulative).
# TYPE generation_tokens_total counter
generation_tokens_total{{model="{MODEL}"}} {peak["generation_tokens_total"]}
# HELP num_requests_running In-flight requests being decoded right now.
# TYPE num_requests_running gauge
num_requests_running{{model="{MODEL}"}} {peak["num_running"]}
# HELP num_requests_waiting Requests queued waiting for a KV-cache slot.
# TYPE num_requests_waiting gauge
num_requests_waiting{{model="{MODEL}"}} {peak["num_waiting"]}
# HELP gpu_cache_usage_perc Fraction of the KV cache in use (0-1).
# TYPE gpu_cache_usage_perc gauge
gpu_cache_usage_perc{{model="{MODEL}"}} {peak["kv_cache_util"]}
# HELP time_to_first_token_seconds_p99 TTFT 99th percentile (seconds).
# TYPE time_to_first_token_seconds_p99 gauge
time_to_first_token_seconds_p99{{model="{MODEL}"}} {peak["ttft_p99_ms"] / 1000:.3f}
# HELP inter_token_latency_seconds_p99 ITL 99th percentile (seconds).
# TYPE inter_token_latency_seconds_p99 gauge
inter_token_latency_seconds_p99{{model="{MODEL}"}} {peak["itl_p99_ms"] / 1000:.3f}
# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization (%).
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{{gpu="0",UUID="{g}"}} {round(peak["gpu_util"] * 100)}
# HELP DCGM_FI_PROF_SM_ACTIVE Fraction of SMs active (0-1).
# TYPE DCGM_FI_PROF_SM_ACTIVE gauge
DCGM_FI_PROF_SM_ACTIVE{{gpu="0",UUID="{g}"}} {peak["sm_active"]}
# HELP DCGM_FI_DEV_FB_USED Frame-buffer memory used (MiB).
# TYPE DCGM_FI_DEV_FB_USED gauge
DCGM_FI_DEV_FB_USED{{gpu="0",UUID="{g}"}} 75210
# HELP DCGM_FI_DEV_POWER_USAGE Power draw (W).
# TYPE DCGM_FI_DEV_POWER_USAGE gauge
DCGM_FI_DEV_POWER_USAGE{{gpu="0",UUID="{g}"}} 681.4
# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature (C).
# TYPE DCGM_FI_DEV_GPU_TEMP gauge
DCGM_FI_DEV_GPU_TEMP{{gpu="0",UUID="{g}"}} 78
"""


# Nightly RAGAS continuous-eval runs; faithfulness drifts below the 0.85 gate on night 9.
_FAITH = [0.94, 0.93, 0.94, 0.92, 0.91, 0.90, 0.88, 0.87, 0.86, 0.83, 0.82, 0.84, 0.81, 0.80]
EVAL_RUNS = [
    {
        "run": i + 1,
        "date": f"2026-05-{i + 1:02d}",
        "faithfulness": f,
        "answer_relevancy": round(0.90 - 0.005 * i, 3),
        "context_precision": round(0.88 - 0.004 * i, 3),
        "n_questions": 120,
    }
    for i, f in enumerate(_FAITH)
]
FAITHFULNESS_GATE = 0.85


def _validate(series: list[dict]) -> None:
    assert len(series) == 61, f"expected 61 samples, got {len(series)}"
    # counters must be monotonic non-decreasing
    for a, b in zip(series, series[1:]):
        assert b["request_success_total"] >= a["request_success_total"]
        assert b["request_failure_total"] >= a["request_failure_total"]
    # leading indicator must precede the SLO breach
    kv_warn_t = next(s["t_s"] for s in series if s["kv_cache_util"] >= KV_WARN)
    ttft_breach_t = next(s["t_s"] for s in series if s["ttft_p99_ms"] > TTFT_P99_SLO_MS)
    assert kv_warn_t < ttft_breach_t, "KV warning must lead the TTFT breach"
    assert ttft_breach_t - kv_warn_t == 120, f"expected 120s lead, got {ttft_breach_t - kv_warn_t}"
    assert sum(r["faithfulness"] < FAITHFULNESS_GATE for r in EVAL_RUNS) > 0


def main() -> None:
    series = _series()
    _validate(series)
    peak = next(s for s in series if s["t_s"] == 1200)

    (HERE / "metrics.json").write_text(json.dumps({
        "model": MODEL, "step_s": STEP_S,
        "slo": {"ttft_p99_ms": TTFT_P99_SLO_MS, "itl_p99_ms": ITL_P99_SLO_MS, "kv_warn": KV_WARN},
        "samples": series,
    }, indent=2) + "\n")
    (HERE / "metrics_raw.txt").write_text(_raw_scrape(peak))
    (HERE / "eval_runs.json").write_text(json.dumps({
        "faithfulness_gate": FAITHFULNESS_GATE, "runs": EVAL_RUNS,
    }, indent=2) + "\n")

    print(f"[ok] wrote {len(series)} samples       -> {HERE / 'metrics.json'}")
    print(f"[ok] wrote raw scrape (t=1200) -> {HERE / 'metrics_raw.txt'}")
    print(f"[ok] wrote {len(EVAL_RUNS)} eval runs       -> {HERE / 'eval_runs.json'}")


if __name__ == "__main__":
    main()
