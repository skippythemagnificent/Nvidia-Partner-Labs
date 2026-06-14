# Lab 06 — MLOps & Platform

## What

The observability-and-quality capstone — fully analytical, no GPU or NIM calls. You
replay a sample Triton `:8002/metrics` scrape and a 30-minute time series (with a
built-in incident) and build the loop that catches a production outage *and* a quality
regression:

- `parse_prometheus` — turn a raw exposition-format scrape (Triton + DCGM) into a
  metric dict.
- `counter_rate` — recover requests/sec and tokens/sec from cumulative counters (what
  PromQL `rate()` does).
- `percentile` — tail-latency summary (p50/p95/p99); the SLA lives in the tail.
- `slo_breach_window` — the contiguous window where TTFT p99 violated its SLO.
- `alert_lead_time` — the capstone: the KV-cache warning leads the latency breach by
  **120 s** of runway.
- `regression_gate` — a nightly RAGAS faithfulness gate that fires a NeMo Customizer
  retrain, closing the loop.

The incident traces straight back to earlier labs: KV-cache saturation → 503s (Lab 03)
and the latency roofline (Lab 04).

## Why

A NIM cluster under load looks healthy at the pod level right up until p99 explodes —
because the signal lives in metrics nobody boarded: KV-cache utilization, queue depth,
GPU SM occupancy, and retrieval-quality drift. Triton, DCGM, and RAGAS each show one
slice; the team needs all three on one Grafana board, an alert on the *leading*
indicator, and a closed-loop trigger back to retraining when quality drifts.

## How

```bash
task lab:data:generate LAB=06-mlops-platform    # writes metrics_raw.txt, metrics.json, eval_runs.json

# Option A: replay sample metrics (no infra)
task lab:run LAB=06-mlops-platform
task lab:test LAB=06-mlops-platform

# Option B: live infra
task infra:up STACK=staging
task infra:env STACK=staging            # writes PROMETHEUS_URL + GRAFANA_URL to .env
task lab:run LAB=06-mlops-platform
```

**Stack:** Triton metrics, Prometheus, DCGM Exporter, Grafana, RAGAS, NeMo Customizer.

**You'll measure:** RPS/tokens-per-sec from counters, TTFT-p99 tail percentiles, the
SLO-breach window (t=1080→1470s), the alert lead time (120 s from the KV-cache
warning), 503s during the incident (~1,500), and the RAGAS faithfulness drift that
trips the 0.85 gate on night 10.
