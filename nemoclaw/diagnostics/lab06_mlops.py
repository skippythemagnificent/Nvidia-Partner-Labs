"""Lab 06 — MLOps & platform observability (offline metric analysis).

Faithful port of the verified solution logic in
``labs/06-mlops-platform/build_lab.py``. Pinned by the test suite against the labs'
canonical numbers (TTFT p99 breach window 1080-1470s, alert lead time 120s, RAGAS
regression gate trips run #10).
"""

from __future__ import annotations

import re

SPARK = "▁▂▃▄▅▆▇█"


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition text into {metric_name: value} (labels dropped)."""
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(3))
    return out


def counter_rate(samples: list[dict], key: str, step_s: int) -> list[float]:
    """Per-interval rate of a cumulative counter (the PromQL rate() idea)."""
    return [
        (samples[i][key] - samples[i - 1][key]) / step_s for i in range(1, len(samples))
    ]


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated p-th percentile (p in [0, 1])."""
    v = sorted(values)
    k = (len(v) - 1) * p
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def slo_breach_window(
    samples: list[dict], key: str, threshold: float
) -> tuple[int, int] | None:
    """First and last timestamps where `key` exceeds `threshold` (None if never)."""
    breaching = [s["t_s"] for s in samples if s[key] > threshold]
    if not breaching:
        return None
    return min(breaching), max(breaching)


def alert_lead_time(samples, lead_key, lead_warn, slo_key, slo_threshold):
    """Seconds of warning a leading indicator gives before the SLO breaches."""
    warn_t = next(s["t_s"] for s in samples if s[lead_key] >= lead_warn)
    breach_t = next(s["t_s"] for s in samples if s[slo_key] > slo_threshold)
    return warn_t, breach_t, breach_t - warn_t


def regression_gate(runs: list[dict], gate: float) -> dict | None:
    """First eval run that drops below the faithfulness gate (triggers a retrain)."""
    return next((r for r in runs if r["faithfulness"] < gate), None)


def sparkline(values) -> str:
    """Tiny inline chart so a metric's shape is visible without a plotting lib."""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    return "".join(
        SPARK[min(len(SPARK) - 1, int((v - lo) / span * (len(SPARK) - 1)))]
        for v in values
    )
