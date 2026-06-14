"""Validation for Lab 06.

Reimplements the reference observability math (mirroring the notebook) and asserts the
incident's canonical numbers — parse, rates, percentiles, the SLO-breach window, the
alert lead time, and the RAGAS gate — plus data invariants and end-to-end execution of
the solution notebook. The learner `lab.ipynb` is not executed here; its stubs fail by
design.
"""
from __future__ import annotations

import json
import re

import pytest


# ── reference logic (mirrors the notebook) ───────────────────────────────────


def parse_prometheus(text: str) -> dict[str, float]:
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(3))
    return out


def counter_rate(samples, key, step_s):
    return [(samples[i][key] - samples[i - 1][key]) / step_s for i in range(1, len(samples))]


def percentile(values, p):
    v = sorted(values)
    k = (len(v) - 1) * p
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def slo_breach_window(samples, key, threshold):
    b = [s["t_s"] for s in samples if s[key] > threshold]
    return (min(b), max(b)) if b else None


def alert_lead_time(samples, lead_key, lead_warn, slo_key, slo_threshold):
    warn_t = next(s["t_s"] for s in samples if s[lead_key] >= lead_warn)
    breach_t = next(s["t_s"] for s in samples if s[slo_key] > slo_threshold)
    return warn_t, breach_t, breach_t - warn_t


def regression_gate(runs, gate):
    return next((r for r in runs if r["faithfulness"] < gate), None)


# ── data invariants ──────────────────────────────────────────────────────────


def test_series_shape_and_monotonic_counters(metrics):
    samples = metrics["samples"]
    assert len(samples) == 61
    assert metrics["step_s"] == 30
    for a, b in zip(samples, samples[1:]):
        assert b["request_success_total"] >= a["request_success_total"]
        assert b["request_failure_total"] >= a["request_failure_total"]


def test_slo_block_present(metrics):
    assert metrics["slo"] == {"ttft_p99_ms": 500, "itl_p99_ms": 50, "kv_warn": 0.9}


# ── parse ────────────────────────────────────────────────────────────────────


def test_parse_prometheus(raw_scrape):
    scrape = parse_prometheus(raw_scrape)
    assert len(scrape) == 13
    assert scrape["gpu_cache_usage_perc"] == 0.99
    assert scrape["num_requests_waiting"] == 41.0
    assert scrape["DCGM_FI_DEV_GPU_UTIL"] == 99.0
    assert scrape["time_to_first_token_seconds_p99"] == pytest.approx(0.88)
    # comment lines and blank lines must not become metrics
    assert not any(k.startswith("#") for k in scrape)


# ── rates ────────────────────────────────────────────────────────────────────


def test_counter_rate_recovers_load(metrics):
    samples = metrics["samples"]
    rps = counter_rate(samples, "request_success_total", metrics["step_s"])
    assert len(rps) == 60
    assert round(rps[0]) == 20                 # baseline
    assert round(max(rps)) == 55               # incident peak (offered ~58, minus 503s)
    fail = counter_rate(samples, "request_failure_total", metrics["step_s"])
    assert max(fail) == pytest.approx(3.48, abs=0.05)
    assert fail[0] == 0.0                       # no failures at baseline


# ── percentiles ──────────────────────────────────────────────────────────────


def test_percentile_interpolation():
    assert percentile([10, 20, 30, 40], 0.5) == 25.0
    assert percentile([1], 0.99) == 1.0


def test_ttft_tail_far_exceeds_median(metrics):
    ttft = [s["ttft_p99_ms"] for s in metrics["samples"]]
    assert percentile(ttft, 0.50) == 180
    assert percentile(ttft, 0.99) > metrics["slo"]["ttft_p99_ms"]   # tail breaches SLO


# ── breach window + lead time ────────────────────────────────────────────────


def test_breach_window(metrics):
    samples = metrics["samples"]
    window = slo_breach_window(samples, "ttft_p99_ms", metrics["slo"]["ttft_p99_ms"])
    assert window == (1080, 1470)


def test_alert_lead_time_is_120s(metrics):
    samples, slo = metrics["samples"], metrics["slo"]
    warn_t, breach_t, lead = alert_lead_time(
        samples, "kv_cache_util", slo["kv_warn"], "ttft_p99_ms", slo["ttft_p99_ms"])
    assert (warn_t, breach_t, lead) == (960, 1080, 120)
    assert lead > 0, "the leading indicator must precede the breach"


# ── RAGAS gate ───────────────────────────────────────────────────────────────


def test_regression_gate(eval_runs):
    failing = regression_gate(eval_runs["runs"], eval_runs["faithfulness_gate"])
    assert failing is not None
    assert failing["run"] == 10
    assert failing["faithfulness"] < eval_runs["faithfulness_gate"]
    # everything before run 10 must pass the gate
    for r in eval_runs["runs"][:9]:
        assert r["faithfulness"] >= eval_runs["faithfulness_gate"]


# ── build integrity + solution execution ─────────────────────────────────────


def test_learner_has_stubs_solution_does_not(solution_nb):
    lab = solution_nb.parent.parent.parent / "labs/06-mlops-platform/lab.ipynb"
    lab_src = " ".join("".join(c["source"]) for c in json.loads(lab.read_text())["cells"])
    sol_src = " ".join("".join(c["source"]) for c in json.loads(solution_nb.read_text())["cells"])
    assert "TODO" in lab_src and "NotImplementedError" in lab_src, "learner copy lost its stubs"
    assert "TODO" not in sol_src, "solution still contains TODO markers"


@pytest.mark.slow
def test_solution_notebook_executes(solution_nb):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    nb = nbformat.read(str(solution_nb), as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(solution_nb.parent)}},
    )
    client.execute()
