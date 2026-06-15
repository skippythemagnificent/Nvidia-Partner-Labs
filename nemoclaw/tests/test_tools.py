"""Pin NeMoClaw's diagnostic tools to the labs' verified canonical numbers.

If a lab's expected output changes, these tests fail — that's the point: the MCP
tools and the lab notebooks must not drift apart. RAG tests load local models and are
marked `slow` (run with `-m slow` or just run the whole file).
"""

from __future__ import annotations

import asyncio

import pytest

from nemoclaw.diagnostics import fixtures as fx
from nemoclaw.diagnostics import lab03_nim as l3
from nemoclaw.diagnostics import lab04_gpu as l4
from nemoclaw.diagnostics import lab05_agents as l5
from nemoclaw.diagnostics import lab06_mlops as l6


# ── Lab 03 — NIM deployment ──────────────────────────────────────────────────
def test_parse_startup_time_to_ready():
    rep = l3.parse_startup(fx.nim_logs()["startup_healthy"])
    assert rep.time_to_ready_s == 148.043
    assert rep.kv_cache_tokens == 117440
    assert rep.precision == "fp8"


def test_diagnose_known_failures():
    logs = fx.nim_logs()
    assert l3.diagnose(logs["ngc_auth_failure"]).failure == "ngc_auth"
    assert l3.diagnose(logs["kv_cache_exhaustion"]).failure == "kv_cache_exhaustion"
    assert l3.diagnose(logs["airgapped_offline"]).failure == "airgap_cache_miss"
    assert l3.diagnose(logs["startup_healthy"]).failure == "unknown"


def test_profile_split_h100_vs_a100():
    dep, profs = fx.nim_deployment(), fx.nim_profiles()
    h = l3.select_profile(dep["dev_box_gpu"], profs)
    a = l3.select_profile(dep["airgapped_target_gpu"], profs)
    assert (h.profile_id, h.precision) == ("8835c31752fd", "fp8")
    assert (a.profile_id, a.precision) == ("6f1ac2d40b77", "fp16")


def test_kv_cache_ceiling():
    assert l3.max_concurrent_requests(117440, l3.MAX_MODEL_LEN) == 28


def test_airgap_missing_artifacts():
    dep, profs, cache = fx.nim_deployment(), fx.nim_profiles(), fx.nim_cache_manifest()
    target = l3.select_profile(dep["airgapped_target_gpu"], profs)
    required = next(p["artifacts"] for p in profs if p["id"] == target.profile_id)
    assert l3.missing_artifacts(required, cache["present_artifacts"]) == [
        "models/6f1ac2d40b77/config.json",
        "models/6f1ac2d40b77/rank0.engine",
    ]


# ── Lab 04 — GPU architecture ────────────────────────────────────────────────
def test_kv_bytes_per_token():
    M = fx.models()
    assert l4.kv_cache_bytes_per_token(M["llama31_8b"], "fp16") == 131072
    assert l4.kv_cache_bytes_per_token(M["llama31_70b"], "fp16") == 327680


def test_70b_needs_tensor_parallelism():
    M, G = fx.models(), fx.gpus()
    usable = G["a100_80_sxm"]["memory_gb"] * 1e9 * l4.MEM_UTIL
    assert l4.weight_bytes(M["llama31_70b"], "fp16") > usable  # doesn't fit one
    assert l4.min_tp_to_fit(M["llama31_70b"], G["a100_80_sxm"], "fp16") == 2
    assert l4.min_tp_to_fit(M["llama31_70b"], G["a100_80_sxm"], "fp8") == 1


def test_chat_capacity_plan():
    plan = l4.capacity_planner(
        fx.workloads()["chat"], fx.gpus()["h100_sxm"], fx.models()["llama31_8b"], "fp16"
    )
    assert plan.num_gpus == 1
    assert plan.binding_constraint == "KV cache"
    assert plan.fits_on_one_gpu is True
    assert plan.ttft_ok is True


# ── Lab 06 — MLOps observability ─────────────────────────────────────────────
def test_slo_breach_window():
    m = fx.metrics()
    assert l6.slo_breach_window(
        m["samples"], "ttft_p99_ms", m["slo"]["ttft_p99_ms"]
    ) == (1080, 1470)


def test_alert_lead_time_is_120s():
    m = fx.metrics()
    s, slo = m["samples"], m["slo"]
    _, _, lead = l6.alert_lead_time(
        s, "kv_cache_util", slo["kv_warn"], "ttft_p99_ms", slo["ttft_p99_ms"]
    )
    assert lead == 120


def test_ragas_regression_gate_trips_run_10():
    ev = fx.eval_runs()
    failing = l6.regression_gate(ev["runs"], ev["faithfulness_gate"])
    assert failing is not None
    assert failing["run"] == 10


# ── Lab 05 — agents ──────────────────────────────────────────────────────────
def test_robust_routing_is_accurate_and_deterministic():
    router = l5.MockLLMRouter(seed=7)
    tickets = fx.tickets()
    acc = sum(
        l5.robust_route(t["text"], router).department.value == t["department"]
        for t in tickets
    )
    assert acc == len(tickets) == 12
    # ambiguous ticket t07: robust collapses to a single decision, naive does not
    t07 = next(t for t in tickets if t["id"] == "t07")["text"]
    assert len(l5.decision_distribution(t07, l5.robust_route_dept)) == 1
    assert len(l5.decision_distribution(t07, l5.naive_route)) > 1


def test_guardrails_block_count():
    blocked = sum(not l5.guardrails_check(a["text"]).allowed for a in fx.adversarial())
    assert blocked == 4


# ── MCP server wiring ────────────────────────────────────────────────────────
def test_server_registers_expected_tools():
    from nemoclaw.server.__main__ import mcp

    names = {t.name for t in asyncio.run(mcp.list_tools())}
    for expected in (
        "nim_diagnose_log",
        "gpu_capacity_plan",
        "mlops_alert_lead_time",
        "agent_route_ticket",
        "rag_eval",
        "list_scenarios",
    ):
        assert expected in names


# ── Lab 01/02 — RAG (slow: loads local embedding/cross-encoder models) ───────
@pytest.mark.slow
def test_rag_hit_rate_fixed_vs_sentence():
    from nemoclaw.diagnostics import lab0102_rag as rag

    corpus, evalset = fx.rag_corpus("01"), fx.rag_eval("01")
    fixed = rag.build_index(rag.chunk_corpus(corpus, rag._make_chunker("fixed")))
    sent = rag.build_index(rag.chunk_corpus(corpus, rag._make_chunker("sentence")))
    fixed_hr, _ = rag.hit_rate(fixed, evalset, 3)
    sent_hr, _ = rag.hit_rate(sent, evalset, 3)
    assert fixed_hr == pytest.approx(0.5, abs=1e-9)
    assert sent_hr == pytest.approx(0.75, abs=1e-9)
    assert sent_hr > fixed_hr  # sentence-aware chunking recovers boundary-split answers
