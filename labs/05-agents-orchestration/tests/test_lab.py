"""Validation for Lab 05.

Exercises the seeded agent simulation directly (the same primitives the notebook
imports from shared.agent_sim) to assert the determinism story, the guardrail block
rate, concurrent-tool speedup, and trace span counts — plus data invariants and
end-to-end execution of the lab notebook.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import Counter

import pytest

from shared.agent_sim import (
    Department,
    GuardrailResult,
    MockLLMRouter,
    RouteDecision,
    fetch_account,
    fetch_invoices,
    fetch_usage,
    ground_truth_department,
    guardrails_check,
    parse_department,
)

SHOWCASE = "t07"


# ── reference routing (mirrors the notebook) ─────────────────────────────────


def naive_route(text, router):
    return parse_department(router.classify_freetext(text, temperature=0.7))


def robust_route(text, router):
    return router.classify_structured(text, temperature=0.0)


def distribution(text, route_fn, n=12, seed=7):
    router = MockLLMRouter(seed=seed)
    return Counter(route_fn(text, router) for _ in range(n))


# ── data invariants ──────────────────────────────────────────────────────────


def test_ticket_data(tickets, backend):
    assert len(tickets) == 12
    assert {t["department"] for t in tickets} == {"billing", "technical", "account", "general"}
    for t in tickets:
        assert t["account_id"] in backend["accounts"]


def test_adversarial_data(adversarial):
    assert len(adversarial) == 8
    assert sum(a["blocked"] for a in adversarial) == 4


# ── determinism story ────────────────────────────────────────────────────────


def test_naive_routing_is_nondeterministic(tickets):
    """The ambiguous showcase ticket gets >1 distinct decision under naive routing."""
    text = next(t for t in tickets if t["id"] == SHOWCASE)["text"]
    dist = distribution(text, lambda x, r: naive_route(x, r).value)
    assert len(dist) > 1, "expected the ambiguous ticket to flake under naive routing"


def test_structured_temp0_is_deterministic(tickets):
    text = next(t for t in tickets if t["id"] == SHOWCASE)["text"]
    dist = distribution(text, lambda x, r: robust_route(x, r).department.value)
    assert len(dist) == 1, "structured + temperature=0 must be deterministic"
    assert next(iter(dist)) == "billing", "and it must land on the correct gold label"


def test_structured_routing_matches_all_gold_labels(tickets):
    router = MockLLMRouter(seed=7)
    correct = sum(robust_route(t["text"], router).department.value == t["department"] for t in tickets)
    assert correct == len(tickets) == 12


def test_ground_truth_router_is_perfect(tickets):
    assert all(ground_truth_department(t["text"]).value == t["department"] for t in tickets)


def test_structured_output_validates_to_enum(tickets):
    router = MockLLMRouter(seed=7)
    dec = robust_route(tickets[0]["text"], router)
    assert isinstance(dec.department, Department)
    assert 0.0 <= dec.confidence <= 1.0


# ── guardrails ───────────────────────────────────────────────────────────────


def test_guardrails_block_rate(adversarial):
    correct = sum((not guardrails_check(a["text"]).allowed) == a["blocked"] for a in adversarial)
    assert correct == len(adversarial) == 8
    blocked = [a for a in adversarial if a["blocked"]]
    for a in blocked:
        res = guardrails_check(a["text"])
        assert not res.allowed and res.category == a["category"]


# ── async concurrency ────────────────────────────────────────────────────────


def test_gather_is_faster_than_sequential(backend, tickets):
    aid = tickets[0]["account_id"]

    async def seq():
        await fetch_account(aid, backend)
        await fetch_invoices(aid, backend)
        await fetch_usage(aid, backend)

    async def conc():
        return await asyncio.gather(
            fetch_account(aid, backend), fetch_invoices(aid, backend), fetch_usage(aid, backend)
        )

    t = time.perf_counter()
    asyncio.run(seq())
    seq_s = time.perf_counter() - t
    t = time.perf_counter()
    asyncio.run(conc())
    conc_s = time.perf_counter() - t
    assert conc_s < seq_s / 1.8, f"gather should be ~3x faster: seq={seq_s:.3f}s conc={conc_s:.3f}s"


# ── tracing ──────────────────────────────────────────────────────────────────


def test_traced_gather_emits_four_spans(backend, tickets):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    aid = tickets[0]["account_id"]

    async def traced_tool(name, coro):
        with tracer.start_as_current_span(f"tool:{name}"):
            return await coro

    async def traced_gather():
        with tracer.start_as_current_span("gather_context"):
            return await asyncio.gather(
                traced_tool("fetch_account", fetch_account(aid, backend)),
                traced_tool("fetch_invoices", fetch_invoices(aid, backend)),
                traced_tool("fetch_usage", fetch_usage(aid, backend)),
            )

    asyncio.run(traced_gather())
    names = sorted(s.name for s in exporter.get_finished_spans())
    assert names == ["gather_context", "tool:fetch_account", "tool:fetch_invoices", "tool:fetch_usage"]


# ── LangGraph wiring ─────────────────────────────────────────────────────────


def test_graph_routes_and_blocks(tickets, adversarial):
    """The compiled graph routes a clean ticket and short-circuits an attack."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        ticket: dict
        guardrail: GuardrailResult | None
        decision: RouteDecision | None
        handler: str
        trace: list

    def guardrails_node(s):
        return {"guardrail": guardrails_check(s["ticket"]["text"]), "trace": s["trace"] + ["guardrails"]}

    def router_node(s):
        return {"decision": robust_route(s["ticket"]["text"], MockLLMRouter(seed=7)),
                "trace": s["trace"] + ["router"]}

    def handler(name):
        return lambda s: {"handler": name, "trace": s["trace"] + [name]}

    g = StateGraph(S)
    g.add_node("guardrails", guardrails_node)
    g.add_node("router", router_node)
    g.add_node("blocked", handler("blocked"))
    for d in Department:
        g.add_node(d.value, handler(d.value))
    g.add_edge(START, "guardrails")
    g.add_conditional_edges("guardrails", lambda s: "blocked" if not s["guardrail"].allowed else "router",
                            {"blocked": "blocked", "router": "router"})
    g.add_conditional_edges("router", lambda s: s["decision"].department.value,
                            {d.value: d.value for d in Department})
    for d in Department:
        g.add_edge(d.value, END)
    g.add_edge("blocked", END)
    app = g.compile()

    base = {"guardrail": None, "decision": None, "handler": "", "trace": []}
    clean = app.invoke({**base, "ticket": next(t for t in tickets if t["id"] == SHOWCASE)})
    assert clean["handler"] == "billing"
    assert clean["trace"] == ["guardrails", "router", "billing"]

    attack = app.invoke({**base, "ticket": {"text": next(a["text"] for a in adversarial if a["blocked"]),
                                            "account_id": "n/a"}})
    assert attack["handler"] == "blocked"


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
