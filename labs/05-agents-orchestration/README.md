# Lab 05 — Agents & Orchestration

## What

A LangGraph multi-agent customer-service workflow — router, guardrail, and async
tools — built to be **provably deterministic**. The notebook runs offline against
`shared/agent_sim.py` (a seeded stand-in for an LLM NIM and the backend services), so
"flaky" is reproducible and the fix is testable. Exercises:

- `naive_route` — reproduce the production flake: free-text LLM answer + brittle
  keyword parsing → the same ticket routes two different ways.
- `robust_route` — the fix: **structured output** (a `RouteDecision` Pydantic enum) +
  **`temperature=0`**. Variance collapses to one decision; 12/12 gold labels matched.
- `gate` + `route_by_department` — the LangGraph conditional edges
  (`guardrails → router → handler`), plus an adversarial guardrail demo.
- `gather_context` — `asyncio.gather` for concurrent tool calls (~3× faster).
- `traced_gather_context` — OpenTelemetry spans that *prove* the calls overlapped.

The production swap (a real LLM NIM via `ChatOpenAI(base_url=NIM_LLM_URL)
.with_structured_output(...)`) is shown as a comment; the graph is unchanged.

## Why

An ISV's multi-agent CS workflow is flawless in staging and flaky in production —
same input, different routing, a tool that sometimes doesn't fire. The fix isn't
"more prompt engineering": it's knowing which decisions in the graph must be
deterministic (parsing → structured output, sampling → `temperature=0`), screening
input at the edge with guardrails, and instrumenting the flow so the team can *prove*
the route is stable.

## How

```bash
task lab:data:generate LAB=05-agents-orchestration   # tickets, backend, adversarial probes
task lab:run LAB=05-agents-orchestration
task lab:test LAB=05-agents-orchestration
```

No NIM calls — the LLM and tools are simulated for reproducibility (real LangGraph,
asyncio, and OpenTelemetry libraries throughout).

**Stack:** LLM NIM + NeMo Guardrails NIM (simulated), LangGraph, asyncio,
OpenTelemetry. NeMo Agent Toolkit (NAT) is the NVIDIA-native alternative (challenge).

**You'll measure:** routing-decision variance across 12 runs (2 distinct → 1 after the
fix), routing accuracy (12/12), guardrail block rate (4/4 attacks, 8/8 labels),
concurrent-tool speedup (~3×), and trace span counts.
