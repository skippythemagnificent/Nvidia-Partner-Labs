"""Author the Lab 05 notebooks from a single source.

Emits two notebooks that stay in sync:
  - labs/05-agents-orchestration/lab.ipynb        — learner copy (exercise cells stubbed)
  - solutions/05-agents-orchestration/lab.ipynb   — completed, nbmake-clean copy

Same committed-builder pattern as Labs 01–04: edit THIS file (never the .ipynb), then:

    uv run python labs/05-agents-orchestration/build_lab.py

Lab 05 uses the *real* LangGraph, asyncio, and OpenTelemetry libraries, but the LLM
router and backend tools are the seeded, offline simulation in `shared/agent_sim.py`
so non-determinism is reproducible and the notebook runs with no GPU, no network, and
no NVIDIA_API_KEY. The production swap (a real LLM NIM) is shown as a comment.
"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/05-agents-orchestration/lab.ipynb"
SOL = ROOT / "solutions/05-agents-orchestration/lab.ipynb"

CELLS = []
def md(s): CELLS.append(("md", s))
def code(sol, stub=None): CELLS.append(("code", (sol, stub)))

# ── Scenario ─────────────────────────────────────────────────────────────────
md("""# Lab 05 — Agents & Orchestration

## Scenario

An ISV's multi-agent customer-service workflow is flawless in the staging demo and
flaky in production. The same ticket routes to **billing** on one run and **technical**
on the next; a tool that should fire sometimes doesn't. Engineering's instinct is
"prompt it harder." That's the wrong fix. The real problem is that some decisions in
the graph are **non-deterministic by construction** — free-text parsing and a sampling
temperature — and nothing is **instrumented** to prove which.

You'll reproduce the flake on demand, then engineer it away: **structured output** to
kill parse ambiguity, **`temperature=0`** to kill sampling variance, a **LangGraph**
state machine with an input **guardrail**, **async** concurrent tool calls, and
**OpenTelemetry** traces so the team can *show* the routing is now stable.

Everything runs offline against `shared/agent_sim.py`, a seeded stand-in for an LLM
NIM and the backend services — so "flaky" is reproducible and the fix is testable. In
production you swap the mock for a real LLM NIM; the graph is unchanged.""")

# ── Setup ────────────────────────────────────────────────────────────────────
md("""## Setup

Loads the tickets, the mock data backend, and the adversarial probes; wires up an
OpenTelemetry tracer with an **in-memory** span exporter (no collector needed); and
imports the simulated router, tools, and guardrail from `shared.agent_sim`.""")

code(r'''import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict

from rich import print as rprint
from rich.table import Table
from rich.console import Console

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "shared").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.agent_sim import (
    Department, RouteDecision, GuardrailResult, MockLLMRouter,
    parse_department, ground_truth_department, guardrails_check,
    fetch_account, fetch_invoices, fetch_usage,
)
from shared.utils import timed_call

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

DATA = REPO_ROOT / "labs/05-agents-orchestration/data"
TICKETS = json.loads((DATA / "tickets.json").read_text())
BACKEND = json.loads((DATA / "backend.json").read_text())
ADVERSARIAL = json.loads((DATA / "adversarial.json").read_text())

# In-memory tracing: spans land in `span_exporter` instead of a remote collector.
span_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
tracer = _provider.get_tracer("lab05-agents")

_console = Console()
SHOWCASE = "t07"   # ambiguous ticket: billing vs technical signals are tied

# Production swap (requires a running LLM NIM + NVIDIA_API_KEY):
#   from langchain_openai import ChatOpenAI
#   llm = ChatOpenAI(base_url=os.environ["NIM_LLM_URL"], model="meta/llama-3.1-8b-instruct",
#                    temperature=0).with_structured_output(RouteDecision)
#   decision = llm.invoke(f"Route this ticket: {text}")

print(f"tickets: {len(TICKETS)} | adversarial probes: {len(ADVERSARIAL)} | offline sim (no NIM, no key)")''')

md("""**Expected output:**
```
tickets: 12 | adversarial probes: 8 | offline sim (no NIM, no key)
```""")

# ── 1. Reproduce the non-determinism ─────────────────────────────────────────
md("""## 1 · Reproduce the flake

### Concept

The original router does what most v1 agents do: ask the LLM in plain language *"which
department?"*, then **string-match** the answer. Two independent things make that
non-deterministic:

- **Sampling** — at any `temperature > 0` the model samples, so an *ambiguous* ticket
  (one with strong signals for two departments) genuinely lands differently across runs.
- **Parsing** — free-text answers vary in surface form ("Billing" vs "Route to the
  payments team"), and a brittle keyword parser silently mis-maps the ones it doesn't
  recognize, dropping them to a catch-all.

### Your task — `naive_route`

Build the original pipeline: ask the router in free-text mode (`temperature=0.7`), then
`parse_department` the string. Run it `n` times on the showcase ticket and watch the
decision distribution.

**Step by step:**

1. `text = router.classify_freetext(ticket_text, temperature=0.7)`.
2. `return parse_department(text)`.""")

code(r'''def naive_route(ticket_text: str, router: MockLLMRouter) -> Department:
    """The original pipeline: free-text LLM answer, then brittle keyword parsing."""
    text = router.classify_freetext(ticket_text, temperature=0.7)
    return parse_department(text)


def decision_distribution(ticket_text: str, route_fn, n_runs: int = 12) -> Counter:
    """Route the same ticket n times (fresh seeded router) and tally the decisions."""
    router = MockLLMRouter(seed=7)
    return Counter(route_fn(ticket_text, router).value for _ in range(n_runs))


showcase = next(t for t in TICKETS if t["id"] == SHOWCASE)
dist = decision_distribution(showcase["text"], naive_route)
print(f"ticket {SHOWCASE} (gold={showcase['department']}): {showcase['text']}\n")
print(f"naive routing over 12 runs: {dict(dist)}")
print(f"distinct decisions: {len(dist)}  ->  {'NON-DETERMINISTIC' if len(dist) > 1 else 'stable'}")''',
r'''def naive_route(ticket_text: str, router: MockLLMRouter) -> Department:
    """The original pipeline: free-text LLM answer, then brittle keyword parsing."""
    # TODO: ask router.classify_freetext(ticket_text, temperature=0.7), then return
    # parse_department(...) of that free-text answer.
    raise NotImplementedError("Complete naive_route before continuing")


def decision_distribution(ticket_text: str, route_fn, n_runs: int = 12) -> Counter:
    """Route the same ticket n times (fresh seeded router) and tally the decisions."""
    router = MockLLMRouter(seed=7)
    return Counter(route_fn(ticket_text, router).value for _ in range(n_runs))


showcase = next(t for t in TICKETS if t["id"] == SHOWCASE)
dist = decision_distribution(showcase["text"], naive_route)
print(f"ticket {SHOWCASE} (gold={showcase['department']}): {showcase['text']}\n")
print(f"naive routing over 12 runs: {dict(dist)}")
print(f"distinct decisions: {len(dist)}  ->  {'NON-DETERMINISTIC' if len(dist) > 1 else 'stable'}")''')

md("""**Expected output** (same ticket, same code, two different answers — this is the
production flake, reproduced):
```
ticket t07 (gold=billing): I was charged for an API call that returned an error — can I get a refund for the failed request?

naive routing over 12 runs: {'technical': 6, 'billing': 6}
distinct decisions: 2  ->  NON-DETERMINISTIC
```
The ticket has tied billing/technical signals, so sampling splits it ~50/50 — and a
coin-flip router is a support backlog. (On a *clear* ticket the same naive pipeline
still wobbles, but from parsing alone: try the challenge.)""")

# ── 2. The fix: structured output + temperature=0 ────────────────────────────
md("""## 2 · The fix — structured output **and** `temperature=0`

### Concept

The two failure sources need two distinct fixes, and you need **both**:

- **Structured output** constrains the model to a schema — here a `RouteDecision` with
  a `Department` enum — so there is nothing to parse and no invalid value can come
  back. This kills *parse* variance. (In production: `.with_structured_output(...)` on
  the LLM NIM, which uses guided/JSON decoding.)
- **`temperature=0`** makes decoding greedy, so the same input yields the same output.
  This kills *sampling* variance.

Structured output alone still samples; `temperature=0` alone can still emit an
unparseable string. Together they make routing a pure function of the input.

### Your task — `robust_route`

Return the router's **structured** decision at `temperature=0`.

**Step by step:** `return router.classify_structured(ticket_text, temperature=0.0)` —
a validated `RouteDecision`. (`robust_route_dept` adapts it to the same
`(text, router) -> Department` shape so you can reuse `decision_distribution`.)""")

code(r'''def robust_route(ticket_text: str, router: MockLLMRouter) -> RouteDecision:
    """Deterministic routing: structured output at temperature=0."""
    return router.classify_structured(ticket_text, temperature=0.0)


def robust_route_dept(ticket_text: str, router: MockLLMRouter) -> Department:
    return robust_route(ticket_text, router).department


dist = decision_distribution(showcase["text"], robust_route_dept)
print(f"robust routing over 12 runs: {dict(dist)}  ->  distinct={len(dist)}\n")

router = MockLLMRouter(seed=7)
correct = sum(robust_route(t["text"], router).department.value == t["department"] for t in TICKETS)
print(f"routing accuracy vs gold labels: {correct}/{len(TICKETS)}")
rprint(robust_route(showcase["text"], MockLLMRouter(seed=7)))''',
r'''def robust_route(ticket_text: str, router: MockLLMRouter) -> RouteDecision:
    """Deterministic routing: structured output at temperature=0."""
    # TODO: return router.classify_structured(ticket_text, temperature=0.0)
    raise NotImplementedError("Complete robust_route before continuing")


def robust_route_dept(ticket_text: str, router: MockLLMRouter) -> Department:
    return robust_route(ticket_text, router).department


dist = decision_distribution(showcase["text"], robust_route_dept)
print(f"robust routing over 12 runs: {dict(dist)}  ->  distinct={len(dist)}\n")

router = MockLLMRouter(seed=7)
correct = sum(robust_route(t["text"], router).department.value == t["department"] for t in TICKETS)
print(f"routing accuracy vs gold labels: {correct}/{len(TICKETS)}")
rprint(robust_route(showcase["text"], MockLLMRouter(seed=7)))''')

md("""**Expected output** (one decision, every run — and it's the correct one):
```
robust routing over 12 runs: {'billing': 12}  ->  distinct=1

routing accuracy vs gold labels: 12/12
RouteDecision(department=<Department.BILLING: 'billing'>, confidence=0.5, reason='matched 2 billing signal(s)')
```
Variance collapses to a single decision, and structured routing matches every gold
label. **This** is the decision you make deterministic before anything else in the
graph depends on it.""")

# ── 3. The LangGraph state machine ───────────────────────────────────────────
md("""## 3 · Wire it into a LangGraph state machine

### Concept

Real agents are graphs: nodes do work, **edges decide what runs next**, and a shared
**state** object threads through. We'll build:

```
START → guardrails → (blocked?) → router → (which department?) → handler → END
```

Two edges are *conditional* — they branch on state the graph computed:

- after **guardrails**, a blocked input skips straight to a terminal node;
- after **router**, the `Department` chooses which handler runs.

LangGraph calls a function you supply to pick the next node from the current state.
That's the deterministic control flow your stable router now makes trustworthy.

### Your task — the two routing functions

Implement the conditional-edge functions the graph dispatches on:

1. `gate(state)` → `"blocked"` if the guardrail denied the input, else `"router"`.
2. `route_by_department(state)` → the routed department as a string
   (`state["decision"].department.value`).""")

code(r'''from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    ticket: dict
    guardrail: GuardrailResult | None
    decision: RouteDecision | None
    handler: str
    trace: list[str]


def guardrails_node(state: AgentState) -> dict:
    gr = guardrails_check(state["ticket"]["text"])
    return {"guardrail": gr, "trace": state["trace"] + ["guardrails"]}


def router_node(state: AgentState) -> dict:
    decision = robust_route(state["ticket"]["text"], MockLLMRouter(seed=7))
    return {"decision": decision, "trace": state["trace"] + ["router"]}


def _handler(name: str):
    def node(state: AgentState) -> dict:
        return {"handler": name, "trace": state["trace"] + [name]}
    return node


def gate(state: AgentState) -> str:
    """Conditional edge after guardrails: blocked inputs skip the agent."""
    return "blocked" if not state["guardrail"].allowed else "router"


def route_by_department(state: AgentState) -> str:
    """Conditional edge after the router: pick the handler for the decided department."""
    return state["decision"].department.value


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("guardrails", guardrails_node)
    g.add_node("router", router_node)
    g.add_node("blocked", _handler("blocked"))
    for dept in Department:
        g.add_node(dept.value, _handler(dept.value))
    g.add_edge(START, "guardrails")
    g.add_conditional_edges("guardrails", gate, {"blocked": "blocked", "router": "router"})
    g.add_conditional_edges("router", route_by_department, {d.value: d.value for d in Department})
    for dept in Department:
        g.add_edge(dept.value, END)
    g.add_edge("blocked", END)
    return g.compile()


graph = build_graph()
result = graph.invoke({"ticket": showcase, "guardrail": None, "decision": None, "handler": "", "trace": []})
print(f"ticket {SHOWCASE}: handler={result['handler']!r}")
print(f"path: {' -> '.join(result['trace'])}")''',
r'''from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    ticket: dict
    guardrail: GuardrailResult | None
    decision: RouteDecision | None
    handler: str
    trace: list[str]


def guardrails_node(state: AgentState) -> dict:
    gr = guardrails_check(state["ticket"]["text"])
    return {"guardrail": gr, "trace": state["trace"] + ["guardrails"]}


def router_node(state: AgentState) -> dict:
    decision = robust_route(state["ticket"]["text"], MockLLMRouter(seed=7))
    return {"decision": decision, "trace": state["trace"] + ["router"]}


def _handler(name: str):
    def node(state: AgentState) -> dict:
        return {"handler": name, "trace": state["trace"] + [name]}
    return node


def gate(state: AgentState) -> str:
    """Conditional edge after guardrails: blocked inputs skip the agent."""
    # TODO: return "blocked" if the guardrail did not allow the input, else "router".
    raise NotImplementedError("Complete gate before continuing")


def route_by_department(state: AgentState) -> str:
    """Conditional edge after the router: pick the handler for the decided department."""
    # TODO: return state["decision"].department.value
    raise NotImplementedError("Complete route_by_department before continuing")


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("guardrails", guardrails_node)
    g.add_node("router", router_node)
    g.add_node("blocked", _handler("blocked"))
    for dept in Department:
        g.add_node(dept.value, _handler(dept.value))
    g.add_edge(START, "guardrails")
    g.add_conditional_edges("guardrails", gate, {"blocked": "blocked", "router": "router"})
    g.add_conditional_edges("router", route_by_department, {d.value: d.value for d in Department})
    for dept in Department:
        g.add_edge(dept.value, END)
    g.add_edge("blocked", END)
    return g.compile()


graph = build_graph()
result = graph.invoke({"ticket": showcase, "guardrail": None, "decision": None, "handler": "", "trace": []})
print(f"ticket {SHOWCASE}: handler={result['handler']!r}")
print(f"path: {' -> '.join(result['trace'])}")''')

md("""**Expected output:**
```
ticket t07: handler='billing'
path: guardrails -> router -> billing
```
The state machine is explicit and inspectable: every ticket follows
`guardrails → router → <department>`, and because the router is deterministic, so is
the path.""")

# ── guardrails defense section ───────────────────────────────────────────────
md("""### 🛡️ The guardrail in action — adversarial inputs

The `guardrails` node isn't decoration: it's the first line of defense. NeMo
Guardrails (deployed as its own NIM in production) screens input for prompt-injection
and data-exfiltration *before* the agent or its tools ever run. Replay the adversarial
probes through the graph and confirm they short-circuit to `blocked`.""")

code(r'''tbl = Table(title="Guardrail screening of adversarial probes")
for c in ("probe", "input (truncated)", "handler", "expected"):
    tbl.add_column(c)
blocked_correct = 0
for a in ADVERSARIAL:
    res = graph.invoke({"ticket": {"text": a["text"], "account_id": "n/a"},
                        "guardrail": None, "decision": None, "handler": "", "trace": []})
    is_blocked = res["handler"] == "blocked"
    blocked_correct += int(is_blocked == a["blocked"])
    tbl.add_row(a["id"], a["text"][:42] + "…", res["handler"],
                "blocked" if a["blocked"] else "allowed")
_console.print(tbl)
print(f"\nguardrail decisions matching labels: {blocked_correct}/{len(ADVERSARIAL)} "
      f"| blocked {sum(a['blocked'] for a in ADVERSARIAL)}/{len(ADVERSARIAL)}")''')

md("""**Expected output** (every injection / exfiltration attempt is stopped; benign
lookalikes pass):
```
                       Guardrail screening of adversarial probes
... a1 prompt-injection -> blocked ... a3 data-exfiltration -> blocked ...
... a5 (real refund)    -> billing ... a6 (real 500 error)  -> technical ...

guardrail decisions matching labels: 8/8 | blocked 4/8
```
The four attacks route to `blocked`; the four genuine requests flow on to their
department. A blocked input never reaches a tool — that's the point of screening at
the graph's edge.""")

# ── 4. Async concurrent tools ────────────────────────────────────────────────
md("""## 4 · Gather context with **concurrent** tool calls

### Concept

Once routed, an agent enriches the ticket with context — account, invoices, usage.
These are independent I/O-bound lookups, so calling them **sequentially** wastes wall
time: three 50 ms calls take 150 ms back-to-back, but ~50 ms if you fire them at once.
`asyncio.gather` runs coroutines concurrently and collects their results in order.
(CLAUDE.md rule: agent tool calls are async, never sequential blocking calls.)

### Your task — `gather_context`

Fan out the three tool coroutines concurrently and return their results.

**Step by step:** `await asyncio.gather(fetch_account(account_id, BACKEND),
fetch_invoices(account_id, BACKEND), fetch_usage(account_id, BACKEND))` and return the
three results (e.g. as a dict).""")

code(r'''async def gather_context(account_id: str) -> dict:
    """Fetch account, invoices, and usage concurrently."""
    account, invoices, usage = await asyncio.gather(
        fetch_account(account_id, BACKEND),
        fetch_invoices(account_id, BACKEND),
        fetch_usage(account_id, BACKEND),
    )
    return {"account": account, "invoices": invoices, "usage": usage}


async def gather_sequential(account_id: str) -> dict:
    account = await fetch_account(account_id, BACKEND)
    invoices = await fetch_invoices(account_id, BACKEND)
    usage = await fetch_usage(account_id, BACKEND)
    return {"account": account, "invoices": invoices, "usage": usage}


aid = showcase["account_id"]
_, seq_ms = await _timed(gather_sequential, aid)   # noqa: F821  (defined below)
ctx, conc_ms = await _timed(gather_context, aid)
print(f"sequential: {seq_ms:5.0f} ms | concurrent gather: {conc_ms:5.0f} ms | "
      f"speedup ~{seq_ms / conc_ms:.1f}x")
print(f"account tier: {ctx['account']['tier']} | invoices: {len(ctx['invoices']['invoices'])} | "
      f"api calls 30d: {ctx['usage']['usage']['api_calls_30d']}")''',
r'''async def gather_context(account_id: str) -> dict:
    """Fetch account, invoices, and usage concurrently."""
    # TODO: await asyncio.gather(fetch_account(account_id, BACKEND),
    #   fetch_invoices(account_id, BACKEND), fetch_usage(account_id, BACKEND))
    # and return {"account": ..., "invoices": ..., "usage": ...}.
    raise NotImplementedError("Complete gather_context before continuing")


async def gather_sequential(account_id: str) -> dict:
    account = await fetch_account(account_id, BACKEND)
    invoices = await fetch_invoices(account_id, BACKEND)
    usage = await fetch_usage(account_id, BACKEND)
    return {"account": account, "invoices": invoices, "usage": usage}


aid = showcase["account_id"]
_, seq_ms = await _timed(gather_sequential, aid)   # noqa: F821  (defined below)
ctx, conc_ms = await _timed(gather_context, aid)
print(f"sequential: {seq_ms:5.0f} ms | concurrent gather: {conc_ms:5.0f} ms | "
      f"speedup ~{seq_ms / conc_ms:.1f}x")
print(f"account tier: {ctx['account']['tier']} | invoices: {len(ctx['invoices']['invoices'])} | "
      f"api calls 30d: {ctx['usage']['usage']['api_calls_30d']}")''')

md("""**Expected output** (≈3× faster — the three 50 ms lookups overlap):
```
sequential:   ~154 ms | concurrent gather:  ~51 ms | speedup ~3.0x
account tier: free | invoices: 3 | api calls 30d: 7000
```
Sequential `await`s block one another; `gather` overlaps the I/O. For an agent making
several tool calls per turn, this is the difference between a snappy and a sluggish
assistant.""")

# helper cell (not an exercise) — defined after first use is fine in a notebook only
# because the using cell is executed after this one. We place it BEFORE in build order.

# ── 5. Instrument with OpenTelemetry ─────────────────────────────────────────
md("""## 5 · Prove it with traces

### Concept

"It's deterministic now, trust me" doesn't survive a postmortem. **Distributed
tracing** records each step as a **span** in a tree, so you can *show* exactly what
ran, in what order, and how long it took. OpenTelemetry is the vendor-neutral
standard; here spans land in an in-memory exporter, but the same code exports to
Jaeger/Tempo in production. You instrument by wrapping work in
`tracer.start_as_current_span(name)`; nested `with` blocks become parent/child spans.

### Your task — `traced_gather_context`

Wrap the concurrent context-gather in spans: one parent span for the gather and one
child span per tool call.

**Step by step:**

1. Open a parent span: `with tracer.start_as_current_span("gather_context"):`.
2. Inside, define a small async wrapper that opens
   `tracer.start_as_current_span(f"tool:{coro_name}")` around each `await`.
3. `await asyncio.gather(...)` the three wrapped tool calls and return the results.""")

code(r'''async def _traced_tool(name: str, coro):
    with tracer.start_as_current_span(f"tool:{name}"):
        return await coro


async def traced_gather_context(account_id: str) -> dict:
    """Concurrent gather, each tool call wrapped in its own span under a parent span."""
    with tracer.start_as_current_span("gather_context"):
        account, invoices, usage = await asyncio.gather(
            _traced_tool("fetch_account", fetch_account(account_id, BACKEND)),
            _traced_tool("fetch_invoices", fetch_invoices(account_id, BACKEND)),
            _traced_tool("fetch_usage", fetch_usage(account_id, BACKEND)),
        )
    return {"account": account, "invoices": invoices, "usage": usage}


span_exporter.clear()
await traced_gather_context(showcase["account_id"])
spans = span_exporter.get_finished_spans()
print(f"spans recorded: {len(spans)}")
for s in spans:
    print(f"  {s.name:24} {(s.end_time - s.start_time) / 1e6:.0f} ms")''',
r'''async def _traced_tool(name: str, coro):
    with tracer.start_as_current_span(f"tool:{name}"):
        return await coro


async def traced_gather_context(account_id: str) -> dict:
    """Concurrent gather, each tool call wrapped in its own span under a parent span."""
    # TODO: open a parent span "gather_context"; inside it, asyncio.gather the three
    # tool calls each wrapped via _traced_tool("<name>", <coro>); return the results.
    raise NotImplementedError("Complete traced_gather_context before continuing")


span_exporter.clear()
await traced_gather_context(showcase["account_id"])
spans = span_exporter.get_finished_spans()
print(f"spans recorded: {len(spans)}")
for s in spans:
    print(f"  {s.name:24} {(s.end_time - s.start_time) / 1e6:.0f} ms")''')

md("""**Expected output** (one parent + three child spans; the parent's wall time is
~one tool, not three, because they ran concurrently):
```
spans recorded: 4
  tool:fetch_account        ~50 ms
  tool:fetch_invoices       ~50 ms
  tool:fetch_usage          ~50 ms
  gather_context            ~51 ms
```
Children close before the parent, and the parent ≈ a single tool's latency — the trace
*proves* the calls overlapped. Wire this onto the router and guardrail nodes too and
you can answer "why did ticket X route there?" from the trace alone.""")

# ── Challenge ────────────────────────────────────────────────────────────────
md("""## Challenge

1. **Parse-only flake.** Run `decision_distribution` with `naive_route` on a *clear*
   ticket (e.g. `t01`, all-billing). It's unambiguous, yet the distribution still has
   more than one entry. Which failure source is left, and which half of the §2 fix
   removes it on its own?
2. **End-to-end trace.** Wrap `guardrails_node`, `router_node`, and the handler in
   spans too, so one ticket produces a full `handle_ticket` trace. How many spans is a
   clean billing ticket end-to-end, and how many is a blocked adversarial one (hint:
   it never reaches the tools)?
3. **NeMo Agent Toolkit.** The graph here is hand-wired LangGraph. Sketch how you'd
   express the same router → tools → guardrail flow in NVIDIA's NeMo Agent Toolkit
   (NAT), and note one thing NAT gives you for free that you wrote by hand (hint:
   per-node telemetry and config-driven wiring).""")

# ── Key takeaways ────────────────────────────────────────────────────────────
md("""## Key takeaways

- **Non-determinism has two sources, and you fix both.** Free-text **parsing** →
  structured output; sampling **temperature** → `temperature=0`. Either alone still
  flakes; together, routing is a pure function of the input.
- **Make control-flow decisions deterministic first.** A LangGraph conditional edge is
  only as trustworthy as the state it branches on — pin the router before the graph
  depends on it.
- **Screen at the edge.** A guardrail node blocks prompt-injection and exfiltration
  *before* any tool runs; a blocked input must never reach the data plane.
- **Tool calls are concurrent.** `asyncio.gather` overlaps independent I/O — ~3× here,
  more as tool count grows.
- **Instrument so you can prove it.** OpenTelemetry spans turn "trust me" into a trace
  that shows the route, the order, and the latency.

**References**
- LangGraph: https://langchain-ai.github.io/langgraph/
- NeMo Agent Toolkit (NAT): https://docs.nvidia.com/nemo/agent-toolkit/
- NeMo Guardrails: https://docs.nvidia.com/nemo/guardrails/
- Structured output (guided decoding) on NIM: https://docs.nvidia.com/nim/large-language-models/latest/
- OpenTelemetry (Python): https://opentelemetry.io/docs/languages/python/""")

# ── helper injection ─────────────────────────────────────────────────────────
# `_timed` is used by §4; define it in the setup region by appending to the first
# code cell is awkward, so we inject a tiny helper cell right after setup. To keep the
# build linear, we register it here and reorder before writing.
HELPER_SRC = r'''async def _timed(coro_fn, *args):
    """await a coroutine function and return (result, elapsed_ms)."""
    import time
    start = time.perf_counter()
    result = await coro_fn(*args)
    return result, (time.perf_counter() - start) * 1000'''


# ── build ─────────────────────────────────────────────────────────────────────
def make(use_stub: bool):
    nb = new_notebook()
    cells = []
    # Insert the async helper as the second code cell (right after setup) so §4 can use it.
    inserted = False
    seen_code = 0
    for kind, payload in CELLS:
        if kind == "md":
            cells.append(new_markdown_cell(payload))
        else:
            sol, stub = payload
            cells.append(new_code_cell(stub if (use_stub and stub) else sol))
            seen_code += 1
            if seen_code == 1 and not inserted:   # after the setup code cell
                cells.append(new_code_cell(HELPER_SRC))
                inserted = True
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
