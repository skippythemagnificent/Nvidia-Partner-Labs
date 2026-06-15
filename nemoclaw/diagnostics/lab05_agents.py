"""Lab 05 — agents & orchestration (deterministic, offline simulation).

The Lab 05 logic already lives in a clean importable module, ``shared.agent_sim``;
this file just re-exports it and adds the thin routing wrappers the notebook defines,
so NeMoClaw can demonstrate the non-determinism failure and its structured-output fix.
"""

from __future__ import annotations

from collections import Counter

from shared.agent_sim import (  # noqa: F401  (re-exported for callers)
    Department,
    GuardrailResult,
    MockLLMRouter,
    RouteDecision,
    ground_truth_department,
    guardrails_check,
    parse_department,
)


def naive_route(ticket_text: str, router: MockLLMRouter) -> Department:
    """The original pipeline: free-text LLM answer, then brittle keyword parsing."""
    text = router.classify_freetext(ticket_text, temperature=0.7)
    return parse_department(text)


def robust_route(ticket_text: str, router: MockLLMRouter) -> RouteDecision:
    """Deterministic routing: structured output at temperature=0."""
    return router.classify_structured(ticket_text, temperature=0.0)


def robust_route_dept(ticket_text: str, router: MockLLMRouter) -> Department:
    return robust_route(ticket_text, router).department


def decision_distribution(ticket_text: str, route_fn, n_runs: int = 12) -> Counter:
    """Route the same ticket n times (fresh seeded router) and tally the decisions."""
    router = MockLLMRouter(seed=7)
    return Counter(route_fn(ticket_text, router).value for _ in range(n_runs))
