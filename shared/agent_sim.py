"""Deterministic, offline stand-ins for an LLM-NIM agent stack (Lab 05).

Lab 05 teaches agent *orchestration* — non-determinism, structured output,
temperature, stateful routing, async tool calls, tracing, and guardrails. To make
those lessons reproducible and runnable with no GPU, no network, and no
`NVIDIA_API_KEY`, the LLM and the backend tools are simulated here with controllable,
seeded behaviour. The LangGraph wiring, OpenTelemetry tracing, and asyncio in the
notebook are the *real* libraries; only the model and the data backend are mocked.

In production you would swap `MockLLMRouter` for a real LLM NIM call — e.g.
`langchain_openai.ChatOpenAI(base_url=os.environ["NIM_LLM_URL"], ...)
.with_structured_output(RouteDecision)` — and the rest of the graph is unchanged.

Why a simulation instead of the real API Catalog: the whole point of the lab is to
*reproduce* run-to-run variance on demand and then prove it gone. A seeded mock makes
"flaky" deterministic, so a test can assert the fix actually worked.
"""
from __future__ import annotations

import asyncio
import random
from enum import Enum

from pydantic import BaseModel, Field


class Department(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class RouteDecision(BaseModel):
    """Structured router output — the schema a real NIM would be constrained to."""
    department: Department
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class GuardrailResult(BaseModel):
    allowed: bool
    category: str          # "ok" | "prompt_injection" | "data_exfiltration"
    reason: str


# Keyword signals per department. The "ground truth" route is the argmax of these
# hit counts; ties and near-ties are what make some tickets genuinely ambiguous.
_SIGNALS: dict[Department, tuple[str, ...]] = {
    Department.BILLING: ("charge", "invoice", "refund", "payment", "billed", "subscription", "price", "fee"),
    Department.TECHNICAL: ("error", "500", "timeout", "api", "crash", "bug", "latency", "endpoint", "fails"),
    Department.ACCOUNT: ("password", "login", "locked", "email", "2fa", "seat", "permission", "access"),
    Department.GENERAL: ("question", "how do i", "documentation", "feedback", "hello"),
}

# Surface phrasings the mock LLM emits in free-text mode. Some (e.g. "payments team",
# "infra") deliberately dodge the brittle keyword parser to create parse errors.
_PHRASINGS: dict[Department, tuple[str, ...]] = {
    Department.BILLING: ("Billing", "This is a billing issue.", "Route to the payments team.", "Looks like a charges question."),
    Department.TECHNICAL: ("Technical", "Send to engineering.", "This is a platform/infra problem.", "Technical support needed."),
    Department.ACCOUNT: ("Account", "Account management.", "This is an identity/access request.", "Send to the accounts team."),
    Department.GENERAL: ("General", "General inquiry.", "Not sure — general support.", "Route to the front desk."),
}

# Substrings the brittle parser looks for. Note it has no entry for "payments",
# "infra", "identity", "front desk" — so those phrasings mis-parse to GENERAL.
_PARSE_KEYS: dict[Department, tuple[str, ...]] = {
    Department.BILLING: ("billing", "charges"),
    Department.TECHNICAL: ("technical", "engineering"),
    Department.ACCOUNT: ("account",),
    Department.GENERAL: ("general",),
}


def department_scores(text: str) -> dict[Department, int]:
    """Keyword-hit score per department (the simulated model's latent signal)."""
    t = text.lower()
    return {d: sum(t.count(k) for k in kws) for d, kws in _SIGNALS.items()}


def ground_truth_department(text: str) -> Department:
    """Deterministic 'correct' route: the argmax of department signals."""
    scores = department_scores(text)
    return max(scores, key=lambda d: (scores[d], -list(Department).index(d)))


def _ranked(text: str) -> list[Department]:
    scores = department_scores(text)
    return sorted(Department, key=lambda d: (-scores[d], list(Department).index(d)))


class MockLLMRouter:
    """A seeded stand-in for an LLM classifying a ticket into a department.

    The two knobs mirror a real model:

    - ``temperature`` — at 0 the router is greedy (always the top-ranked department,
      no RNG draws, fully reproducible). Above 0 it *samples*: with some probability
      it returns the runner-up instead, which is genuine run-to-run variance.
    - output mode — ``classify_freetext`` returns prose you must parse (brittle),
      while ``classify_structured`` returns a validated :class:`RouteDecision` (no
      parsing). Structured output removes *parse* errors; ``temperature=0`` removes
      *sampling* variance. You need both to be deterministic.
    """

    def __init__(self, seed: int = 7, runner_up_prob: float = 0.35):
        self._rng = random.Random(seed)
        self._runner_up_prob = runner_up_prob

    def _sample_department(self, text: str, temperature: float) -> Department:
        ranked = _ranked(text)
        if temperature <= 0:
            return ranked[0]                       # greedy: no RNG, deterministic
        scores = department_scores(text)
        top, runner_up = ranked[0], ranked[1]
        # Only genuinely ambiguous tickets waver: flip probability scales with how
        # close the runner-up's signal is to the top's (0 if the runner-up is empty).
        if scores[runner_up] == 0:
            return top
        closeness = scores[runner_up] / scores[top]
        p_flip = self._runner_up_prob * closeness * min(temperature / 0.7, 1.0)
        return runner_up if self._rng.random() < p_flip else top

    def classify_freetext(self, text: str, temperature: float = 0.7) -> str:
        """Return a free-text department guess (must be parsed downstream)."""
        dept = self._sample_department(text, temperature)
        if temperature <= 0:
            return _PHRASINGS[dept][0]             # canonical phrasing, deterministic
        return self._rng.choice(_PHRASINGS[dept])  # variable surface form

    def classify_structured(self, text: str, temperature: float = 0.0) -> RouteDecision:
        """Return a validated RouteDecision (the structured-output path)."""
        dept = self._sample_department(text, temperature)
        scores = department_scores(text)
        total = sum(scores.values()) or 1
        confidence = round(scores[dept] / total, 3) if scores[dept] else 0.25
        return RouteDecision(
            department=dept,
            confidence=confidence,
            reason=f"matched {scores[dept]} {dept.value} signal(s)",
        )


def parse_department(freetext: str) -> Department:
    """Brittle parser for free-text router output; unknown phrasings fall to GENERAL."""
    t = freetext.lower()
    for dept, keys in _PARSE_KEYS.items():
        if any(k in t for k in keys):
            return dept
    return Department.GENERAL          # silent fallback — the source of mis-routes


# ── Guardrails (NeMo Guardrails NIM stand-in) ────────────────────────────────
_INJECTION = ("ignore previous instructions", "ignore all prior", "system prompt", "you are now", "disregard the above")
_EXFIL = ("all customers", "every customer", "dump the database", "other users", "list all accounts", "everyone's")


def guardrails_check(text: str) -> GuardrailResult:
    """Block prompt-injection and data-exfiltration attempts before the agent runs."""
    t = text.lower()
    if any(p in t for p in _INJECTION):
        return GuardrailResult(allowed=False, category="prompt_injection",
                               reason="input attempts to override system instructions")
    if any(p in t for p in _EXFIL):
        return GuardrailResult(allowed=False, category="data_exfiltration",
                               reason="input requests data beyond the requesting account")
    return GuardrailResult(allowed=True, category="ok", reason="passed input guardrails")


# ── Async backend tools (mock data plane) ────────────────────────────────────
# Each simulates a remote lookup with latency, so concurrent (gather) vs sequential
# wall-time is observable. The backend is injected so the notebook/tests stay pure.
TOOL_LATENCY_S = 0.05


async def fetch_account(account_id: str, backend: dict) -> dict:
    await asyncio.sleep(TOOL_LATENCY_S)
    return backend["accounts"].get(account_id, {"account_id": account_id, "status": "unknown"})


async def fetch_invoices(account_id: str, backend: dict) -> dict:
    await asyncio.sleep(TOOL_LATENCY_S)
    return {"account_id": account_id, "invoices": backend["invoices"].get(account_id, [])}


async def fetch_usage(account_id: str, backend: dict) -> dict:
    await asyncio.sleep(TOOL_LATENCY_S)
    return {"account_id": account_id, "usage": backend["usage"].get(account_id, {})}


TOOLS = (fetch_account, fetch_invoices, fetch_usage)
