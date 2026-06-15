"""The named real-world scenarios each lab dramatizes.

NeMoClaw frames its troubleshooting around these so a user can say "the air-gapped
NIM won't start" and the agent knows which lab tools to reach for. Each scenario maps
to a lab, the tools that diagnose it, and the fixture(s) that reproduce it.
"""

from __future__ import annotations

from pydantic import BaseModel


class Scenario(BaseModel):
    key: str
    lab: str
    title: str
    brief: str
    tools: list[str]


SCENARIOS: list[Scenario] = [
    Scenario(
        key="rag_quality_drop",
        lab="01-rag-fundamentals",
        title="Startup's basic RAG returns wrong answers",
        brief="A startup shipped RAG with fixed-size chunking and no reranking; "
        "answers are wrong because chunk boundaries split the answer span. "
        "Compare fixed vs sentence-aware chunking on retrieval hit-rate.",
        tools=["rag_eval", "rag_retrieve"],
    ),
    Scenario(
        key="fintech_rerank",
        lab="02-rag-reranking",
        title="Fintech FAQ->PDF migration broke answer quality",
        brief="A Series B fintech's support RAG degraded after migrating its corpus. "
        "The bi-encoder retrieves the right neighborhood but ranks the answer "
        "chunk too low; add a cross-encoder reranking stage and measure MRR.",
        tools=["rag_two_stage", "rag_eval"],
    ),
    Scenario(
        key="nim_wont_start",
        lab="03-nim-deployment",
        title="Air-gapped federal NIM won't come up",
        brief="An ISV deploying a NIM on-prem for an air-gapped federal client hits "
        "startup failures: NGC auth (401/ImagePullBackOff), KV-cache exhaustion "
        "(503s under load), or an offline cache missing the target GPU's engine. "
        "Diagnose the log and pinpoint the fix.",
        tools=[
            "nim_diagnose_log",
            "nim_select_profile",
            "nim_check_airgap_cache",
            "nim_kv_capacity",
        ],
    ),
    Scenario(
        key="gpu_sizing",
        lab="04-gpu-architecture",
        title="Sizing GPU infra for multi-tenant inference",
        brief="An ML engineer must size GPU capacity for a workload + SLA: how many "
        "GPUs, which dtype, does the model fit, and what binds throughput "
        "(KV cache vs ITL SLA). 70B fp16 doesn't fit one 80GB GPU — needs TP.",
        tools=[
            "gpu_capacity_plan",
            "gpu_fits_on_one",
            "gpu_max_concurrency",
            "gpu_kv_per_token",
        ],
    ),
    Scenario(
        key="agent_nondeterminism",
        lab="05-agents-orchestration",
        title="Multi-agent customer service is non-deterministic in prod",
        brief="An ISV's multi-agent support router gives different answers on identical "
        "tickets (free-text output + brittle parsing + temperature). Reproduce the "
        "variance, then fix it with structured output at temperature=0. Plus "
        "guardrails against prompt-injection / data-exfiltration.",
        tools=[
            "agent_route_distribution",
            "agent_route_ticket",
            "agent_guardrails_check",
        ],
    ),
    Scenario(
        key="slo_incident",
        lab="06-mlops-platform",
        title="NIM cluster SLO breach under production load",
        brief="An MLOps team's NIM cluster breached its TTFT p99 SLO during a load "
        "spike, dropping requests with 503s. Find the breach window, prove the "
        "KV-cache-utilization alert leads the breach by ~2 min, and check whether "
        "nightly RAGAS faithfulness tripped the regression gate.",
        tools=[
            "mlops_summarize_incident",
            "mlops_slo_breach",
            "mlops_alert_lead_time",
            "mlops_ragas_gate",
        ],
    ),
]

_BY_LAB = {s.lab: s for s in SCENARIOS}
_BY_KEY = {s.key: s for s in SCENARIOS}


def get_scenario(lab_or_key: str) -> Scenario | None:
    """Look up a scenario by lab folder name (e.g. '03-nim-deployment') or key."""
    return _BY_LAB.get(lab_or_key) or _BY_KEY.get(lab_or_key)
