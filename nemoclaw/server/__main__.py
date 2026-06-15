"""NeMoClaw MCP server — exposes the lab diagnostic logic as MCP tools.

Run standalone (e.g. for the MCP Inspector or Claude Desktop):

    uv run python -m nemoclaw.server      # stdio transport

The NeMo Agent Toolkit workflow in ``nemoclaw/agent`` launches this same command as
an MCP client and drives the tools with an LLM NIM. Tool docstrings are written for
the *model* to read — they say when to reach for each tool and what the inputs mean.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from nemoclaw import scenarios
from nemoclaw.diagnostics import fixtures as fx
from nemoclaw.diagnostics import lab03_nim as l3
from nemoclaw.diagnostics import lab04_gpu as l4
from nemoclaw.diagnostics import lab05_agents as l5
from nemoclaw.diagnostics import lab06_mlops as l6

mcp = FastMCP(
    "nemoclaw",
    instructions=(
        "NeMoClaw troubleshoots the NVIDIA Partner Lab scenarios: RAG retrieval "
        "quality (Labs 01-02), NIM deployment failures (Lab 03), GPU capacity "
        "planning (Lab 04), agent orchestration non-determinism (Lab 05), and NIM "
        "cluster SLO incidents (Lab 06). Start with list_scenarios to map a user's "
        "problem to the right tools."
    ),
)


def _resolve(table: dict[str, dict], key: str, kind: str) -> dict:
    if key not in table:
        raise ValueError(f"unknown {kind} {key!r}; choose one of {sorted(table)}")
    return table[key]


# ── Scenario discovery ───────────────────────────────────────────────────────
@mcp.tool()
def list_scenarios() -> list[dict[str, Any]]:
    """List the named real-world troubleshooting scenarios NeMoClaw can handle.

    Call this first to map a user's described problem to a lab and its tools.
    Returns each scenario's key, lab, title, one-paragraph brief, and the tool names
    that diagnose it.
    """
    return [s.model_dump() for s in scenarios.SCENARIOS]


@mcp.tool()
def describe_scenario(lab_or_key: str) -> dict[str, Any]:
    """Get the full brief for one scenario, by its key (e.g. 'nim_wont_start') or
    lab folder name (e.g. '03-nim-deployment')."""
    s = scenarios.get_scenario(lab_or_key)
    if s is None:
        return {
            "error": f"no scenario for {lab_or_key!r}",
            "available": [sc.key for sc in scenarios.SCENARIOS],
        }
    return s.model_dump()


# ── Lab 03 — NIM deployment & troubleshooting ────────────────────────────────
@mcp.tool()
def nim_diagnose_log(log_name: str = "", log_text: str = "") -> dict[str, Any]:
    """Diagnose a NIM startup/runtime log against the known-failure playbook.

    Pass either `log_name` (a built-in sample: 'startup_healthy', 'ngc_auth_failure',
    'kv_cache_exhaustion', 'airgapped_offline') OR paste raw log lines as `log_text`.
    Returns the matched failure key, root cause, fix, and the evidence line. For a
    healthy startup it also returns the parsed startup report (GPU, profile, KV-cache
    token budget, time-to-ready in seconds).
    """
    if log_name:
        logs = fx.nim_logs()
        if log_name not in logs:
            return {
                "error": f"unknown log_name {log_name!r}",
                "available": sorted(logs),
            }
        log = logs[log_name]
    elif log_text:
        log = log_text
    else:
        return {
            "error": "provide log_name or log_text",
            "available_logs": sorted(fx.nim_logs()),
        }

    dx = l3.diagnose(log)
    out: dict[str, Any] = {"diagnosis": dx.model_dump()}
    if dx.failure == "unknown" and "Uvicorn running" in log:
        try:
            out["startup_report"] = l3.parse_startup(log).model_dump()
        except Exception:  # noqa: BLE001 - best-effort enrichment only
            pass
    return out


@mcp.tool()
def nim_select_profile(
    gpu: str, count: int = 1, compute_capability: str = ""
) -> dict[str, Any]:
    """Reproduce the NIM ngc_injector's TRT-LLM profile auto-selection for a GPU.

    `gpu` is a family like 'h100', 'a100', or 'l40s'; `count` is how many GPUs;
    `compute_capability` (e.g. '9.0' for H100, '8.0' for A100, '8.9' for L40S)
    decides fp8 eligibility — if omitted, a sensible default for the family is used.
    Returns the selected profile id, backend, precision, tp, and why. Use this to
    explain why a dev box (H100/fp8) and an air-gapped target (A100/fp16) resolve to
    DIFFERENT engines — the root cause of the air-gap cache miss.
    """
    cc_default = {"h100": "9.0", "a100": "8.0", "l40s": "8.9"}
    cc = compute_capability or cc_default.get(gpu.lower(), "9.0")
    detected = {"gpu": gpu.lower(), "compute_capability": cc, "count": count}
    choice = l3.select_profile(detected, fx.nim_profiles())
    if choice is None:
        return {"error": f"no compatible profile for {detected}"}
    return {"detected": detected, "choice": choice.model_dump()}


@mcp.tool()
def nim_kv_capacity(
    kv_cache_tokens: int, tokens_per_request: int = 4096, offered_concurrency: int = 0
) -> dict[str, Any]:
    """How many full-context requests a NIM's KV cache can hold at once.

    Integer-divides the KV-cache token budget by the per-request reservation
    (max_model_len, default 4096). If `offered_concurrency` is given, also reports how
    many requests exceed the ceiling (the ones that get 503s). For the Lab 03 scenario
    use kv_cache_tokens=117440 -> ceiling 28; an offered load of 64 means 36 over.
    """
    ceiling = l3.max_concurrent_requests(kv_cache_tokens, tokens_per_request)
    out: dict[str, Any] = {
        "kv_cache_tokens": kv_cache_tokens,
        "tokens_per_request": tokens_per_request,
        "max_concurrent_requests": ceiling,
    }
    if offered_concurrency:
        out["offered_concurrency"] = offered_concurrency
        out["over_ceiling"] = max(0, offered_concurrency - ceiling)
    return out


@mcp.tool()
def nim_check_airgap_cache() -> dict[str, Any]:
    """Check whether the air-gapped offline cache contains the engine the target GPU
    needs. Reproduces the Lab 03 federal scenario: the cache was prepared on an H100
    dev box (fp8 profile) but the A100 target resolves to a different fp16 profile, so
    its engine artifacts are missing. Returns the target profile, what the cache was
    prepared on, and the exact missing artifact paths.
    """
    dep = fx.nim_deployment()
    profiles = fx.nim_profiles()
    cache = fx.nim_cache_manifest()
    target = l3.select_profile(dep["airgapped_target_gpu"], profiles)
    required = next(p["artifacts"] for p in profiles if p["id"] == target.profile_id)
    missing = l3.missing_artifacts(required, cache["present_artifacts"])
    return {
        "cache_prepared_on": cache["prepared_on"],
        "target_gpu": dep["airgapped_target_gpu"],
        "target_profile": target.model_dump(),
        "missing_artifacts": missing,
    }


# ── Lab 04 — GPU architecture / capacity planning ────────────────────────────
@mcp.tool()
def gpu_kv_per_token(model: str = "llama31_8b", dtype: str = "fp16") -> dict[str, Any]:
    """KV-cache bytes per token for a model (2 * layers * kv_heads * head_dim * dtype).

    `model` is 'llama31_8b' or 'llama31_70b'; `dtype` is 'fp16' or 'fp8'. This per-token
    cost drives the concurrency ceiling. 8B fp16 = 131072 B/token (128 KiB); the 70B is
    2.5x deeper so 320 KiB.
    """
    m = _resolve(fx.models(), model, "model")
    b = l4.kv_cache_bytes_per_token(m, dtype)
    return {
        "model": m["name"],
        "dtype": dtype,
        "bytes_per_token": b,
        "kib_per_token": b / 1024,
    }


@mcp.tool()
def gpu_max_concurrency(
    gpu: str = "h100_sxm",
    model: str = "llama31_8b",
    dtype: str = "fp16",
    context_len: int = 1280,
) -> dict[str, Any]:
    """KV-cache token budget and max concurrent sequences after model weights load.

    `gpu` ∈ {h100_sxm, h100_pcie, a100_80_sxm, l40s}; `context_len` is input+output
    tokens per sequence. Returns the KV token budget and how many sequences of that
    context fit. (90% of HBM is usable; the rest is overhead.)
    """
    g = _resolve(fx.gpus(), gpu, "gpu")
    m = _resolve(fx.models(), model, "model")
    max_tokens, max_seqs = l4.max_concurrent_sequences(g, m, dtype, context_len)
    return {
        "gpu": g["name"],
        "model": m["name"],
        "dtype": dtype,
        "context_len": context_len,
        "weights_gb": round(l4.weight_bytes(m, dtype) / 1e9, 1),
        "kv_tokens": max_tokens,
        "max_sequences": max_seqs,
    }


@mcp.tool()
def gpu_fits_on_one(
    model: str = "llama31_70b", gpu: str = "a100_80_sxm", dtype: str = "fp16"
) -> dict[str, Any]:
    """Does the model's weights fit on ONE GPU, and if not, the minimum tensor-parallel
    size that does. 70B fp16 weights are 141GB and do NOT fit one 80GB GPU (needs tp=2
    + NVLink); 70B fp8 (71GB) just fits. Returns weights_gb, usable_gb, fits, min_tp.
    """
    g = _resolve(fx.gpus(), gpu, "gpu")
    m = _resolve(fx.models(), model, "model")
    usable = g["memory_gb"] * 1e9 * l4.MEM_UTIL
    weights = l4.weight_bytes(m, dtype)
    return {
        "model": m["name"],
        "gpu": g["name"],
        "dtype": dtype,
        "weights_gb": round(weights / 1e9, 1),
        "usable_gb": round(usable / 1e9, 1),
        "fits_on_one_gpu": weights < usable,
        "min_tp_to_fit": l4.min_tp_to_fit(m, g, dtype),
    }


@mcp.tool()
def gpu_decode_throughput(
    gpu: str = "h100_sxm",
    model: str = "llama31_8b",
    dtype: str = "fp16",
    batch: int = 1,
) -> dict[str, Any]:
    """Decode (token-generation) throughput at a batch size, and the memory->compute
    crossover batch (the GPU's FLOP:byte ridge). At batch=1 decode is memory-bandwidth
    bound, so buying more FLOPs doesn't help until you batch past the crossover.
    """
    g = _resolve(fx.gpus(), gpu, "gpu")
    m = _resolve(fx.models(), model, "model")
    tps = l4.decode_tokens_per_s(g, m, dtype, batch)
    return {
        "gpu": g["name"],
        "model": m["name"],
        "dtype": dtype,
        "batch": batch,
        "tokens_per_s": round(tps),
        "crossover_batch": round(l4.crossover_batch(g, dtype)),
    }


@mcp.tool()
def gpu_capacity_plan(
    workload: str = "chat",
    gpu: str = "h100_sxm",
    model: str = "llama31_8b",
    dtype: str = "fp16",
) -> dict[str, Any]:
    """Size a single-GPU-type deployment for a workload + SLA. Capstone planner.

    `workload` ∈ {chat, rag, summarize} (each carries target_rps, token sizes, TTFT/ITL
    SLAs). Returns whether the model fits, the serving batch and what binds it (KV cache
    vs ITL SLA), per-GPU and required tokens/sec, the number of GPUs needed, and whether
    the prefill TTFT meets the SLA. E.g. chat + 8B + H100 fp16 -> 1 GPU, bound by KV cache.
    """
    w = _resolve(fx.workloads(), workload, "workload")
    g = _resolve(fx.gpus(), gpu, "gpu")
    m = _resolve(fx.models(), model, "model")
    return l4.capacity_planner(w, g, m, dtype).model_dump()


# ── Lab 06 — MLOps observability / SLO ───────────────────────────────────────
@mcp.tool()
def mlops_summarize_incident() -> dict[str, Any]:
    """One-shot summary of the Lab 06 NIM-cluster load-spike incident from the 30-min
    simulated Prometheus scrape: peak RPS / tokens-per-sec / failure rate, TTFT p99
    distribution vs SLO, the SLO breach window, requests dropped, and the KV-cache alert
    lead time. Use this to brief the on-call before drilling into specific tools.
    """
    m = fx.metrics()
    s, slo, step = m["samples"], m["slo"], m["step_s"]
    rps = l6.counter_rate(s, "request_success_total", step)
    tok = l6.counter_rate(s, "generation_tokens_total", step)
    fail = l6.counter_rate(s, "request_failure_total", step)
    ttft99 = [x["ttft_p99_ms"] for x in s]
    window = l6.slo_breach_window(s, "ttft_p99_ms", slo["ttft_p99_ms"])
    warn_t, breach_t, lead = l6.alert_lead_time(
        s, "kv_cache_util", slo["kv_warn"], "ttft_p99_ms", slo["ttft_p99_ms"]
    )
    dropped = s[-1]["request_failure_total"] - s[0]["request_failure_total"]
    return {
        "slo": slo,
        "peak_rps": round(max(rps)),
        "baseline_rps": round(rps[0]),
        "peak_tokens_per_s": round(max(tok)),
        "peak_failures_per_s": round(max(fail), 1),
        "ttft_p99_ms": {
            "p50": l6.percentile(ttft99, 0.50),
            "p99": l6.percentile(ttft99, 0.99),
        },
        "breach_window_s": list(window) if window else None,
        "requests_dropped_in_window": dropped,
        "kv_warn_at_s": warn_t,
        "ttft_breach_at_s": breach_t,
        "alert_lead_time_s": lead,
        "sparklines": {
            "rps": l6.sparkline(rps),
            "kv_cache_util": l6.sparkline([x["kv_cache_util"] for x in s]),
            "ttft_p99": l6.sparkline(ttft99),
        },
    }


@mcp.tool()
def mlops_slo_breach(
    metric_key: str = "ttft_p99_ms", threshold: float = 0.0
) -> dict[str, Any]:
    """First and last timestamps (seconds) where a metric exceeds a threshold during the
    incident. `metric_key` e.g. 'ttft_p99_ms', 'itl_p99_ms', 'kv_cache_util'. If
    `threshold` is 0 the matching SLO threshold is used (ttft_p99_ms->500, itl_p99_ms->50,
    kv_cache_util->kv_warn 0.9). Returns the breach window and its duration.
    """
    m = fx.metrics()
    s, slo = m["samples"], m["slo"]
    if not threshold:
        threshold = {
            "ttft_p99_ms": slo["ttft_p99_ms"],
            "itl_p99_ms": slo["itl_p99_ms"],
            "kv_cache_util": slo["kv_warn"],
        }.get(metric_key, 0.0)
    window = l6.slo_breach_window(s, metric_key, threshold)
    if window is None:
        return {"metric_key": metric_key, "threshold": threshold, "breached": False}
    return {
        "metric_key": metric_key,
        "threshold": threshold,
        "breached": True,
        "breach_start_s": window[0],
        "breach_end_s": window[1],
        "duration_s": window[1] - window[0],
    }


@mcp.tool()
def mlops_alert_lead_time() -> dict[str, Any]:
    """Prove the KV-cache-utilization warning leads the TTFT p99 SLO breach. Returns when
    KV util first crossed the warn line, when TTFT p99 first breached the SLO, and the
    lead time between them (120s for this incident — enough runway to autoscale or shed
    load). This is why you page on the LEADING indicator, not on TTFT itself.
    """
    m = fx.metrics()
    s, slo = m["samples"], m["slo"]
    warn_t, breach_t, lead = l6.alert_lead_time(
        s, "kv_cache_util", slo["kv_warn"], "ttft_p99_ms", slo["ttft_p99_ms"]
    )
    return {
        "kv_warn_at_s": warn_t,
        "ttft_breach_at_s": breach_t,
        "alert_lead_time_s": lead,
        "kv_warn_threshold": slo["kv_warn"],
        "ttft_slo_ms": slo["ttft_p99_ms"],
    }


@mcp.tool()
def mlops_ragas_gate(gate: float = 0.0) -> dict[str, Any]:
    """Check nightly RAGAS faithfulness against the regression gate. Returns the first
    eval run that dropped below the gate (which should trigger a NeMo Customizer
    retrain), or null if all runs pass. If `gate` is 0 the dataset's configured gate is
    used. For this dataset the gate trips at run #10.
    """
    ev = fx.eval_runs()
    g = gate or ev["faithfulness_gate"]
    failing = l6.regression_gate(ev["runs"], g)
    baseline = ev["runs"][0]["faithfulness"]
    return {
        "gate": g,
        "baseline_faithfulness": baseline,
        "failing_run": failing,
        "drift_from_baseline": round(baseline - failing["faithfulness"], 3)
        if failing
        else 0.0,
    }


# ── Lab 05 — agents & orchestration ──────────────────────────────────────────
@mcp.tool()
def agent_route_ticket(ticket_text: str, mode: str = "robust") -> dict[str, Any]:
    """Route a support ticket to a department (billing/technical/account/general).

    `mode='naive'` uses free-text LLM output + brittle keyword parsing at temperature
    0.7 (the buggy production pipeline — can mis-parse and waver). `mode='robust'` uses
    structured output at temperature=0 (deterministic, the fix). Returns the department
    and, for robust mode, the confidence and reason, plus the ground-truth route.
    """
    router = l5.MockLLMRouter(seed=7)
    truth = l5.ground_truth_department(ticket_text).value
    if mode == "naive":
        dept = l5.naive_route(ticket_text, router).value
        return {"mode": "naive", "department": dept, "ground_truth": truth}
    decision = l5.robust_route(ticket_text, router)
    return {"mode": "robust", **decision.model_dump(), "ground_truth": truth}


@mcp.tool()
def agent_route_distribution(
    ticket_text: str, mode: str = "naive", n_runs: int = 12
) -> dict[str, Any]:
    """Route the SAME ticket n times and tally the decisions — exposes non-determinism.

    `mode='naive'` typically yields multiple distinct decisions on ambiguous tickets
    (the prod bug); `mode='robust'` collapses to a single decision. Returns the decision
    counts and how many distinct outcomes occurred (>1 means non-deterministic).
    """
    route_fn = l5.naive_route if mode == "naive" else l5.robust_route_dept
    dist = l5.decision_distribution(ticket_text, route_fn, n_runs=n_runs)
    return {
        "mode": mode,
        "n_runs": n_runs,
        "distribution": dict(dist),
        "distinct_decisions": len(dist),
        "deterministic": len(dist) == 1,
    }


@mcp.tool()
def agent_guardrails_check(text: str) -> dict[str, Any]:
    """Run input guardrails (NeMo Guardrails NIM stand-in) on a user message. Blocks
    prompt-injection ('ignore previous instructions', 'system prompt', ...) and
    data-exfiltration ('all customers', 'dump the database', ...). Returns allowed,
    category, and reason.
    """
    return l5.guardrails_check(text).model_dump()


# ── Lab 01/02 — RAG retrieval quality (lazy: loads local models on first call) ──
@mcp.tool()
def rag_retrieve(
    query: str, lab: str = "01", k: int = 5, chunker: str = "sentence"
) -> dict[str, Any]:
    """Retrieve the top-k chunks for a query from a lab corpus (bi-encoder, cosine).

    `lab`='01' (NVIDIA dev-blog corpus) or '02' (banking-policy corpus). `chunker` is
    'fixed' (140-char windows, the naive baseline) or 'sentence' (≤320-char sentence
    groups). Returns each hit's doc_id, cosine score, and text. NOTE: first call loads a
    local embedding model (~10-20s).
    """
    from nemoclaw.diagnostics import lab0102_rag as rag

    chunks = rag.chunk_corpus(fx.rag_corpus(lab), rag._make_chunker(chunker))
    index = rag.build_index(chunks)
    hits = rag.retrieve(index, query, k)
    return {
        "lab": lab,
        "chunker": chunker,
        "query": query,
        "results": [
            {
                "doc_id": h["doc_id"],
                "score": round(h["score"], 3),
                "text": h["text"][:200],
            }
            for h in hits
        ],
    }


@mcp.tool()
def rag_eval(lab: str = "01", chunker: str = "sentence", k: int = 3) -> dict[str, Any]:
    """Measure retrieval hit-rate@k over a lab's labeled eval set, for a chunking
    strategy. Use this to prove the fix: on Lab 01, fixed 140-char chunking hits ~50%
    while sentence-aware chunking hits ~75% (the chunk-boundary-split answers recover).
    Returns hit_rate and the ids of missed questions. First call loads a local model.
    """
    from nemoclaw.diagnostics import lab0102_rag as rag

    chunks = rag.chunk_corpus(fx.rag_corpus(lab), rag._make_chunker(chunker))
    index = rag.build_index(chunks)
    hr, misses = rag.hit_rate(index, fx.rag_eval(lab), k)
    return {
        "lab": lab,
        "chunker": chunker,
        "k": k,
        "hit_rate": round(hr, 3),
        "missed_question_ids": misses,
    }


@mcp.tool()
def rag_two_stage(
    query: str, lab: str = "02", k1: int = 20, k2: int = 5, chunker: str = "sentence"
) -> dict[str, Any]:
    """Two-stage retrieval: bi-encoder retrieves k1 candidates, cross-encoder reranks to
    k2. Shows how reranking promotes the answer chunk — each result carries its
    embed_rank, rerank_rank, and rank_delta (positive = promoted). Best on the Lab 02
    banking corpus. First call loads local embed + cross-encoder models.
    """
    from nemoclaw.diagnostics import lab0102_rag as rag

    chunks = rag.chunk_corpus(fx.rag_corpus(lab), rag._make_chunker(chunker))
    index = rag.build_index(chunks)
    ranked = rag.two_stage(index, query, k1, k2)
    return {
        "lab": lab,
        "query": query,
        "k1": k1,
        "k2": k2,
        "results": [
            {**r, "text": r["text"][:200], "rerank_logit": round(r["rerank_logit"], 3)}
            for r in ranked
        ],
    }


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
