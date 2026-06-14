# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Generate the Lab 04 capacity-planning inputs: GPU specs, model specs, workloads.

Run with:  uv run python labs/04-gpu-architecture/data/generate.py
       or:  task lab:data:generate LAB=04-gpu-architecture

Lab 04 is analytical — no GPU and no NIM calls. The learner builds a back-of-the-
envelope inference capacity model from three small fact tables:

  gpus.json       Datasheet specs for four inference GPUs (H100 SXM, H100 PCIe,
                  A100 80GB, L40S): HBM capacity + bandwidth, dense FP16/FP8 compute,
                  NVLink bandwidth, and an illustrative cloud $/hr. These are the only
                  numbers the roofline and KV-cache math need.
  models.json     Llama-3.1 8B and 70B shape: parameter count and the attention
                  dimensions (layers, KV heads, head dim) that set KV-cache size.
  workloads.json  Three multi-tenant SLAs (chat, RAG, batch summarization) with a
                  target RPS, input/output token counts, and TTFT + inter-token
                  latency budgets — the demand side of the sizing problem.

All figures are public datasheet / rule-of-thumb values rounded for teaching; dense
(non-sparsity) TFLOPS are used so the roofline arithmetic is honest. The $/hr values
are illustrative, not a quote. Hand-authored so the capacity numbers stay stable.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

# ── GPUs ─────────────────────────────────────────────────────────────────────
# tflops are DENSE (no 2:4 sparsity). bandwidth in GB/s, memory in GB. fp8_tflops
# is null where the SKU has no FP8 tensor cores (A100 / Ampere).
GPUS = [
    {
        "id": "h100_sxm", "name": "H100 80GB SXM5", "form_factor": "SXM5",
        "memory_gb": 80, "mem_bandwidth_gb_s": 3352,
        "fp16_tflops": 989.5, "fp8_tflops": 1978.9,
        "nvlink_gb_s": 900, "usd_per_hr": 3.00,
    },
    {
        "id": "h100_pcie", "name": "H100 80GB PCIe", "form_factor": "PCIe",
        "memory_gb": 80, "mem_bandwidth_gb_s": 2039,
        "fp16_tflops": 756.5, "fp8_tflops": 1513.0,
        "nvlink_gb_s": 600, "usd_per_hr": 2.40,
    },
    {
        "id": "a100_80_sxm", "name": "A100 80GB SXM4", "form_factor": "SXM4",
        "memory_gb": 80, "mem_bandwidth_gb_s": 2039,
        "fp16_tflops": 312.0, "fp8_tflops": None,
        "nvlink_gb_s": 600, "usd_per_hr": 1.80,
    },
    {
        "id": "l40s", "name": "L40S 48GB", "form_factor": "PCIe",
        "memory_gb": 48, "mem_bandwidth_gb_s": 864,
        "fp16_tflops": 362.0, "fp8_tflops": 733.0,
        "nvlink_gb_s": None, "usd_per_hr": 1.00,
    },
]

# ── Models ───────────────────────────────────────────────────────────────────
# Llama-3.1 dense decoders with grouped-query attention (kv_heads < attention
# heads), which is what shrinks the KV cache.
MODELS = [
    {
        "id": "llama31_8b", "name": "Llama-3.1-8B-Instruct",
        "params": 8_030_000_000, "num_layers": 32,
        "hidden": 4096, "attn_heads": 32, "kv_heads": 8, "head_dim": 128,
    },
    {
        "id": "llama31_70b", "name": "Llama-3.1-70B-Instruct",
        "params": 70_600_000_000, "num_layers": 80,
        "hidden": 8192, "attn_heads": 64, "kv_heads": 8, "head_dim": 128,
    },
]

# ── Workloads ────────────────────────────────────────────────────────────────
# The demand side: target throughput + latency SLA per tenant pattern.
WORKLOADS = [
    {
        "id": "chat", "name": "Interactive multi-tenant chat",
        "target_rps": 50, "input_tokens": 1024, "output_tokens": 256,
        "ttft_ms_sla": 500, "itl_ms_sla": 40,
    },
    {
        "id": "rag", "name": "RAG question answering",
        "target_rps": 30, "input_tokens": 3000, "output_tokens": 400,
        "ttft_ms_sla": 800, "itl_ms_sla": 50,
    },
    {
        "id": "summarize", "name": "Long-context batch summarization",
        "target_rps": 8, "input_tokens": 6000, "output_tokens": 800,
        "ttft_ms_sla": 4000, "itl_ms_sla": 80,
    },
]


def _kv_bytes_per_token(model: dict, dtype_bytes: int = 2) -> int:
    """KV cache bytes per token = 2 (K,V) * layers * kv_heads * head_dim * dtype."""
    return 2 * model["num_layers"] * model["kv_heads"] * model["head_dim"] * dtype_bytes


def main() -> None:
    # Precompute the fp16 KV-cache-per-token so the data file is self-documenting;
    # the notebook exercise re-derives it from the attention dims.
    for m in MODELS:
        m["kv_bytes_per_token_fp16"] = _kv_bytes_per_token(m, 2)

    (HERE / "gpus.json").write_text(json.dumps({"gpus": GPUS}, indent=2) + "\n")
    (HERE / "models.json").write_text(json.dumps({"models": MODELS}, indent=2) + "\n")
    (HERE / "workloads.json").write_text(json.dumps({"workloads": WORKLOADS}, indent=2) + "\n")

    print(f"[ok] wrote {len(GPUS)} GPUs       -> {HERE / 'gpus.json'}")
    print(f"[ok] wrote {len(MODELS)} models     -> {HERE / 'models.json'}")
    print(f"[ok] wrote {len(WORKLOADS)} workloads  -> {HERE / 'workloads.json'}")


if __name__ == "__main__":
    main()
