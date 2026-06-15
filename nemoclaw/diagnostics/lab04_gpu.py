"""Lab 04 — GPU architecture for inference (analytical capacity modeling).

Faithful port of the verified solution logic in
``labs/04-gpu-architecture/build_lab.py``. Pinned by the test suite against the
labs' canonical numbers (8B KV=131072 B/tok, 70B fp16 min_tp=2, chat/8B/H100 -> 1 GPU
bound by KV cache, etc.).
"""

from __future__ import annotations

import math

from pydantic import BaseModel

MEM_UTIL = 0.90  # fraction of HBM usable for weights + KV cache


def dtype_bytes(dtype: str) -> int:
    return 1 if dtype == "fp8" else 2


def peak_flops(gpu: dict, dtype: str) -> float | None:
    """Dense tensor-core FLOP/s for the dtype, or None if unsupported (A100 fp8)."""
    tflops = gpu["fp8_tflops"] if dtype == "fp8" else gpu["fp16_tflops"]
    return None if tflops is None else tflops * 1e12


def mem_bandwidth(gpu: dict) -> float:
    return gpu["mem_bandwidth_gb_s"] * 1e9


def weight_bytes(model: dict, dtype: str) -> float:
    return model["params"] * dtype_bytes(dtype)


class PlannerResult(BaseModel):
    gpu: str
    model: str
    dtype: str
    fits_on_one_gpu: bool
    batch: int
    binding_constraint: str  # "KV cache" or "ITL SLA"
    per_gpu_tokens_per_s: float
    required_tokens_per_s: float
    num_gpus: int
    ttft_ms: float
    ttft_ok: bool


def kv_cache_bytes_per_token(model: dict, dtype: str) -> int:
    """KV cache size for one token: 2 * layers * kv_heads * head_dim * dtype_bytes."""
    return (
        2
        * model["num_layers"]
        * model["kv_heads"]
        * model["head_dim"]
        * dtype_bytes(dtype)
    )


def max_concurrent_sequences(
    gpu: dict, model: dict, dtype: str, context_len: int
) -> tuple[int, int]:
    """(tokens of KV cache, concurrent sequences) that fit after weights."""
    usable = gpu["memory_gb"] * 1e9 * MEM_UTIL
    kv_budget = usable - weight_bytes(model, dtype)
    max_tokens = int(kv_budget // kv_cache_bytes_per_token(model, dtype))
    max_sequences = max_tokens // context_len
    return max_tokens, max_sequences


def decode_tokens_per_s(gpu: dict, model: dict, dtype: str, batch: int) -> float:
    """Decode throughput = min(memory-bound, compute-bound) at this batch size."""
    memory_bound = batch * mem_bandwidth(gpu) / weight_bytes(model, dtype)
    compute_bound = peak_flops(gpu, dtype) / (2 * model["params"])
    return min(memory_bound, compute_bound)


def crossover_batch(gpu: dict, dtype: str) -> float:
    """Batch where decode flips memory->compute bound = the GPU's FLOP:byte ridge."""
    return peak_flops(gpu, dtype) / mem_bandwidth(gpu)


def prefill_ttft_ms(gpu: dict, model: dict, dtype: str, input_tokens: int) -> float:
    """Compute-bound prefill time (TTFT floor) in milliseconds."""
    return 2 * model["params"] * input_tokens / peak_flops(gpu, dtype) * 1000


def min_tp_to_fit(
    model: dict, gpu: dict, dtype: str, n_options=(1, 2, 4, 8)
) -> int | None:
    """Smallest tensor-parallel size whose per-GPU weight shard fits in usable HBM."""
    usable = gpu["memory_gb"] * 1e9 * MEM_UTIL
    for tp in n_options:
        if weight_bytes(model, dtype) / tp < usable:
            return tp
    return None


def capacity_planner(
    workload: dict, gpu: dict, model: dict, dtype: str
) -> PlannerResult:
    """Size a single-GPU deployment for a workload + SLA (tp=1)."""
    context = workload["input_tokens"] + workload["output_tokens"]
    required_tps = workload["target_rps"] * workload["output_tokens"]

    usable = gpu["memory_gb"] * 1e9 * MEM_UTIL
    fits = weight_bytes(model, dtype) < usable

    _, batch_by_kv = max_concurrent_sequences(gpu, model, dtype, context)
    batch_by_itl = int(
        workload["itl_ms_sla"] / 1000 * peak_flops(gpu, dtype) / (2 * model["params"])
    )
    ttft = prefill_ttft_ms(gpu, model, dtype, workload["input_tokens"])

    batch = max(1, min(batch_by_kv, batch_by_itl))
    per_gpu_tps = decode_tokens_per_s(gpu, model, dtype, batch)
    num_gpus = math.ceil(required_tps / per_gpu_tps)

    return PlannerResult(
        gpu=gpu["name"],
        model=model["name"],
        dtype=dtype,
        fits_on_one_gpu=fits,
        batch=batch,
        binding_constraint="KV cache" if batch_by_kv < batch_by_itl else "ITL SLA",
        per_gpu_tokens_per_s=round(per_gpu_tps),
        required_tokens_per_s=required_tps,
        num_gpus=num_gpus,
        ttft_ms=round(ttft, 1),
        ttft_ok=ttft <= workload["ttft_ms_sla"],
    )
