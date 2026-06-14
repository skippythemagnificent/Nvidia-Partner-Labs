"""Validation for Lab 04.

Reimplements the reference capacity model (mirroring the notebook) and asserts the
canonical numbers and roofline relationships, plus data invariants and end-to-end
execution of the solution notebook. The learner `lab.ipynb` is not executed here —
its stubs fail by design.
"""
from __future__ import annotations

import json
import math

import pytest

MEM_UTIL = 0.90


# ── reference model (mirrors the notebook) ───────────────────────────────────


def _dtype_bytes(dtype):
    return 1 if dtype == "fp8" else 2


def _peak_flops(gpu, dtype):
    t = gpu["fp8_tflops"] if dtype == "fp8" else gpu["fp16_tflops"]
    return None if t is None else t * 1e12


def _bw(gpu):
    return gpu["mem_bandwidth_gb_s"] * 1e9


def _weight_bytes(model, dtype):
    return model["params"] * _dtype_bytes(dtype)


def kv_per_token(model, dtype):
    return 2 * model["num_layers"] * model["kv_heads"] * model["head_dim"] * _dtype_bytes(dtype)


def max_sequences(gpu, model, dtype, ctx):
    usable = gpu["memory_gb"] * 1e9 * MEM_UTIL
    kv_budget = usable - _weight_bytes(model, dtype)
    max_tokens = int(kv_budget // kv_per_token(model, dtype))
    return max_tokens, max_tokens // ctx


def decode_tps(gpu, model, dtype, batch):
    mem = batch * _bw(gpu) / _weight_bytes(model, dtype)
    comp = _peak_flops(gpu, dtype) / (2 * model["params"])
    return min(mem, comp)


def crossover_batch(gpu, dtype):
    return _peak_flops(gpu, dtype) / _bw(gpu)


def prefill_ttft_ms(gpu, model, dtype, input_tokens):
    return 2 * model["params"] * input_tokens / _peak_flops(gpu, dtype) * 1000


def plan_num_gpus(workload, gpu, model, dtype):
    ctx = workload["input_tokens"] + workload["output_tokens"]
    required = workload["target_rps"] * workload["output_tokens"]
    _, b_kv = max_sequences(gpu, model, dtype, ctx)
    b_itl = int(workload["itl_ms_sla"] / 1000 * _peak_flops(gpu, dtype) / (2 * model["params"]))
    batch = max(1, min(b_kv, b_itl))
    per_gpu = decode_tps(gpu, model, dtype, batch)
    return batch, math.ceil(required / per_gpu), ("KV cache" if b_kv < b_itl else "ITL SLA")


# ── data invariants ──────────────────────────────────────────────────────────


def test_gpu_table_shape(gpus):
    assert {"h100_sxm", "h100_pcie", "a100_80_sxm", "l40s"} <= set(gpus)
    a100 = gpus["a100_80_sxm"]
    assert a100["fp8_tflops"] is None, "A100 (Ampere) has no FP8 tensor cores"
    for g in gpus.values():
        assert g["fp16_tflops"] > 0 and g["mem_bandwidth_gb_s"] > 0 and g["memory_gb"] > 0


def test_model_table_shape(models):
    for m in models.values():
        assert {"params", "num_layers", "kv_heads", "head_dim"} <= set(m)
        assert m["kv_heads"] < m["attn_heads"], "GQA: fewer KV heads than attention heads"


# ── KV cache math ────────────────────────────────────────────────────────────


def test_kv_per_token_known_values(models):
    assert kv_per_token(models["llama31_8b"], "fp16") == 131072    # 128 KiB
    assert kv_per_token(models["llama31_70b"], "fp16") == 327680   # 320 KiB
    assert kv_per_token(models["llama31_8b"], "fp8") == 65536      # halves at fp8


def test_concurrency_ceiling(gpus, models):
    _, seqs_80 = max_sequences(gpus["h100_sxm"], models["llama31_8b"], "fp16", 1280)
    _, seqs_48 = max_sequences(gpus["l40s"], models["llama31_8b"], "fp16", 1280)
    assert seqs_80 == 333
    assert seqs_48 == 161
    assert seqs_48 < seqs_80, "smaller HBM holds fewer sequences"


# ── roofline ─────────────────────────────────────────────────────────────────


def test_crossover_equals_flop_byte_ridge(gpus):
    """The memory->compute crossover batch equals the GPU's FLOP:byte ratio."""
    g = gpus["h100_sxm"]
    assert round(crossover_batch(g, "fp16")) == 295
    assert round(crossover_batch(g, "fp16")) == round(g["fp16_tflops"] * 1e12 / _bw(g))


def test_decode_saturates_at_compute_ceiling(gpus, models):
    g, m = gpus["h100_sxm"], models["llama31_8b"]
    ceiling = _peak_flops(g, "fp16") / (2 * m["params"])
    assert decode_tps(g, m, "fp16", 1) < decode_tps(g, m, "fp16", 32) < ceiling   # memory-bound, linear
    assert decode_tps(g, m, "fp16", 100000) == pytest.approx(ceiling)             # saturated


def test_single_user_decode_tracks_bandwidth_not_flops(gpus, models):
    """At batch=1 the H100/A100 ratio is bandwidth (~1.64x), not FLOPs (~3.17x)."""
    a1 = decode_tps(gpus["a100_80_sxm"], models["llama31_8b"], "fp16", 1)
    h1 = decode_tps(gpus["h100_sxm"], models["llama31_8b"], "fp16", 1)
    bw_ratio = gpus["h100_sxm"]["mem_bandwidth_gb_s"] / gpus["a100_80_sxm"]["mem_bandwidth_gb_s"]
    assert h1 / a1 == pytest.approx(bw_ratio, rel=1e-6)
    assert round(h1 / a1, 2) == 1.64


# ── prefill ──────────────────────────────────────────────────────────────────


def test_prefill_tracks_flops(gpus, models):
    """Prefill is compute-bound: H100/A100 TTFT ratio is the FLOPs ratio (~3.17x)."""
    h = prefill_ttft_ms(gpus["h100_sxm"], models["llama31_8b"], "fp16", 1024)
    a = prefill_ttft_ms(gpus["a100_80_sxm"], models["llama31_8b"], "fp16", 1024)
    assert round(h, 1) == 16.6
    flops_ratio = gpus["h100_sxm"]["fp16_tflops"] / gpus["a100_80_sxm"]["fp16_tflops"]
    assert a / h == pytest.approx(flops_ratio, rel=1e-6)


# ── capacity planner + weight-fit ────────────────────────────────────────────


def test_planner_chat_8b(gpus, models, workloads):
    chat = workloads["chat"]
    batch, ngpu, binding = plan_num_gpus(chat, gpus["h100_sxm"], models["llama31_8b"], "fp16")
    assert (batch, ngpu, binding) == (333, 1, "KV cache")
    _, ngpu_l40s, _ = plan_num_gpus(chat, gpus["l40s"], models["llama31_8b"], "fp16")
    assert ngpu_l40s == 2, "the 48GB L40S needs two GPUs for 50 rps"


def test_70b_does_not_fit_one_gpu_fp16(gpus, models):
    usable = gpus["h100_sxm"]["memory_gb"] * 1e9 * MEM_UTIL
    assert _weight_bytes(models["llama31_70b"], "fp16") > usable, "141GB must exceed 72GB usable"
    assert _weight_bytes(models["llama31_70b"], "fp8") < usable, "fp8 70B (70.6GB) just fits"


# ── build integrity + solution execution ─────────────────────────────────────


def test_learner_has_stubs_solution_does_not(solution_nb):
    lab = solution_nb.parent.parent.parent / "labs/04-gpu-architecture/lab.ipynb"
    lab_src = " ".join("".join(c["source"]) for c in json.loads(lab.read_text())["cells"])
    sol_src = " ".join("".join(c["source"]) for c in json.loads(solution_nb.read_text())["cells"])
    assert "TODO" in lab_src and "replace this line" in lab_src, "learner copy lost its stubs"
    assert "TODO" not in sol_src and "replace this line" not in sol_src, "solution has stubs"


@pytest.mark.slow
def test_solution_notebook_executes(solution_nb):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    nb = nbformat.read(str(solution_nb), as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(solution_nb.parent)}},
    )
    client.execute()
