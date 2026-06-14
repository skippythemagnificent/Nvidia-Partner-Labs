# Lab 04 — GPU Architecture for Inference

## What

A fully analytical capacity-planning notebook — no GPU, no NIM calls. From three
datasheet tables (`data/gpus.json`, `models.json`, `workloads.json`) you build the
back-of-the-envelope model that sizes an inference deployment:

- `kv_cache_bytes_per_token` — the KV-cache tax, and why grouped-query attention
  (few KV heads) makes long context affordable.
- `max_concurrent_sequences` — weights eat HBM first; what's left ÷ KV-per-token ÷
  context length is your concurrency ceiling.
- `decode_tokens_per_s` + `crossover_batch` — the decode **roofline**: memory-bound
  at small batch, compute-bound past the FLOP:byte ridge.
- `prefill_ttft_ms` — prefill is compute-bound; TTFT tracks FLOPs (and FP8).
- `capacity_planner` — the capstone: target RPS + SLA → recommended GPU count, with
  the binding constraint (KV cache vs ITL) called out.

Two failure sections drive the real lessons home: a **70B that won't fit on one
80 GB GPU** (→ tensor parallelism + NVLink), and **buying FLOPs for a memory-bound
workload** (the "6× faster" half-truth).

## Why

An ML engineer is sizing infra for a multi-tenant LLM service and the budget review
is Thursday. "H100 is 6× faster than A100" is true in one regime and wrong in
several. This lab is the framework to defend a SKU and a GPU count with arithmetic:
prefill-bound vs decode-bound, KV-cache headroom, the batch ceiling before TTFT or
ITL breaks SLA, and when FP8, tensor parallelism, or more bandwidth is the right
lever.

## How

```bash
task lab:data:generate LAB=04-gpu-architecture   # writes gpus/models/workloads.json
task lab:run LAB=04-gpu-architecture
task lab:test LAB=04-gpu-architecture
```

No NIM calls — this lab is analytical. The notebook prints throughput curves and a
config-recommendation cell.

**You'll measure:** KV bytes/token (128 KiB for 8B fp16), concurrency ceiling
(~333 seqs on an 80 GB GPU), the roofline crossover batch (= the GPU's FLOP:byte
ridge, 295 on H100 SXM), prefill TTFT, FP8 vs FP16 deltas, and the GPU count a
workload+SLA actually needs.
