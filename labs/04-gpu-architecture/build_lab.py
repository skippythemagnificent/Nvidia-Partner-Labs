"""Author the Lab 04 notebooks from a single source.

Emits two notebooks that stay in sync:
  - labs/04-gpu-architecture/lab.ipynb        — learner copy (exercise cells stubbed)
  - solutions/04-gpu-architecture/lab.ipynb   — completed, nbmake-clean copy

Same committed-builder pattern as Labs 01–03: edit THIS file (never the .ipynb),
each `code(solution, stub)` carries both forms, markdown is shared, then regenerate:

    uv run python labs/04-gpu-architecture/build_lab.py

Lab 04 is analytical: no GPU, no NIM calls. Every cell computes from the datasheet
tables in data/, so the solution notebook executes anywhere with no network.
"""
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/04-gpu-architecture/lab.ipynb"
SOL = ROOT / "solutions/04-gpu-architecture/lab.ipynb"

CELLS = []
def md(s): CELLS.append(("md", s))
def code(sol, stub=None): CELLS.append(("code", (sol, stub)))

# ── Scenario ─────────────────────────────────────────────────────────────────
md("""# Lab 04 — GPU Architecture for Inference

## Scenario

An ML engineer is sizing GPU infrastructure for a multi-tenant LLM service, and the
budget review is Thursday. The vendor deck says *"H100 is 6× faster than A100"* — a
number that's true for exactly one workload regime and misleading for several others.
Procurement wants a single SKU and a GPU count; the engineer needs to defend that
config with arithmetic, not a benchmark screenshot.

This lab builds the **back-of-the-envelope inference capacity model** that defends a
sizing decision: how much HBM the weights eat, how many tokens of KV cache fit in
what's left (your concurrency ceiling), where decode flips from **memory-bound** to
**compute-bound**, what TTFT and inter-token latency the SLA allows, and finally a
**capacity planner** that turns a target RPS + SLA into a GPU count. It's fully
analytical — no GPU required — and every number comes from the datasheet tables in
`data/`, produced by `data/generate.py`. You'll also see the two results that reshape
real deployments: the **roofline crossover batch** and the moment a 70B model simply
**won't fit on one GPU**.""")

# ── Setup ────────────────────────────────────────────────────────────────────
md("""## Setup

Loads the GPU, model, and workload tables and defines the small physical helpers the
exercises build on: peak compute (FLOP/s), memory bandwidth (bytes/s), bytes per
element for a dtype, and total weight bytes. `MEM_UTIL` is the fraction of HBM a NIM
leaves usable for weights + KV after framework overhead.""")

code(r'''import json
import math
import sys
from pathlib import Path

from pydantic import BaseModel
from rich import print as rprint
from rich.table import Table
from rich.console import Console

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "shared").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.utils import display_metrics_table

DATA = REPO_ROOT / "labs/04-gpu-architecture/data"
GPUS = {g["id"]: g for g in json.loads((DATA / "gpus.json").read_text())["gpus"]}
MODELS = {m["id"]: m for m in json.loads((DATA / "models.json").read_text())["models"]}
WORKLOADS = {w["id"]: w for w in json.loads((DATA / "workloads.json").read_text())["workloads"]}

MEM_UTIL = 0.90          # fraction of HBM usable for weights + KV cache
_console = Console()


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
    binding_constraint: str        # "KV cache" or "ITL SLA"
    per_gpu_tokens_per_s: float
    required_tokens_per_s: float
    num_gpus: int
    ttft_ms: float
    ttft_ok: bool


print(f"GPUs: {list(GPUS)}")
print(f"models: {list(MODELS)} | workloads: {list(WORKLOADS)}")''')

md("""**Expected output:**
```
GPUs: ['h100_sxm', 'h100_pcie', 'a100_80_sxm', 'l40s']
models: ['llama31_8b', 'llama31_70b'] | workloads: ['chat', 'rag', 'summarize']
```""")

# ── 1. KV cache per token ────────────────────────────────────────────────────
md("""## 1 · The KV cache tax — bytes per token

### Concept

Every token a request keeps in context costs HBM: the attention **key and value**
vectors for that token, at every layer, must stay resident so future tokens can
attend to them. That's the KV cache, and its per-token size is fixed by the model's
shape:

```
kv_bytes_per_token = 2 (K and V) × num_layers × kv_heads × head_dim × dtype_bytes
```

The `2` is K **and** V. Note it scales with `kv_heads`, **not** the full attention
head count — that's the whole point of **grouped-query attention (GQA)**: Llama-3.1
uses 8 KV heads to serve 32–64 attention heads, shrinking the cache 4–8×. This single
number drives your concurrency ceiling in §2 and reappeared as the 503-causing budget
in Lab 03.

> **Where do `num_layers`, `kv_heads`, and `head_dim` come from?** They aren't
> derived — they're fixed **architecture hyperparameters** the model's designers
> chose, published in the model card and the model's `config.json`
> (`num_hidden_layers`, `num_key_value_heads`, `head_dim`). `data/models.json` just
> records them. Depth (`num_layers`) and width (`hidden`) are tuned *together* to hit
> a target parameter count at a depth-to-width ratio scaling-law research favors — so
> the 8B's 32 layers and the 70B's 80 layers are empirical design points, not formula
> outputs. You look them up; you don't compute them. (And because KV scales linearly
> with `num_layers`, the 70B's depth is exactly why its KV cache is 2.5× the 8B's.)

### Your task — `kv_cache_bytes_per_token`

Implement the formula above.

**Walkthrough.** For Llama-3.1-8B at fp16: `2 × 32 layers × 8 kv_heads × 128 head_dim
× 2 bytes = 131,072` bytes = **128 KiB per token**. The 70B has 80 layers (same 8 KV
heads, same head_dim), so it's `2 × 80 × 8 × 128 × 2 = 327,680` bytes — 2.5× the 8B
because it's 2.5× deeper.""")

code(r'''def kv_cache_bytes_per_token(model: dict, dtype: str) -> int:
    """KV cache size for one token: 2 * layers * kv_heads * head_dim * dtype_bytes."""
    return 2 * model["num_layers"] * model["kv_heads"] * model["head_dim"] * dtype_bytes(dtype)


for mid in ("llama31_8b", "llama31_70b"):
    kb = kv_cache_bytes_per_token(MODELS[mid], "fp16") / 1024
    print(f"{MODELS[mid]['name']:<26} {kb:6.0f} KiB/token (fp16)")''',
r'''def kv_cache_bytes_per_token(model: dict, dtype: str) -> int:
    """KV cache size for one token: 2 * layers * kv_heads * head_dim * dtype_bytes."""
    # TODO: return 2 * num_layers * kv_heads * head_dim * dtype_bytes(dtype).
    # Use kv_heads (GQA), not attn_heads.
    raise NotImplementedError("Complete kv_cache_bytes_per_token before continuing")


for mid in ("llama31_8b", "llama31_70b"):
    kb = kv_cache_bytes_per_token(MODELS[mid], "fp16") / 1024
    print(f"{MODELS[mid]['name']:<26} {kb:6.0f} KiB/token (fp16)")''')

md("""**Expected output:**
```
Llama-3.1-8B-Instruct         128 KiB/token (fp16)
Llama-3.1-70B-Instruct        320 KiB/token (fp16)
```""")

# ── 2. HBM budget -> concurrency ─────────────────────────────────────────────
md("""## 2 · Where does the HBM go? Weights first, KV with the rest

### What is HBM?

**HBM (High Bandwidth Memory)** is the GPU's on-package video memory — the `memory_gb`
in the specs table (e.g. 80 GB on an H100). It's stacked DRAM sitting right next to
the GPU die on the same package over a very wide bus, which is what makes it *high
bandwidth* (HBM2e on the A100 / H100 PCIe, HBM3 at ~3.35 TB/s on the H100 SXM). Two
of its numbers drive this entire lab, and they correspond to the two ways it can run
out:

- **Capacity (`memory_gb`)** — *how much fits.* Weights + KV cache must both live
  here; this section's concurrency ceiling, and the "70B won't fit" failure below,
  are capacity problems.
- **Bandwidth (`mem_bandwidth_gb_s`)** — *how fast you can read it.* Decode rereads
  every weight from HBM per token, so token speed tracks bandwidth, not FLOPs — the
  roofline story in §3.

One-liner to carry through the lab: **capacity (weights + KV) is an HBM-*size*
problem; decode speed is an HBM-*bandwidth* problem.**

### Concept

HBM holds two big things during inference: the **model weights** (fixed) and the **KV
cache** (grows with concurrency × context). Weights are paid first; whatever usable
HBM remains is your KV budget, and that budget — divided by the per-request context
length — is your **maximum concurrent sequences**. This is the hardware that sets the
Lab 03 KV-cache ceiling.

```
usable_hbm   = memory_gb × 1e9 × MEM_UTIL
kv_budget    = usable_hbm − weight_bytes
max_tokens   = kv_budget // kv_bytes_per_token
max_sequences = max_tokens // context_len
```

### Your task — `max_concurrent_sequences`

Fill in the four-line budget above and return `(max_tokens, max_sequences)`.

**Step by step:**

1. `usable = gpu["memory_gb"] * 1e9 * MEM_UTIL`.
2. `kv_budget = usable - weight_bytes(model, dtype)`.
3. `max_tokens = int(kv_budget // kv_cache_bytes_per_token(model, dtype))`.
4. `max_sequences = max_tokens // context_len`; return both.""")

code(r'''def max_concurrent_sequences(gpu: dict, model: dict, dtype: str, context_len: int) -> tuple[int, int]:
    """(tokens of KV cache, concurrent sequences) that fit after weights."""
    usable = gpu["memory_gb"] * 1e9 * MEM_UTIL
    kv_budget = usable - weight_bytes(model, dtype)
    max_tokens = int(kv_budget // kv_cache_bytes_per_token(model, dtype))
    max_sequences = max_tokens // context_len
    return max_tokens, max_sequences


CONTEXT = 1280   # chat workload: 1024 in + 256 out
rows = []
for gid in ("h100_sxm", "a100_80_sxm", "l40s"):
    g = GPUS[gid]
    wt = weight_bytes(MODELS["llama31_8b"], "fp16") / 1e9
    toks, seqs = max_concurrent_sequences(g, MODELS["llama31_8b"], "fp16", CONTEXT)
    rows.append((g["name"], wt, toks, seqs))

tbl = Table(title="Llama-3.1-8B fp16 — HBM budget per GPU (context=1280)")
for c in ("GPU", "weights (GB)", "KV tokens", "max sequences"):
    tbl.add_column(c, justify="right" if c != "GPU" else "left")
for name, wt, toks, seqs in rows:
    tbl.add_row(name, f"{wt:.1f}", f"{toks:,}", str(seqs))
_console.print(tbl)''',
r'''def max_concurrent_sequences(gpu: dict, model: dict, dtype: str, context_len: int) -> tuple[int, int]:
    """(tokens of KV cache, concurrent sequences) that fit after weights."""
    # TODO: usable = memory_gb * 1e9 * MEM_UTIL; subtract weight_bytes(model, dtype)
    # to get the KV budget; floor-divide by kv_cache_bytes_per_token for max_tokens;
    # floor-divide that by context_len for max_sequences. Return (max_tokens, max_sequences).
    raise NotImplementedError("Complete max_concurrent_sequences before continuing")


CONTEXT = 1280   # chat workload: 1024 in + 256 out
rows = []
for gid in ("h100_sxm", "a100_80_sxm", "l40s"):
    g = GPUS[gid]
    wt = weight_bytes(MODELS["llama31_8b"], "fp16") / 1e9
    toks, seqs = max_concurrent_sequences(g, MODELS["llama31_8b"], "fp16", CONTEXT)
    rows.append((g["name"], wt, toks, seqs))

tbl = Table(title="Llama-3.1-8B fp16 — HBM budget per GPU (context=1280)")
for c in ("GPU", "weights (GB)", "KV tokens", "max sequences"):
    tbl.add_column(c, justify="right" if c != "GPU" else "left")
for name, wt, toks, seqs in rows:
    tbl.add_row(name, f"{wt:.1f}", f"{toks:,}", str(seqs))
_console.print(tbl)''')

md("""**Expected output** (weights are identical across GPUs; the 48 GB L40S has far
less left for KV, so it serves fewer sequences):
```
        Llama-3.1-8B fp16 — HBM budget per GPU (context=1280)
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ GPU                  ┃ weights (GB) ┃ KV tokens ┃ max sequences ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ H100 80GB SXM5       │         16.1 │   426,788 │           333 │
│ A100 80GB SXM4       │         16.1 │   426,788 │           333 │
│ L40S 48GB            │         16.1 │   207,061 │           161 │
└──────────────────────┴──────────────┴───────────┴───────────────┘
```
Both 80 GB cards hold ~333 concurrent 1280-token sequences; the L40S, with 32 GB less
HBM, holds ~half. Capacity here is a **memory** story, identical compute notwithstanding.""")

# ── 3. Roofline: memory-bound vs compute-bound decode ────────────────────────
md("""## 3 · Why "6× faster" is a half-truth — the decode roofline

### Concept

Token generation (**decode**) is **memory-bound**, not compute-bound. To generate one
token the GPU must read *every weight* from HBM, but does only ~`2 × params` FLOPs with
them — an arithmetic intensity of ~1 FLOP/byte at fp16. The GPU's tensor cores can do
hundreds of FLOPs per byte, so at small batch they sit idle waiting on memory. Decode
speed is set by **bandwidth**, which is why an A100 (2.0 TB/s) and an H100 PCIe
(2.0 TB/s) decode at nearly the same rate despite very different FLOPs.

**Batching** is the fix: reading the weights once and applying them to `B` sequences
amortizes the memory cost over `B` tokens, so throughput climbs ~linearly with batch —
**until** the work finally saturates the tensor cores and decode becomes
compute-bound. Model it as the smaller of the two ceilings:

```
memory_bound_tps  = batch × bandwidth / weight_bytes      # grows with batch
compute_bound_tps = peak_flops / (2 × params)             # flat ceiling
tokens_per_s      = min(memory_bound_tps, compute_bound_tps)
```

### Your task — `decode_tokens_per_s`

Return `min(memory_bound, compute_bound)` per the formulas above.

**Walkthrough.** At `batch=1` on the H100 SXM, 8B fp16: memory-bound =
`1 × 3.35e12 / 16.06e9 ≈ 209` tok/s (one user, bandwidth-starved). The compute ceiling
is `989.5e12 / (2 × 8.03e9) ≈ 61,613` tok/s. They meet at **batch ≈ 295** — and that
crossover batch equals the GPU's FLOP-per-byte ratio, the roofline "ridge". Below it
you're wasting compute; above it, more batch buys nothing.""")

code(r'''def decode_tokens_per_s(gpu: dict, model: dict, dtype: str, batch: int) -> float:
    """Decode throughput = min(memory-bound, compute-bound) at this batch size."""
    memory_bound = batch * mem_bandwidth(gpu) / weight_bytes(model, dtype)
    compute_bound = peak_flops(gpu, dtype) / (2 * model["params"])
    return min(memory_bound, compute_bound)


def crossover_batch(gpu: dict, dtype: str) -> float:
    """Batch where decode flips memory->compute bound = the GPU's FLOP:byte ridge."""
    return peak_flops(gpu, dtype) / mem_bandwidth(gpu)


g = GPUS["h100_sxm"]
m = MODELS["llama31_8b"]
print(f"crossover batch (H100 SXM, fp16): {crossover_batch(g, 'fp16'):.0f}  "
      f"(= ridge {peak_flops(g, 'fp16') / mem_bandwidth(g):.0f} FLOP/byte)\n")

tbl = Table(title="Llama-3.1-8B fp16 decode throughput vs batch (H100 SXM)")
tbl.add_column("batch", justify="right"); tbl.add_column("tok/s", justify="right")
tbl.add_column("regime"); tbl.add_column("")
peak = decode_tokens_per_s(g, m, "fp16", 100000)
for b in (1, 8, 32, 128, 256, 512):
    tps = decode_tokens_per_s(g, m, "fp16", b)
    regime = "memory-bound" if tps < peak else "compute-bound"
    bar = "█" * round(40 * tps / peak)
    tbl.add_row(str(b), f"{tps:,.0f}", regime, bar)
_console.print(tbl)''',
r'''def decode_tokens_per_s(gpu: dict, model: dict, dtype: str, batch: int) -> float:
    """Decode throughput = min(memory-bound, compute-bound) at this batch size."""
    # TODO: memory_bound = batch * mem_bandwidth(gpu) / weight_bytes(model, dtype)
    #       compute_bound = peak_flops(gpu, dtype) / (2 * params)
    # return the smaller of the two.
    raise NotImplementedError("Complete decode_tokens_per_s before continuing")


def crossover_batch(gpu: dict, dtype: str) -> float:
    """Batch where decode flips memory->compute bound = the GPU's FLOP:byte ridge."""
    return peak_flops(gpu, dtype) / mem_bandwidth(gpu)


g = GPUS["h100_sxm"]
m = MODELS["llama31_8b"]
print(f"crossover batch (H100 SXM, fp16): {crossover_batch(g, 'fp16'):.0f}  "
      f"(= ridge {peak_flops(g, 'fp16') / mem_bandwidth(g):.0f} FLOP/byte)\n")

tbl = Table(title="Llama-3.1-8B fp16 decode throughput vs batch (H100 SXM)")
tbl.add_column("batch", justify="right"); tbl.add_column("tok/s", justify="right")
tbl.add_column("regime"); tbl.add_column("")
peak = decode_tokens_per_s(g, m, "fp16", 100000)
for b in (1, 8, 32, 128, 256, 512):
    tps = decode_tokens_per_s(g, m, "fp16", b)
    regime = "memory-bound" if tps < peak else "compute-bound"
    bar = "█" * round(40 * tps / peak)
    tbl.add_row(str(b), f"{tps:,.0f}", regime, bar)
_console.print(tbl)''')

md("""**Expected output** (linear climb while memory-bound, then a hard ceiling):
```
crossover batch (H100 SXM, fp16): 295  (= ridge 295 FLOP/byte)

   Llama-3.1-8B fp16 decode throughput vs batch (H100 SXM)
┏━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ batch ┃  tok/s ┃ regime        ┃                                         ┃
┡━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│     1 │    209 │ memory-bound  │                                         │
│     8 │  1,670 │ memory-bound  │ █                                       │
│    32 │  6,679 │ memory-bound  │ ████                                    │
│   128 │ 26,716 │ memory-bound  │ █████████████████                       │
│   256 │ 53,432 │ memory-bound  │ ██████████████████████████████████      │
│   512 │ 61,613 │ compute-bound │ ████████████████████████████████████████│
└───────┴────────┴───────────────┴─────────────────────────────────────────┘
```
From batch 1 → 256 throughput scales ~linearly (256× the work, 256× the tokens) — pure
bandwidth amortization. By batch 512 it's pinned at the compute ceiling (~61.6k tok/s);
extra batch only adds latency now. **This** is why the "6×" headline is regime-specific:
single-user decode tracks bandwidth (~1.6×), saturated throughput tracks FLOPs (~3×+).""")

# ── 4. Prefill TTFT ──────────────────────────────────────────────────────────
md("""## 4 · The other half — prefill and TTFT

### Concept

Before the first token streams, the model must process the *entire prompt* in one
shot: **prefill**. Unlike decode, prefill does `2 × params` FLOPs for **every input
token** at once, so it has high arithmetic intensity and is **compute-bound** — the
regime where FLOPs (and FP8) actually matter. Prefill time is your **time-to-first-
token (TTFT)** floor:

```
ttft_seconds ≈ 2 × params × input_tokens / peak_flops
```

So decode is a memory problem and prefill is a compute problem — the same GPU lives on
*both* sides of the roofline within a single request. Long-context workloads (RAG,
summarization) are prefill-heavy and benefit most from FP8/more FLOPs; chatty
short-prompt workloads are decode-heavy and benefit most from bandwidth.

### Your task — `prefill_ttft_ms`

Return the TTFT floor in **milliseconds** from the formula above.""")

code(r'''def prefill_ttft_ms(gpu: dict, model: dict, dtype: str, input_tokens: int) -> float:
    """Compute-bound prefill time (TTFT floor) in milliseconds."""
    return 2 * model["params"] * input_tokens / peak_flops(gpu, dtype) * 1000


m = MODELS["llama31_8b"]
metrics = {}
for gid in ("h100_sxm", "a100_80_sxm"):
    metrics[f"TTFT {GPUS[gid]['name']} (1024 in)"] = prefill_ttft_ms(GPUS[gid], m, "fp16", 1024)
metrics["TTFT H100 SXM fp8 (1024 in)"] = prefill_ttft_ms(GPUS["h100_sxm"], m, "fp8", 1024)
display_metrics_table(metrics, title="Llama-3.1-8B prefill TTFT floor")''',
r'''def prefill_ttft_ms(gpu: dict, model: dict, dtype: str, input_tokens: int) -> float:
    """Compute-bound prefill time (TTFT floor) in milliseconds."""
    # TODO: 2 * params * input_tokens / peak_flops(gpu, dtype), converted to ms (*1000).
    raise NotImplementedError("Complete prefill_ttft_ms before continuing")


m = MODELS["llama31_8b"]
metrics = {}
for gid in ("h100_sxm", "a100_80_sxm"):
    metrics[f"TTFT {GPUS[gid]['name']} (1024 in)"] = prefill_ttft_ms(GPUS[gid], m, "fp16", 1024)
metrics["TTFT H100 SXM fp8 (1024 in)"] = prefill_ttft_ms(GPUS["h100_sxm"], m, "fp8", 1024)
display_metrics_table(metrics, title="Llama-3.1-8B prefill TTFT floor")''')

md("""**Expected output** (prefill is compute-bound, so the H100's FLOPs — and FP8 —
pull TTFT down hard; this is the opposite of the decode story):
```
              Llama-3.1-8B prefill TTFT floor
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric                                     ┃   Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ TTFT H100 80GB SXM5 (1024 in) │ 16.6199 │
│ TTFT A100 80GB SXM4 (1024 in) │ 52.7097 │
│ TTFT H100 SXM fp8 (1024 in)   │  8.3104 │
└───────────────────────────────┴─────────┘
```
Here the H100 really is ~3.2× the A100 (the FLOPs ratio), and FP8 halves it again.
Prefill is where compute headroom shows up; decode is where bandwidth does.""")

# ── 5. Capacity planner (capstone) ───────────────────────────────────────────
md("""## 5 · Capstone — the capacity planner

### Concept

Now combine everything into the number procurement wants: **how many GPUs** for a
workload. The chain:

1. **Demand:** `required_tokens_per_s = target_rps × output_tokens`.
2. **Batch ceiling** is the *smaller* of two limits:
   - **KV cache** — `max_concurrent_sequences` from §2 (memory).
   - **ITL SLA** — the inter-token-latency budget. In the compute-bound region each
     decode step costs `batch × 2 × params / peak_flops`, so the largest batch that
     keeps a step within the SLA is `itl_sla_s × peak_flops / (2 × params)`.
3. **Supply:** `per_gpu_tps = decode_tokens_per_s(batch)` at that batch.
4. **Count:** `num_gpus = ceil(required_tps / per_gpu_tps)`, and check TTFT (§4)
   against the SLA.

### Your task — `capacity_planner`

The demand, both batch limits, TTFT, and the weight-fit check are computed for you.
Implement the three lines that turn them into a recommendation.

**Step by step:**

1. `batch = max(1, min(batch_by_kv, batch_by_itl))`.
2. `per_gpu_tps = decode_tokens_per_s(gpu, model, dtype, batch)`.
3. `num_gpus = math.ceil(required_tps / per_gpu_tps)`.

(`binding_constraint` is `"KV cache"` when KV is the smaller limit, else `"ITL SLA"`.)""")

code(r'''def capacity_planner(workload: dict, gpu: dict, model: dict, dtype: str) -> PlannerResult:
    """Size a single-GPU deployment for a workload + SLA (tp=1)."""
    context = workload["input_tokens"] + workload["output_tokens"]
    required_tps = workload["target_rps"] * workload["output_tokens"]

    usable = gpu["memory_gb"] * 1e9 * MEM_UTIL
    fits = weight_bytes(model, dtype) < usable

    _, batch_by_kv = max_concurrent_sequences(gpu, model, dtype, context)
    batch_by_itl = int(workload["itl_ms_sla"] / 1000 * peak_flops(gpu, dtype) / (2 * model["params"]))
    ttft = prefill_ttft_ms(gpu, model, dtype, workload["input_tokens"])

    batch = max(1, min(batch_by_kv, batch_by_itl))
    per_gpu_tps = decode_tokens_per_s(gpu, model, dtype, batch)
    num_gpus = math.ceil(required_tps / per_gpu_tps)

    return PlannerResult(
        gpu=gpu["name"], model=model["name"], dtype=dtype,
        fits_on_one_gpu=fits,
        batch=batch,
        binding_constraint="KV cache" if batch_by_kv < batch_by_itl else "ITL SLA",
        per_gpu_tokens_per_s=round(per_gpu_tps),
        required_tokens_per_s=required_tps,
        num_gpus=num_gpus,
        ttft_ms=round(ttft, 1),
        ttft_ok=ttft <= workload["ttft_ms_sla"],
    )


chat = WORKLOADS["chat"]
print(f"workload: {chat['name']} | {chat['target_rps']} rps | "
      f"SLA ttft<{chat['ttft_ms_sla']}ms itl<{chat['itl_ms_sla']}ms\n")
for gid in ("h100_sxm", "a100_80_sxm", "l40s"):
    r = capacity_planner(chat, GPUS[gid], MODELS["llama31_8b"], "fp16")
    print(f"{r.gpu:<16} {r.dtype} | batch {r.batch:>3} ({r.binding_constraint:<9}) | "
          f"{r.per_gpu_tokens_per_s:>8,.0f} tok/s/gpu | need {r.num_gpus} gpu(s) "
          f"for {r.required_tokens_per_s:,.0f} tok/s | TTFT {r.ttft_ms}ms "
          f"{'OK' if r.ttft_ok else 'SLA!'}")''',
r'''def capacity_planner(workload: dict, gpu: dict, model: dict, dtype: str) -> PlannerResult:
    """Size a single-GPU deployment for a workload + SLA (tp=1)."""
    context = workload["input_tokens"] + workload["output_tokens"]
    required_tps = workload["target_rps"] * workload["output_tokens"]

    usable = gpu["memory_gb"] * 1e9 * MEM_UTIL
    fits = weight_bytes(model, dtype) < usable

    _, batch_by_kv = max_concurrent_sequences(gpu, model, dtype, context)
    batch_by_itl = int(workload["itl_ms_sla"] / 1000 * peak_flops(gpu, dtype) / (2 * model["params"]))
    ttft = prefill_ttft_ms(gpu, model, dtype, workload["input_tokens"])

    # TODO: batch = max(1, min(batch_by_kv, batch_by_itl));
    #       per_gpu_tps = decode_tokens_per_s(gpu, model, dtype, batch);
    #       num_gpus = math.ceil(required_tps / per_gpu_tps).
    batch = None  # replace this line
    per_gpu_tps = None  # replace this line
    num_gpus = None  # replace this line
    assert None not in (batch, per_gpu_tps, num_gpus), "Complete capacity_planner before continuing"

    return PlannerResult(
        gpu=gpu["name"], model=model["name"], dtype=dtype,
        fits_on_one_gpu=fits,
        batch=batch,
        binding_constraint="KV cache" if batch_by_kv < batch_by_itl else "ITL SLA",
        per_gpu_tokens_per_s=round(per_gpu_tps),
        required_tokens_per_s=required_tps,
        num_gpus=num_gpus,
        ttft_ms=round(ttft, 1),
        ttft_ok=ttft <= workload["ttft_ms_sla"],
    )


chat = WORKLOADS["chat"]
print(f"workload: {chat['name']} | {chat['target_rps']} rps | "
      f"SLA ttft<{chat['ttft_ms_sla']}ms itl<{chat['itl_ms_sla']}ms\n")
for gid in ("h100_sxm", "a100_80_sxm", "l40s"):
    r = capacity_planner(chat, GPUS[gid], MODELS["llama31_8b"], "fp16")
    print(f"{r.gpu:<16} {r.dtype} | batch {r.batch:>3} ({r.binding_constraint:<9}) | "
          f"{r.per_gpu_tokens_per_s:>8,.0f} tok/s/gpu | need {r.num_gpus} gpu(s) "
          f"for {r.required_tokens_per_s:,.0f} tok/s | TTFT {r.ttft_ms}ms "
          f"{'OK' if r.ttft_ok else 'SLA!'}")''')

md("""**Expected output** (8B fits on one GPU everywhere; the chat SLA is loose, so
**KV cache** — not latency — binds the batch, and one H100/A100 covers 50 rps):
```
workload: Interactive multi-tenant chat | 50 rps | SLA ttft<500ms itl<40ms

H100 80GB SXM5   fp16 | batch 333 (KV cache ) |   61,613 tok/s/gpu | need 1 gpu(s) for 12,800 tok/s | TTFT 16.6ms OK
A100 80GB SXM4   fp16 | batch 333 (KV cache ) |   19,427 tok/s/gpu | need 1 gpu(s) for 12,800 tok/s | TTFT 52.7ms OK
L40S 48GB        fp16 | batch 161 (KV cache ) |    8,662 tok/s/gpu | need 2 gpu(s) for 12,800 tok/s | TTFT 45.4ms OK
```
Required load is 50 rps × 256 out = **12,800 tok/s**. One H100 SXM (61.6k tok/s
saturated) or one A100 (19.4k) covers it; the L40S needs **2**. The binding constraint
is KV cache, so the lever that helps most here is **FP8** (halves weight + KV bytes →
bigger batch) — try it in the challenge.""")

# ── Failure 1: 70B doesn't fit ───────────────────────────────────────────────
md("""### ⚠️ Failure 1 — the 70B that won't fit (and why NVLink exists)

Swap the 8B for the **70B** and the model breaks the single-GPU assumption before
throughput even enters the picture. Watch the weight-fit check.""")

code(r'''m70 = MODELS["llama31_70b"]
g = GPUS["h100_sxm"]
usable = g["memory_gb"] * 1e9 * MEM_UTIL

for dtype in ("fp16", "fp8"):
    wt = weight_bytes(m70, dtype) / 1e9
    fits = weight_bytes(m70, dtype) < usable
    leftover = (usable - weight_bytes(m70, dtype)) / 1e9
    print(f"70B {dtype}: weights {wt:5.1f} GB vs usable {usable/1e9:.1f} GB "
          f"-> fits={fits}  (KV headroom {leftover:+.1f} GB)")


def min_tp_to_fit(model, gpu, dtype, n_options=(1, 2, 4, 8)) -> int | None:
    """Smallest tensor-parallel size whose per-GPU weight shard fits in usable HBM."""
    usable = gpu["memory_gb"] * 1e9 * MEM_UTIL
    for tp in n_options:
        if weight_bytes(model, dtype) / tp < usable:
            return tp
    return None


print()
for dtype in ("fp16", "fp8"):
    tp = min_tp_to_fit(m70, g, dtype)
    print(f"70B {dtype}: minimum tensor-parallel size to fit weights = tp={tp} "
          f"({weight_bytes(m70, dtype)/tp/1e9:.1f} GB/GPU)")''')

md("""**Expected output:**
```
70B fp16: weights 141.2 GB vs usable 72.0 GB -> fits=False  (KV headroom -69.2 GB)
70B fp8:  weights  70.6 GB vs usable 72.0 GB -> fits=True   (KV headroom +1.4 GB)

70B fp16: minimum tensor-parallel size to fit weights = tp=2 (70.6 GB/GPU)
70B fp8:  minimum tensor-parallel size to fit weights = tp=1 (70.6 GB/GPU)
```
**Root cause.** 70B fp16 weights are **141 GB** — they don't fit in one 80 GB GPU at
all, so `fits_on_one_gpu=False` is a hard stop, not a throughput tweak. You must
**shard the weights across GPUs with tensor parallelism (TP)**, and TP reads partial
activations from every peer on *every* layer — which is why those GPUs need
**NVLink** (900 GB/s on SXM) rather than PCIe. Note fp8 *just barely* fits on one GPU
(1.4 GB KV headroom → near-zero concurrency), so the *practical* floor is still tp≥2
for fp16 / tp≥2 for real fp8 batch. Sizing a 70B without checking weight-fit first is
the classic capacity-planning blunder.""")

# ── Failure 2: ignoring the roofline ─────────────────────────────────────────
md("""### ⚠️ Failure 2 — buying FLOPs for a memory-bound workload

A tempting "fix" for slow single-user decode is to buy a higher-FLOP GPU. Test that
belief: compare single-user (batch=1) decode on the A100 vs the H100 SXM, then the
same at saturating batch.""")

code(r'''m = MODELS["llama31_8b"]
a100, h100 = GPUS["a100_80_sxm"], GPUS["h100_sxm"]

for batch in (1, 512):
    a = decode_tokens_per_s(a100, m, "fp16", batch)
    h = decode_tokens_per_s(h100, m, "fp16", batch)
    print(f"batch {batch:>3}: A100 {a:8,.0f} tok/s | H100 {h:8,.0f} tok/s | "
          f"H100 is {h/a:.2f}x")

print(f"\nFLOPs ratio  H100/A100 = {h100['fp16_tflops']/a100['fp16_tflops']:.2f}x")
print(f"Bandwidth ratio H100/A100 = {h100['mem_bandwidth_gb_s']/a100['mem_bandwidth_gb_s']:.2f}x")''')

md("""**Expected output:**
```
batch   1: A100      127 tok/s | H100      209 tok/s | H100 is 1.64x
batch 512: A100   19,427 tok/s | H100   61,613 tok/s | H100 is 3.17x

FLOPs ratio  H100/A100 = 3.17x
Bandwidth ratio H100/A100 = 1.64x
```
**Root cause.** At batch 1 (memory-bound) the H100's advantage is just the
**bandwidth** ratio (1.64×) — those extra FLOPs sit idle. The 3.17× only appears once
batching makes decode **compute-bound**. So if your problem is single-user latency,
the high-FLOP GPU barely helps; the real lever is **bigger batch** (continuous
batching) or **more bandwidth**. Match the GPU to the regime your workload actually
runs in — that's the whole point of the roofline.""")

# ── Challenge ────────────────────────────────────────────────────────────────
md("""## Challenge

1. **FP8 on the binding constraint.** Re-run `capacity_planner` for the chat workload
   on the H100 SXM with `dtype="fp8"`. Weight + KV bytes halve and FLOPs double — how
   much does the batch ceiling and `num_gpus` change? Which constraint binds now?
2. **Pick a GPU for RAG.** The `rag` workload is long-context (3000 in / 400 out,
   TTFT < 800 ms). Run the planner across all four GPUs for the 8B. Does the
   prefill-heavy profile change which GPU is most cost-effective once you divide by
   `usd_per_hr`? Add a `tokens_per_s_per_dollar` column.
3. **Size the 70B properly.** Extend `capacity_planner` to take a `tp` argument:
   scale memory, bandwidth, and FLOPs by `tp` (an NVLink node behaves like one big
   GPU), report **total** GPUs as `num_nodes × tp`, and size the `rag` workload on
   70B fp16. What tp gives usable KV headroom, and how many GPUs total?""")

# ── Key takeaways ────────────────────────────────────────────────────────────
md("""## Key takeaways

- **Weights first, KV with the rest.** Usable HBM minus weight bytes, divided by KV
  bytes/token, divided by context length, is your concurrency ceiling. GQA (few KV
  heads) is what makes long context affordable.
- **Decode is memory-bound; prefill is compute-bound.** Single-user token speed tracks
  **bandwidth**; TTFT and saturated throughput track **FLOPs** (and FP8). The same GPU
  lives on both sides of the roofline within one request.
- **The crossover batch is the FLOP:byte ridge.** Below it, batching is free
  throughput; above it, only latency. "6× faster" is true only in the compute-bound
  regime.
- **Check weight-fit before anything else.** A 70B fp16 (141 GB) doesn't fit on one
  80 GB GPU — that mandates tensor parallelism and NVLink, not a faster single card.
- **Size from the binding constraint.** KV cache vs ITL SLA decides the batch; match
  the *lever* (FP8, bandwidth, batch, TP) to whichever one binds.

**References**
- H100 architecture & HBM3: https://resources.nvidia.com/en-us-tensor-core
- TRT-LLM continuous batching: https://docs.nvidia.com/tensorrt-llm/
- KV cache & GQA: https://docs.nvidia.com/nim/large-language-models/latest/
- FP8 for inference (Transformer Engine): https://docs.nvidia.com/deeplearning/transformer-engine/
- MIG & multi-tenant GPUs: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/""")

# ── build ─────────────────────────────────────────────────────────────────────
def make(use_stub: bool):
    nb = new_notebook()
    cells = []
    for kind, payload in CELLS:
        if kind == "md":
            cells.append(new_markdown_cell(payload))
        else:
            sol, stub = payload
            cells.append(new_code_cell(stub if (use_stub and stub) else sol))
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
