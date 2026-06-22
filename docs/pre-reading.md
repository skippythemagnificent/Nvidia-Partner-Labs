# Pre-Reading: The Theory Behind the NVIDIA Partner Labs

*A deep dive into the concepts and mathematics underpinning Labs 01–06. Read this before
the hands-on notebooks — it explains **why** the code is shaped the way it is, so the labs
become "confirm the theory" rather than "discover it cold."*

Each chapter maps to one lab. The notebooks are deliberately problem-first (you see a
failure, then fix it); this document is concept-first (the model that predicts the
failure). The two are meant to be read together: every number derived here shows up as a
printed result in a lab cell.

> **Math notation.** Equations use LaTeX (`$…$` inline, `$$…$$` display). They render on
> GitHub, in VS Code, Obsidian, and most markdown viewers. If yours doesn't, the prose
> carries the argument on its own.

---

## Table of Contents

1. [RAG Fundamentals — vectors, cosine, and chunking](#chapter-1--rag-fundamentals)
2. [Reranking — bi- vs cross-encoders and the two-stage cascade](#chapter-2--reranking)
3. [NIM Deployment — profile selection and the KV-cache ceiling](#chapter-3--nim-deployment--troubleshooting)
4. [GPU Architecture — the roofline, KV math, and capacity sizing](#chapter-4--gpu-architecture-for-inference)
5. [Agents — determinism, temperature, and concurrency](#chapter-5--agents--orchestration)
6. [MLOps — rates, percentiles, and leading indicators](#chapter-6--mlops--platform)

A recurring thread ties them together: **Little's Law** and the idea that a system's
behavior is set by its *binding constraint*. Retrieval recall bounds answer quality
(Ch. 1–2); the KV-cache token budget bounds concurrency (Ch. 3–4); the slowest stage
bounds latency (Ch. 6). Learn to find the binding constraint and you can debug any of it.

---

## Chapter 1 — RAG Fundamentals

*Lab 01. The retrieval half of Retrieval-Augmented Generation: how text becomes vectors,
how similarity is measured, and why where you cut the text decides whether the answer is
findable at all.*

### 1.1 RAG as a chain, and the ceiling theorem

A RAG system is a pipeline:

$$\text{chunk} \to \text{embed} \to \text{index} \to \text{retrieve} \to \text{assemble} \to \text{generate}$$

Let $A$ be the event "the generated answer is correct," and $C$ be the event "the text that
answers the question is present in the context handed to the LLM." Generation cannot
invent grounded facts it never received, so:

$$P(A) \le P(C).$$

$P(C)$ is a pure *retrieval* quantity — it has nothing to do with the LLM. This is the
**ceiling theorem** of RAG, and it is why the labs measure retrieval *before* touching the
prompt: if $P(C)$ is 50%, no prompt engineering can push answer accuracy above 50%. Every
stage to the left of "generate" can silently lower $P(C)$.

### 1.2 Embeddings: text as vectors

An **embedding model** is a function $E : \text{text} \to \mathbb{R}^d$ that maps a string
to a $d$-dimensional vector ($d = 1024$ for NV-EmbedQA-E5-v5; $d = 384$ for the lab's local
MiniLM mock). It is trained so that *semantically* similar texts land near each other,
under some geometric notion of "near."

The crucial subtlety in Lab 01 is **asymmetry**. NV-EmbedQA is a *dual encoder*: it has two
related but distinct encoders, $E_q$ for queries and $E_p$ for passages. The relevance of
passage $p$ to query $q$ is the inner product of their respective embeddings:

$$\text{score}(q, p) = \langle E_q(q),\, E_p(p) \rangle.$$

If you embed a query with the passage encoder (by omitting `input_type="query"`), you
compute $\langle E_p(q), E_p(p)\rangle$ — an inner product in the *wrong* space. The vectors
still have the right shape and the call still "succeeds," but the geometry is meaningless.
This is a **silent failure**: no exception, just quietly destroyed recall. Hence the
ironclad rule — *index passages, search with queries, declare the role every time.*

### 1.3 Cosine similarity, and why the store must be configured for it

Cosine similarity measures the angle between two vectors, ignoring their magnitudes:

$$\cos(\theta) = \frac{\langle a, b\rangle}{\lVert a\rVert\,\lVert b\rVert}
= \left\langle \frac{a}{\lVert a\rVert},\, \frac{b}{\lVert b\rVert}\right\rangle.$$

Embedding models are trained to put meaning in *direction*, not length, so cosine is the
right metric. Now the part that bites people: most vector stores default to **Euclidean
(L2)** distance. Are L2 and cosine interchangeable? Only for normalized vectors. Expand the
squared L2 distance of two unit vectors ($\lVert a\rVert = \lVert b\rVert = 1$):

$$\lVert a - b\rVert^2 = \lVert a\rVert^2 + \lVert b\rVert^2 - 2\langle a,b\rangle
= 2 - 2\cos(\theta).$$

So for **unit** vectors, *minimizing L2 distance is exactly maximizing cosine similarity* —
the two produce identical rankings. But if the vectors are **not** normalized, the
$\lVert a\rVert^2 + \lVert b\rVert^2$ terms vary per vector and the rankings diverge. Some
embedding APIs don't return unit vectors, so the lab configures ChromaDB for cosine space
explicitly (`metadata={"hnsw:space": "cosine"}`) and reports `score = 1 − cosine_distance`.
That score lands roughly in $[0.4, 0.8]$ for relevant hits.

Retrieval then returns the top-$K$ passages by score; their text is concatenated into the
prompt as grounded context. Larger $K$ raises recall but spends more tokens and injects more
distractors — a tension that motivates reranking (Chapter 2).

### 1.4 Chunking: the math of where you cut

A **chunk** is the atomic unit retrieval can return. Documents are sliced into chunks before
indexing, so chunking decides what is *retrievable at all*.

**Fixed-size chunking** slides a window of `size` characters across the text, stepping by
`size − overlap`. With $L$ = document length, $s$ = size, $o$ = overlap, the chunk count is

$$n_\text{chunks} = \left\lceil \frac{L - o}{s - o} \right\rceil.$$

Its fatal flaw is that it cuts on character count, not meaning. Consider an answer span of
length $w$ characters whose start position is uniformly distributed within a chunk of size
$s$ (with $o = 0$). The span survives intact inside one chunk only if it doesn't straddle a
boundary, which happens with probability

$$P(\text{span intact}) = 1 - \frac{w}{s} \quad (\text{for } w \le s).$$

So the probability of a **boundary split** is $w/s$: shorter chunks split answers more
often. But you can't just make $s$ huge — a giant chunk dilutes the embedding (the vector
averages over many topics) and wastes context tokens. This is the core chunking trade-off,
and it's why Lab 01's 140-character windows shred the answer to question q08: the sentence
"the latency is dominated by model load" gets cut, its lexically-strong fragment is
retrieved instead, and the answer fragment falls below the top-$K$ cutoff.

**Sentence-aware chunking** removes the split risk by construction: it never cuts
mid-sentence. It greedily packs whole sentences into a chunk until the next sentence would
exceed a budget `max_chars`, then starts a new chunk. Because sentences are the natural unit
of a complete thought, the answer sentence stays whole and retrievable. Holding the
embedding model and $K$ fixed and changing *only* the chunker is a clean controlled
experiment — and it moves Lab 01's hit-rate from 50% to 75% with no model change.

### 1.5 Measuring retrieval: hit-rate@K

You cannot improve what you do not measure. **Retrieval@K hit-rate** estimates $P(C)$
directly. With a labeled eval set where question $i$ carries a verbatim `answer_span`,

$$\text{hit-rate@}K = \frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}\!\left[\text{answer\_span}_i \subseteq \text{concat}\big(\text{top-}K \text{ chunks}\big)\right].$$

It is deterministic, needs no LLM judge, and is an unbiased Monte-Carlo estimator of $P(C)$
over the question distribution (standard error $\sqrt{p(1-p)/N}$). Because it's exact
substring containment, it's cheap, reproducible, and defensible to stakeholders — the
opposite of "the answers feel better now." The baseline of 50% is the villain of Lab 01;
everything after is about raising that number with a metric to prove it.

### 1.6 Why even good chunks aren't enough

Sentence chunking fixes *recall* but not *ranking*. Cosine similarity rewards surface
overlap: a query asking about a **warm** NIM can rank a passage about **cold** starts first,
because they share the tokens "first / slow / NIM." The bi-encoder scored query and passage
**independently**, so it never noticed that "warm" contradicts the cold-start passage. That
blindness is structural, and fixing it requires reading the query and passage *together* — a
cross-encoder reranker. That is Chapter 2.

---

## Chapter 2 — Reranking

*Lab 02. Two model architectures, the speed/accuracy trade-off between them, and the
two-stage cascade that gets you both.*

### 2.1 Bi-encoders vs cross-encoders, formally

A **bi-encoder** factorizes the relevance score into two independent encodings:

$$\text{score}_\text{bi}(q, p) = \langle \phi(q),\, \psi(p) \rangle.$$

Because $\psi(p)$ depends only on $p$, every passage vector can be **precomputed once** and
stored in an index. At query time you embed $q$ once and run an approximate-nearest-neighbor
(ANN) search. The factorization is what makes it scale — and also what limits it: $q$ and
$p$ never interact until the final dot product, so the model cannot reason about
*token-level* relationships between them (e.g. "warm" negating "cold-start").

A **cross-encoder** makes no such factorization:

$$\text{score}_\text{cross}(q, p) = g\big([q ; p]\big),$$

where $g$ is a transformer that attends over the *concatenation* of query and passage. Every
query token can attend to every passage token at every layer, so it captures interactions a
bi-encoder cannot. The price: $g(q,p)$ depends on both inputs jointly, so **nothing is
precomputable** — you pay a full forward pass for every $(q, p)$ pair you want to score.

| | Bi-encoder (embed) | Cross-encoder (rerank) |
|---|---|---|
| Score | $\langle\phi(q),\psi(p)\rangle$ — factorized | $g([q;p])$ — joint |
| Precompute passages? | Yes (index once) | No |
| Per-query cost | 1 embed + ANN search | one forward pass **per candidate** |
| Scales to | millions of chunks | a small shortlist |

### 2.2 The two-stage cascade

The resolution is a cascade: use the cheap bi-encoder to narrow the corpus to a shortlist,
then the expensive cross-encoder to re-score only the shortlist.

$$\text{corpus } (N) \xrightarrow{\text{bi-encoder}} k_1 \text{ candidates}
\xrightarrow{\text{cross-encoder}} k_2 \text{ final}.$$

Let $R_1(k_1) = P(\text{answer} \in \text{stage-1 top-}k_1)$ be stage-1 recall, and
$R_2 = P(\text{reranker ranks the answer in top-}k_2 \mid \text{answer is in the shortlist})$.
Because the reranker can only reorder what it was given, the final recall **factorizes and is
bounded by stage 1**:

$$P(\text{answer} \in \text{final top-}k_2) = R_1(k_1)\cdot R_2 \;\le\; R_1(k_1).$$

Two consequences fall straight out of this one equation:

- **Stage 2 can never fix stage 1's misses** (Failure 1). If $k_1$ is too small, the answer
  is excluded from the shortlist, $R_1$ drops, and no reranker can recover it. In Lab 02,
  $k_1 = 3$ loses an answer sitting at bi-rank 4; $k_1 = 20$ recovers it. The rule of thumb
  is to over-retrieve $4\text{–}10\times$ your final $k_2$, chosen at the knee of the
  $R_1(k_1)$ curve where extra candidates stop adding recall.
- **You can't just rerank everything** (Failure 2). Setting $k_1 = N$ maximizes $R_1$ but
  costs $N$ forward passes per query. The cascade keeps stage-2 cost at $O(k_1)$,
  *independent of corpus size*. (See §2.4.)

### 2.3 Measuring reranking: rank delta and MRR

Track each chunk's stage-1 position `embed_rank` and stage-2 position `rerank_rank`; their
difference is the signed movement:

$$\text{rank\_delta} = \text{embed\_rank} - \text{rerank\_rank} \quad (\text{positive} = \text{promoted}).$$

To score the whole pipeline, use **Mean Reciprocal Rank**. For each query, find the rank of
the *first* chunk containing the answer; its reciprocal rank is $1/\text{rank}$ (and $0$ if
not found). MRR averages this over the $Q$ queries:

$$\text{MRR} = \frac{1}{Q}\sum_{q=1}^{Q} \frac{1}{\text{rank}_q}.$$

The reciprocal is the whole point: it is steeply convex, so it *rewards top positions
disproportionately*. Moving the answer from rank 3 to rank 1 gains $1 - \tfrac{1}{3} = 0.67$,
but rank 5 to rank 3 gains only $\tfrac{1}{3} - \tfrac{1}{5} = 0.13$. MRR therefore measures
exactly what reranking is for — lifting the answer toward the top. In Lab 02, MRR rises
$0.88 \to 0.96$. The gain looks modest only because the mock cross-encoder is a tiny ~22M-
parameter model; the production NV-RerankQA-Mistral-4B-v3 (4B params) widens it. The pipeline
and the code are identical either way.

> **Logits, not probabilities.** The reranker emits an *unbounded* logit (≈ 8 for a strong
> match, negative for a poor one). Ranking only needs the *order*, and order is invariant
> under any monotonic transform, so calibration into a probability is unnecessary — you sort
> by logit and stop.

### 2.4 The cost asymmetry, quantified

Per query, the bi-encoder costs one embedding forward pass plus an ANN lookup (sub-linear,
$\approx O(\log N)$ with an HNSW index). The cross-encoder costs one forward pass per
candidate, i.e. $O(k_1)$. Reranking the *entire* corpus would be $O(N)$ — and crucially,
**nothing is cacheable**, because each score depends on the specific query. Lab 02 reranks 44
chunks in some time $t$; extrapolating to a realistic corpus of 1M chunks:

$$\frac{N}{k_\text{lab}} = \frac{1{,}000{,}000}{44} \approx 22{,}727\times.$$

That is 22,727× the per-query cost, every query, forever. The two-stage cascade exists
precisely to avoid it: over-retrieve cheaply, rerank a bounded shortlist.

---

## Chapter 3 — NIM Deployment & Troubleshooting

*Lab 03. Reading a NIM's startup, reproducing its profile selection as an optimization
problem, and the capacity arithmetic behind HTTP 503s.*

### 3.1 The startup sequence and time-to-ready

A NIM narrates four phases on the way up: GPU detection → TRT-LLM profile auto-selection →
engine load + KV-cache allocation → `Uvicorn running` (endpoint live). **Time-to-ready** is
simply the elapsed wall-clock from the first log line $t_0$ to the "Uvicorn running" line
$t_\text{ready}$:

$$\text{TTR} = t_\text{ready} - t_0 \approx 148\ \text{s}.$$

Most of it is engine deserialization and KV-cache reservation. It's the number you put in a
go-live runbook so the customer doesn't kill a pod that is merely still warming up.

### 3.2 Profile selection as constrained lexicographic optimization

A NIM ships one prebuilt engine per **profile** — a tuple of (GPU family, precision, tensor-
parallel size, tuning goal). At startup the `ngc_injector` picks one. This is a small
optimization problem: maximize an objective over a **feasible set** defined by hard
constraints.

**Feasibility.** A profile $p$ can run on detected hardware $h$ iff all hold:

$$
\underbrace{p.\text{gpu} \in \{h.\text{gpu},\ \text{"any"}\}}_{\text{family match or portable fallback}}
\;\wedge\;
\underbrace{p.\text{tp} \le h.\text{count}}_{\text{enough GPUs}}
\;\wedge\;
\underbrace{(p.\text{precision} = \text{fp8} \Rightarrow h.\text{cc} \ge 8.9)}_{\text{fp8 needs Hopper/Ada}}.
$$

The third constraint is the one that bites: **fp8 requires compute capability $\ge 8.9$**
(Ada/Hopper). The 8-bit float formats (E4M3 / E5M2 — 1 byte) are implemented in the tensor
cores only from sm89 onward. An A100 is sm80, so it is *infeasible* for any fp8 profile and
falls back to fp16 (2 bytes).

**Objective.** Among feasible profiles, pick the most optimized, ranked **lexicographically**
(compare the first component; break ties on the second; then the third):

$$\text{score}(p) = \big(\underbrace{\mathbb{1}[\text{TRT-LLM}]}_{\text{backend}},\;
\underbrace{\mathbb{1}[\text{fp8}]}_{\text{precision}},\;
\underbrace{p.\text{tp}}_{\text{parallelism}}\big),
\qquad p^\star = \arg\max_{p \,\in\, \text{feasible}} \text{score}(p).$$

This is why an H100 dev box resolves to `8835c31752fd` (TRT-LLM, fp8, tp1) but the A100
target resolves to a **different** profile `6f1ac2d40b77` (TRT-LLM, fp16, tp1). Two GPUs, two
optima, two different sets of engine files — which is the seed of the air-gap failure (§3.4).
When you can't risk a surprise, pin `NIM_MODEL_PROFILE` to fix $p^\star$ by hand.

### 3.3 The KV-cache concurrency ceiling

This is the most important arithmetic in the lab. Every in-flight request holds a slice of
the **KV cache** for each token of its context. The cache is a fixed token budget set at
startup. If each request can use up to `max_model_len` tokens, the number of full-context
requests that fit simultaneously is

$$C_\text{max} = \left\lfloor \frac{\text{kv\_cache\_tokens}}{\text{max\_model\_len}} \right\rfloor
= \left\lfloor \frac{117{,}440}{4{,}096} \right\rfloor = 28.$$

Offer more concurrency than this and the surplus requests are rejected with **HTTP 503**:

$$\text{rejected} = \max(0,\ \text{offered} - C_\text{max}) = 64 - 28 = 36.$$

Note that `max_num_seqs = 48` was *configured* — and it's a red herring. The **token budget
binds first**: the cache fills at 28 full-context sequences regardless of what the sequence
cap says. This connects to **Little's Law**, $L = \lambda W$ (mean concurrency = arrival rate
× mean service time): the server can sustain at most $L = C_\text{max}$ concurrent requests,
so a stable arrival rate must satisfy $\lambda \le C_\text{max}/W$. Push past it and the
queue (and latency) diverge — exactly the incident Chapter 6 instruments. The fixes, cheapest
first: cap client concurrency (or add a queue with backpressure), lower `max_model_len`,
raise `gpu_memory_utilization`, or add GPUs via tensor parallelism to enlarge the cache.
*Where* that 117,440-token budget comes from is the subject of Chapter 4.

### 3.4 Air-gapped caches: a set-difference problem

With `NIM_OFFLINE=1` there is no download fallback — the engine must be materialized from a
local cache. The required artifacts for the resolved profile must be a subset of what's on
the media. The missing files are a **set difference**:

$$\text{missing} = \text{required}(p^\star) \setminus \text{present}(\text{cache}).$$

The trap from §3.2: the cache was prepped on the H100 dev box (fp8 profile), but the A100
target resolves to the fp16 profile `6f1ac2d40b77`, whose files
(`config.json`, `rank0.engine`) were never copied. The cure is procedural — run
`nim download-to-cache --profile 6f1ac2d40b77` against the **customer's** SKU, or pin the
profile so selection is deterministic across machines.

---

## Chapter 4 — GPU Architecture for Inference

*Lab 04. The most math-heavy chapter: HBM, the KV-cache tax, the roofline model, and a
capacity planner that turns a target RPS + SLA into a GPU count.*

### 4.1 HBM: capacity vs bandwidth

A GPU's **High Bandwidth Memory** has two numbers that drive everything, corresponding to two
ways it can run out:

- **Capacity** $M$ (e.g. 80 GB) — *how much fits*. Weights and KV cache must both live here.
- **Bandwidth** $B$ (e.g. 3.35 TB/s on H100 SXM) — *how fast you can read it*.

The one-liner to carry through the chapter: **capacity (weights + KV) is a size problem;
decode speed is a bandwidth problem.** Define $u$ = usable fraction after framework overhead
(the labs use $u = 0.90$), so usable HBM is $M\cdot u$.

### 4.2 The KV-cache tax per token

To generate token $t+1$, attention needs the **key** and **value** vectors of every previous
token, at every layer. Those must stay resident in HBM. The per-token cost is fixed by the
model's shape:

$$\text{kv\_bytes/token} = \underbrace{2}_{K \text{ and } V} \times\; L \;\times\; H_{kv}
\;\times\; d_\text{head} \;\times\; b,$$

where $L$ = layers, $H_{kv}$ = **key/value** heads, $d_\text{head}$ = head dimension, and
$b$ = bytes per element. For Llama-3.1-8B at fp16:

$$2 \times 32 \times 8 \times 128 \times 2 = 131{,}072\ \text{bytes} = 128\ \text{KiB/token}.$$

The critical detail is $H_{kv}$, the **KV head count**, not the full attention-head count.
This is **Grouped-Query Attention (GQA)**: multiple query heads share one key/value head.
Llama-3.1 uses 8 KV heads to serve 32–64 attention heads, shrinking the cache $4\text{–}8\times$.
Without GQA, long context would be unaffordable. Note also that KV scales linearly with $L$,
so the 70B (80 layers) costs $2\times80\times8\times128\times2 = 320$ KiB/token — exactly
$2.5\times$ the 8B because it is $2.5\times$ deeper. ($L$, $H_{kv}$, $d_\text{head}$ are
architecture hyperparameters you look up in the model card; you don't derive them.)

### 4.3 From HBM budget to concurrency

Weights are paid first; the KV cache gets the rest. With weight bytes $W = P \cdot b$ ($P$ =
parameter count):

$$
\text{usable} = M u, \quad
\text{kv\_budget} = M u - W, \quad
\text{max\_tokens} = \left\lfloor\frac{M u - W}{\text{kv\_bytes/token}}\right\rfloor, \quad
\text{max\_seqs} = \left\lfloor\frac{\text{max\_tokens}}{\text{context\_len}}\right\rfloor.
$$

For 8B fp16 ($W = 16.06$ GB) at context 1280: an 80 GB H100 holds
$\lfloor(72 - 16.06)\,\text{GB} / 128\,\text{KiB}\rfloor \approx 426{,}788$ KV tokens =
**333** concurrent sequences. A 48 GB L40S holds ~207,061 tokens = **161** sequences — about
half, despite identical weights, because concurrency here is a pure *capacity* story. (This
`max_tokens` is the very `kv_cache_tokens = 117,440`-style budget that caused the 503s in
Chapter 3.)

### 4.4 The roofline: why "6× faster" is a half-truth

This is the conceptual heart of the chapter. **Arithmetic intensity** is FLOPs performed per
byte moved from memory:

$$I = \frac{\text{FLOPs}}{\text{bytes}}.$$

A kernel is **memory-bound** when $I$ is below the GPU's *ridge point* $I^\star = B_\text{peak}^{-1}\cdot F_\text{peak}$ — i.e. $I^\star = F_\text{peak}/B$ — and **compute-bound** above it.

**Decode** (generating one token at batch 1) reads every weight once ($P \cdot b$ bytes) but
does only $\approx 2P$ FLOPs with them, so

$$I_\text{decode} = \frac{2P}{P b} = \frac{2}{b} = 1\ \text{FLOP/byte (fp16)}.$$

The H100 SXM ridge is $F_\text{peak}/B = 989.5\,\text{TFLOP/s} / 3.35\,\text{TB/s} \approx 295$
FLOP/byte. Since $1 \ll 295$, single-stream decode is *deeply* memory-bound: the tensor cores
sit idle waiting on HBM. **Batching** is the fix — read the weights once, apply them to $B$
sequences — which raises intensity linearly, $I(B) = 2B/b$. Model throughput as the smaller of
the two ceilings:

$$
\text{tok/s}(B) = \min\!\Big(
\underbrace{\tfrac{B\,B_\text{w}}{W}}_{\text{memory-bound, grows with }B},\;
\underbrace{\tfrac{F_\text{peak}}{2P}}_{\text{compute ceiling, flat}}
\Big),
$$

where $B_\text{w}$ is bandwidth (renamed to avoid clashing with batch $B$). The two meet at
the **crossover batch**, found by setting them equal:

$$B^\star = \frac{W\,F_\text{peak}}{2P\,B_\text{w}} = \frac{b\,F_\text{peak}}{2 B_\text{w}}
\;\overset{b=2}{=}\; \frac{F_\text{peak}}{B_\text{w}} = 295.$$

For fp16 the crossover batch numerically equals the ridge intensity (295) — a tidy
coincidence of the factor-of-2. Below $B^\star$ batching buys near-linear throughput for free;
above it, throughput is pinned at the compute ceiling ($\approx 61{,}613$ tok/s for 8B on an
H100) and more batch only adds latency. This is why the marketing "6× faster" is
regime-specific (see §4.7).

### 4.5 Prefill and TTFT: the compute-bound half

Before the first token streams, the model processes the *entire* prompt at once — **prefill**
— doing $2P$ FLOPs for *each* of the $n_\text{in}$ input tokens. That is high intensity, so
prefill is **compute-bound**, and its time is the time-to-first-token floor:

$$\text{TTFT} \approx \frac{2P\,n_\text{in}}{F_\text{peak}}.$$

For 8B, 1024 input tokens: 16.6 ms on an H100 (fp16), 52.7 ms on an A100, 8.3 ms on an H100
in fp8. Here the H100 really *is* ~3.2× the A100 (the FLOPs ratio), and fp8 halves it again —
the exact opposite of the decode story. **Decode is a bandwidth problem; prefill is a compute
problem; the same GPU lives on both sides of the roofline within one request.** Long-context
workloads (RAG, summarization) are prefill-heavy and love FLOPs/fp8; chatty short-prompt
workloads are decode-heavy and love bandwidth.

### 4.6 The capacity planner

Combine everything into the number procurement wants — GPUs for a workload + SLA:

1. **Demand:** $\text{required\_tps} = \text{rps} \times \text{output\_tokens}$. (Chat at 50
   rps × 256 out = 12,800 tok/s.)
2. **Batch ceiling** is the smaller of two limits:
   - **KV-cache limit** — `max_seqs` from §4.3 (a memory constraint).
   - **ITL-SLA limit** — in the compute-bound region each decode step costs
     $2PB/F_\text{peak}$ seconds, so the largest batch within an inter-token-latency budget is
     $B_\text{itl} = \text{itl\_sla}\cdot F_\text{peak}/(2P)$.
   $$B = \min(B_\text{kv},\, B_\text{itl}).$$
3. **Supply:** $\text{per\_gpu\_tps} = \text{tok/s}(B)$ from §4.4.
4. **Count:** $\text{num\_gpus} = \lceil \text{required\_tps} / \text{per\_gpu\_tps}\rceil$,
   then check TTFT against its SLA.

For chat the SLA is loose, so **KV cache** (not latency) binds the batch; one H100 or A100
covers 50 rps, the L40S needs two. The binding constraint tells you the lever: when KV binds,
fp8 (which halves both weight and KV bytes) is the highest-leverage knob.

### 4.7 Two failures the model predicts

**The 70B that won't fit.** Weight bytes are $W = P b$. For 70B fp16, $W = 141$ GB — it does
not fit in one 80 GB GPU at all ($\text{fits} = W < Mu$ is false). This is a hard stop before
throughput even enters the picture. You must **shard the weights with tensor parallelism**
across $\text{tp}$ GPUs, each holding $W/\text{tp}$; the minimum is
$\text{tp} \ge \lceil W / (Mu)\rceil$ (rounded up to a supported size), giving tp=2 for fp16.
TP reads partial activations from every peer on *every* layer (an all-reduce per layer), which
is why those GPUs need **NVLink** (900 GB/s on SXM) rather than PCIe.

**Buying FLOPs for a memory-bound workload.** Compare A100 vs H100 SXM decoding 8B fp16:

| batch | A100 (tok/s) | H100 (tok/s) | ratio |
|------:|-------------:|-------------:|------:|
| 1 | 127 | 209 | **1.64×** (= bandwidth ratio) |
| 512 | 19,427 | 61,613 | **3.17×** (= FLOPs ratio) |

At batch 1 (memory-bound) the H100's extra FLOPs sit idle, so you only get its bandwidth
advantage (1.64×). The headline 3.17× appears only once batching pushes decode compute-bound.
Match the GPU to the regime your workload actually runs in — that *is* the roofline.

---

## Chapter 5 — Agents & Orchestration

*Lab 05. Why agents flake in production, the two mathematical sources of non-determinism, and
the concurrency model behind fast tool use.*

### 5.1 Routing as a function, and two sources of non-determinism

A router should be a *function*: the same ticket always routes the same way. A naive router
breaks this in two independent places.

**Sampling.** An LLM emits a probability distribution over the next token via a softmax with a
**temperature** $T$:

$$P(\text{token}_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}.$$

Temperature reshapes the distribution. As $T \to 0^+$, the mass concentrates entirely on the
$\arg\max$ — decoding becomes **greedy and deterministic**. As $T$ grows, the distribution
flattens toward uniform and sampling becomes stochastic. Consider two candidate routes with a
logit gap $\delta = z_1 - z_2$; their probability ratio is

$$\frac{P_1}{P_2} = \exp(\delta / T).$$

For an *ambiguous* ticket the gap $\delta \approx 0$, so $P_1/P_2 \approx 1$ — a near
**50/50 coin flip** at any $T > 0$. That is exactly Lab 05's showcase ticket t07 splitting
{technical: 6, billing: 6} over 12 runs. The fix for sampling variance is to set $T = 0$.

**Parsing.** A free-text answer ("Billing" vs "Route to the payments team") must be mapped to
an enum by a brittle keyword matcher, which silently mis-maps surface forms it doesn't
recognize. The fix is **structured output** — constrained decoding that masks any token which
would violate the schema, so the model can only emit a valid `Department`. This removes the
parser entirely.

The two fixes are orthogonal and you need **both**: structured output still *samples* among
valid values; $T = 0$ alone can still emit an unparseable string. Together they make routing a
pure function of the input — Lab 05's robust router collapses to {billing: 12}, distinct = 1,
and matches all 12 gold labels.

### 5.2 Composition of determinism in a state machine

A LangGraph agent is a directed graph: nodes do work, a shared state threads through, and
**conditional edges** branch on computed state (`gate` → blocked/router; `route_by_department`
→ the handler). The relevant theorem is trivial but load-bearing: *the composition of
deterministic functions is deterministic.* So once the router is pinned (§5.1), the entire
path `guardrails → router → handler` is reproducible. A conditional edge is only as
trustworthy as the state it branches on — which is why you make the routing decision
deterministic **before** the graph depends on it.

A **guardrail** node is a classifier placed at the graph's edge: it screens for prompt
injection and data exfiltration and short-circuits malicious inputs to a terminal `blocked`
node *before* any tool runs. Screening at the edge keeps the data plane clean.

### 5.3 Concurrency: the latency math of tool calls

A routed agent gathers context from several independent I/O-bound tools (account, invoices,
usage). Run sequentially, the total latency is the **sum**:

$$T_\text{seq} = \sum_{i=1}^{n} t_i.$$

But independent I/O can overlap. `asyncio.gather` issues all coroutines and awaits them
concurrently; while one is blocked on I/O the event loop runs the others, so the wall-clock is
the **max**, not the sum:

$$T_\text{conc} = \max_i t_i, \qquad \text{speedup} = \frac{\sum_i t_i}{\max_i t_i}.$$

For $n$ calls of roughly equal latency $t$, the speedup approaches $n$. Lab 05's three ~50 ms
lookups go from ~154 ms to ~51 ms — about 3×. (This is the I/O-bound analogue of Amdahl's law:
the serial fraction here is ~0, so speedup is near-linear in $n$.)

### 5.4 Proving it with traces

"It's deterministic now, trust me" doesn't survive a postmortem. **Distributed tracing**
records each step as a **span** with a start and end time; nested spans form a parent/child
tree. The signature of concurrency is visible directly in the numbers: the parent
`gather_context` span has wall-clock ≈ one child (~51 ms), not the sum of three (~150 ms),
because the children overlapped. The trace turns an assertion into evidence — and is the
on-ramp to the observability discipline of Chapter 6.

---

## Chapter 6 — MLOps & Platform

*Lab 06. The capstone: turning raw metrics into rates and percentiles, finding the breach
window, and the leading-indicator math that buys you runway.*

### 6.1 Counters vs gauges, and the rate

Triton and the DCGM Exporter publish metrics in the Prometheus exposition format. Two kinds
matter:

- **Counters** (e.g. `request_success_total`) only ever increase. Their *value* is
  meaningless; their *slope* is the signal.
- **Gauges** (e.g. `kv_cache_util`, `num_requests_waiting`) are instantaneous state.

To recover a rate (requests/sec, tokens/sec) from a counter you take a discrete derivative —
exactly what PromQL's `rate()` does:

$$\text{rate}[i] = \frac{c[i] - c[i-1]}{\Delta t}.$$

A counter reading 184,223 tells you nothing; an increase of 1,635 over a 30 s step tells you
you're serving ~55 req/s. (Counters are used precisely *because* they're monotonic and
reset-safe — a scrape that misses a sample still recovers the right average slope.)

### 6.2 Tail latency: why percentiles, not means

Latency distributions are heavy-tailed: the mean is dragged around by a few slow requests and
hides incidents. The SLA lives in the **tail**. The $p$-th percentile via linear interpolation
over sorted values $v_{(1)} \le \dots \le v_{(n)}$ is

$$k = (n-1)\,p, \quad f = \lfloor k\rfloor, \quad
\text{percentile}(p) = v_{(f)} + \big(v_{(f+1)} - v_{(f)}\big)\,(k - f).$$

In Lab 06 the *median* TTFT-p99 reading is a healthy 180 ms — which is exactly why a
mean/median dashboard sails past the incident — while the p95/p99 of the readings are 850/871
ms, far past the 500 ms SLO. You alert on the tail because that's where the SLA breaks and the
users are angry.

### 6.3 The SLO-breach window

An alert is not "p99 is high now"; it's "for how long, starting when." The breach window is the
contiguous time interval during which the metric violated its threshold $\theta$:

$$\text{window} = \big[\min\{t : x(t) > \theta\},\ \max\{t : x(t) > \theta\}\big].$$

In the lab: TTFT-p99 breaches 500 ms from $t = 1080$ s to $t = 1470$ s — a 6.5-minute incident
during which ~1,532 requests were dropped (the 503s from the Chapter 3 KV exhaustion, now
visible in production telemetry).

### 6.4 Leading vs lagging indicators, and alert lead time

The latency breach is a **lagging** indicator — by the time TTFT blows the SLO, users are
already hurting. A good alert fires on a **leading** indicator that *predicts* the breach via a
causal chain. That chain is, once again, **Little's Law**: as load rises, the KV cache
saturates → requests queue (queue depth $\uparrow$) → first-token latency rises. KV saturation
moves *first*. The **alert lead time** is the gap between the leading-indicator warning and the
SLO breach:

$$
t_\text{warn} = \min\{t : \text{kv\_util}(t) \ge 0.90\},\quad
t_\text{breach} = \min\{t : \text{ttft\_p99}(t) > 500\},\quad
\text{lead} = t_\text{breach} - t_\text{warn}.
$$

In the lab, $t_\text{warn} = 960$ s and $t_\text{breach} = 1080$ s, so the KV-cache alert buys
**120 s (2 minutes)** of runway — enough to autoscale or shed load *before* users notice. That
is the difference between a dashboard that *describes* an outage and one that *prevents* it:
board the leading indicator (KV utilization, queue depth) next to the symptom (latency), and
page on the leader.

### 6.5 Closing the loop: the quality regression gate

Latency isn't the only thing that regresses; answer quality drifts too. A nightly **RAGAS**
job scores **faithfulness** — the fraction of an answer's claims that are grounded in the
retrieved context:

$$\text{faithfulness} = \frac{\#\,\text{supported claims}}{\#\,\text{total claims}}.$$

A **regression gate** is threshold-crossing detection: fire on the first run whose faithfulness
falls below a gate $g$. In Lab 06, faithfulness slid from a 0.94 baseline to 0.83 (a drift of
0.11) by night 10, tripping the $g = 0.85$ gate. The gate is the *trigger* that closes the
MLOps control loop — **observe → evaluate → retrain (NeMo Customizer) → redeploy** — turning
evaluation from a dashboard you glance at into an automated feedback system.

---

## Epilogue — the one idea

Across six labs, one habit recurs: **find the binding constraint, then act on it.**

- Retrieval recall bounds answer quality, so measure retrieval first (Ch. 1) and remember the
  cascade can't beat its stage-1 ceiling (Ch. 2).
- The KV-cache *token budget* — not the sequence cap — bounds concurrency (Ch. 3), and that
  budget is itself set by HBM capacity, weight bytes, and the roofline (Ch. 4).
- A decision is only deterministic if every input to it is (Ch. 5), and a pipeline is only as
  fast as its slowest stage and as observable as its leading indicator (Ch. 6).

The notebooks make each of these concrete. Read a lab's chapter here, predict the number, then
run the cell and watch the theory hold.
