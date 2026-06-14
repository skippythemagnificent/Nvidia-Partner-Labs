# Changelog

All notable changes to the NVIDIA Partner Lab Series are recorded here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added — Lab 06: MLOps & Platform (fully authored)

The observability + continuous-eval capstone. Fully analytical (no GPU, no NIM calls):
the learner analyzes a simulated Triton/DCGM Prometheus scrape with a built-in incident.

- **`labs/06-mlops-platform/data/generate.py`** — `metrics_raw.txt` (one Triton
  `:8002/metrics` + DCGM Exporter scrape in Prometheus exposition format, captured at
  the incident peak), `metrics.json` (61 samples at 30s over 30 min: cumulative
  counters + KV-cache / GPU / queue / TTFT / ITL gauges, with a load spike that
  saturates the KV cache and breaches the SLOs, then recovers), and `eval_runs.json`
  (14 nightly RAGAS runs whose faithfulness drifts below the 0.85 gate). Deterministic
  via anchor interpolation; the incident is engineered so a leading indicator precedes
  the latency breach by exactly 120s.
- **`labs/06-mlops-platform/build_lab.py`** + generated learner/solution notebooks —
  six Concept + Walkthrough exercises: `parse_prometheus`, `counter_rate` (PromQL
  `rate()`), `percentile` (tail latency), `slo_breach_window`, `alert_lead_time` (the
  capstone), and `regression_gate` (RAGAS → NeMo Customizer). Inline unicode sparklines
  show the incident shape with no plotting dependency. The narrative ties the incident
  back to Lab 03 (KV exhaustion → 503s) and Lab 04 (the roofline).
- **`labs/06-mlops-platform/conftest.py`** + **`tests/test_lab.py`** — 11 tests (fully
  offline): data invariants + monotonic counters, Prometheus parse, rate recovery,
  percentile interpolation, the breach window, the 120s lead time, the RAGAS gate, plus
  learner-keeps-stubs/solution-does-not and full solution-notebook execution. All green.
- **`labs/06-mlops-platform/README.md`** — rewritten to final What/Why/How.

Verified numbers: parsed scrape **13 metrics** (39 lines); RPS **20 → 55**, output
**~13,085 tok/s** peak, **3.5** failures/s (503s) at saturation; TTFT p99 distribution
p50 **180 ms** / p99 **871 ms** vs **500 ms** SLO; breach window **t=1080→1470s** (6m30s,
**1,532** requests dropped); KV-cache warning at **t=960s** leads the TTFT breach at
**t=1080s** → **120s** alert lead time; RAGAS faithfulness trips the 0.85 gate at run
**#10** (drifted 0.11 from 0.94). **Labs 00–06 `tests/` together: 66 passed — the full
series is complete.**

### Added — Lab 05: Agents & Orchestration (fully authored)

A LangGraph multi-agent lab made provably deterministic. Runs offline against a seeded
LLM/tool simulation (no GPU, no NIM, no key) using the *real* LangGraph, asyncio, and
OpenTelemetry libraries.

- **`shared/agent_sim.py`** (new shared module) — a controllable, seeded stand-in for
  an LLM NIM and backend services: `Department`/`RouteDecision`/`GuardrailResult`
  models, `MockLLMRouter` (temperature- and structured-output-aware, closeness-based
  sampling so only genuinely ambiguous tickets waver), `parse_department` (the brittle
  free-text parser), `ground_truth_department`, `guardrails_check` (prompt-injection +
  data-exfiltration), and async backend tools. The production swap to a real LLM NIM
  is documented in the module and the notebook.
- **`labs/05-agents-orchestration/data/generate.py`** — `tickets.json` (12 labeled
  support tickets, several deliberately ambiguous), `backend.json` (accounts/invoices/
  usage for the async tools), `adversarial.json` (8 guardrail probes, 4 attacks).
- **`labs/05-agents-orchestration/build_lab.py`** + generated learner/solution
  notebooks — five Concept + Walkthrough exercises: `naive_route` (reproduce the
  flake), `robust_route` (structured output + `temperature=0`), `gate` +
  `route_by_department` (LangGraph conditional edges), `gather_context`
  (`asyncio.gather`), and `traced_gather_context` (OpenTelemetry spans). Plus an
  adversarial guardrail-screening section.
- **`labs/05-agents-orchestration/conftest.py`** + **`tests/test_lab.py`** — 13 tests
  (fully offline): data invariants, naive non-determinism, structured+temp0 determinism,
  12/12 routing accuracy, guardrail block rate, async speedup, 4-span trace, a compiled
  LangGraph routes-and-blocks check, plus learner-keeps-stubs/solution-does-not and full
  solution-notebook execution. All green.
- **`labs/05-agents-orchestration/README.md`** — rewritten to final What/Why/How.

Verified (seed=7): the ambiguous showcase ticket t07 routes `{technical:6, billing:6}`
under naive free-text routing (**2 distinct decisions**), collapses to `{billing:12}`
(**1 decision**, the correct gold) under structured+`temperature=0`; structured routing
matches **12/12** gold labels; guardrails block **4/4** attacks (8/8 vs labels);
`asyncio.gather` is **~3×** faster than sequential tool calls; the traced gather emits
exactly **4 spans**. Labs 00–05 `tests/` together: **55 passed**.

### Added — Lab 04: GPU Architecture for Inference (fully authored)

A fully analytical lab (no GPU, no NIM calls): the learner builds a back-of-the-
envelope inference capacity model from datasheet tables.

- **`labs/04-gpu-architecture/data/generate.py`** — `gpus.json` (H100 SXM/PCIe,
  A100 80GB, L40S: HBM capacity + bandwidth, dense FP16/FP8 TFLOPS, NVLink, $/hr),
  `models.json` (Llama-3.1 8B + 70B shape — params and the attention dims that set
  KV-cache size), `workloads.json` (chat / RAG / summarization SLAs: RPS, token
  counts, TTFT + ITL budgets). Public datasheet / rule-of-thumb values, dense
  (non-sparsity) TFLOPS so the roofline math is honest.
- **`labs/04-gpu-architecture/build_lab.py`** + generated learner/solution notebooks —
  five Concept + Walkthrough exercises: `kv_cache_bytes_per_token`,
  `max_concurrent_sequences` (HBM budget → concurrency), `decode_tokens_per_s` +
  `crossover_batch` (the decode roofline), `prefill_ttft_ms`, and the capstone
  `capacity_planner` (target RPS + SLA → GPU count, binding constraint). Two failure
  sections: the 70B that won't fit on one 80 GB GPU (→ tensor parallelism + NVLink),
  and buying FLOPs for a memory-bound (decode) workload.
- **`labs/04-gpu-architecture/conftest.py`** + **`tests/test_lab.py`** — 12 tests
  (no mock NIM — fully offline): data invariants, KV-cache math, concurrency ceiling,
  the crossover=FLOP:byte-ridge identity, decode saturation, single-user decode tracks
  bandwidth not FLOPs, prefill tracks FLOPs, the chat planner, 70B weight-fit, plus
  learner-keeps-stubs/solution-does-not and full solution-notebook execution. All green.
- **`labs/04-gpu-architecture/README.md`** — rewritten from scaffold to final What/Why/How.

Verified numbers: KV **128 KiB/token** (8B fp16) / **320 KiB** (70B); concurrency
ceiling **333** seqs on an 80 GB GPU vs **161** on the 48 GB L40S; decode **crossover
batch = the GPU's FLOP:byte ridge** (295 H100 SXM, 153 A100, 419 L40S); decode
saturates at **61,613 tok/s** (8B fp16, H100 SXM); single-user H100/A100 ratio **1.64×**
(bandwidth) vs **3.17×** saturated (FLOPs); prefill TTFT **16.6 ms** (H100) vs **52.7 ms**
(A100); chat/8B needs **1** H100 or A100, **2** L40S; 70B fp16 weights **141 GB** don't
fit one 80 GB GPU (tp≥2). Labs 00+01+02+03+04 `tests/` together: **42 passed**.

### Added — Lab 03: NIM Deployment & Troubleshooting (fully authored)

An offline lab (no GPU, no NIM calls): the learner parses real-shaped deployment
artifacts and writes the diagnosis functions an on-call SE would.

- **`labs/03-nim-deployment/data/generate.py`** — emits four log samples to
  `data/logs/` (a clean H100 startup; a `kubectl describe` `ImagePullBackOff`; a
  KV-cache-exhaustion 503 burst; an air-gapped offline-load abort), plus
  `profiles.json` (a 6-entry `nim list-model-profiles` manifest with per-profile
  cache artifacts), `cache_manifest.json` (the H100/fp8 set someone wrongly burned
  to the air-gap media), and `deployment.json`. Hand-authored so timestamps, token
  counts, and profile IDs stay fixed.
- **`labs/03-nim-deployment/build_lab.py`** + generated learner/solution notebooks —
  five Concept + Walkthrough exercises: `parse_startup` (time-to-ready measure),
  `select_profile` (reproduces the `ngc_injector` auto-selection — GPU family,
  `tp ≤ count`, fp8 needs cc ≥ 8.9, most-optimized wins), `diagnose` (a
  `KNOWN_FAILURES` playbook → structured `Diagnosis`), `max_concurrent_requests`
  (KV-cache concurrency ceiling), and `missing_artifacts` (the air-gap cache gap).
  Three failure sections drive `diagnose` against each broken log.
- **`labs/03-nim-deployment/conftest.py`** + **`tests/test_lab.py`** — 12 tests
  (no mock NIM needed — fully offline): data invariants, startup parsing,
  four profile-selection cases, the failure classifier, the KV-cache ceiling, the
  air-gap artifact gap, learner-keeps-stubs/solution-does-not, and full
  solution-notebook execution. All green.
- **`labs/03-nim-deployment/README.md`** — rewritten from scaffold to final
  What/Why/How.

Verified numbers: time-to-ready **148.043 s**, first-token **243 ms**, H100→fp8
profile `8835c31752fd` vs A100→fp16 `6f1ac2d40b77`, KV ceiling **117440 // 4096 =
28** concurrent (vs 64 offered → 36 over the line → 503s), and the air-gap cache
missing exactly the A100 profile's `config.json` + `rank0.engine`. Labs 00+01+02+03
`tests/` run together: **30 passed**.

### Added — Lecture deck (Labs 01–02)

- **`decks/build_deck.py`** + **`task deck:build`** — generates
  `decks/rag-fundamentals-and-reranking.pptx`, a 24-slide 16:9 PowerPoint lecture
  introducing the RAG fundamentals (Lab 01) and reranking (Lab 02) concepts:
  the problem framing, the chunk→embed→index→retrieve→generate pipeline, the
  asymmetric embedder, cosine search, retrieval@K, the bi-encoder-vs-cross-encoder
  table, two-stage reranking, both labs' failure modes, and the NeMo Retriever NIM
  stack. Every slide carries a speaker-notes script. Added `python-pptx` as a dev
  dependency. The `.pptx` is a build artifact (gitignored; rebuild with
  `task deck:build`); the generator is the source of truth.

### Added — Lab 02: RAG Reranking (fully authored)

- **`shared/nim_client.py`** — added `rerank(query, passages, top_n=...)`, a POST to
  the reranking NIM's `/ranking` endpoint (reranking is not part of the OpenAI
  schema). `_api_key()` now treats an empty `NVIDIA_API_KEY` like a missing one (the
  OpenAI client rejects `""`; the mock ignores auth).
- **`labs/02-rag-reranking/data/generate.py`** — 10 hand-authored retail-banking
  policy documents → `corpus.json` (44 sentence-chunks) + a 12-question `eval.json`,
  several phrased so the bi-encoder ranks an adjacent chunk above the answer.
- **`labs/02-rag-reranking/build_lab.py`** + generated learner/solution notebooks —
  Concept + Walkthrough teaching for the new work: bi-encoder vs cross-encoder, the
  `rerank_candidates` call, `to_ranked_chunks` (the `RankedChunk` model with
  `rank_delta`), and an MRR metric. Two reliable failure demos (over-retrieve `k1`
  too small loses the answer; reranking the whole corpus is O(n)) and an optional
  key-gated faithfulness comparison.
- **`labs/02-rag-reranking/conftest.py`** + **`tests/test_lab.py`** — 9 tests:
  data invariants, reranking-changes-order, top rerank score, MRR non-regression,
  the over-retrieve floor, mock latency < 500 ms, learner-keeps-stubs/solution-does-
  not, and full solution-notebook execution. All green.

Verified narrative (mock bi-encoder MiniLM + mock cross-encoder ms-marco-MiniLM):
two-stage reranking lifts **MRR 0.882 → 0.958**, hit@1 0.83 → 0.92, with a max rank
delta of 7 and no correct answer demoted. The README/notebook state plainly that the
compact mock reranker understates the production NV-RerankQA-Mistral-4B-v3 gain.

### Fixed — test harness

- **`conftest.py`** (new, repo root) — inserts the project root on `sys.path` so
  `import shared` works under pytest. The editable install's `.pth` root does not
  land on `sys.path`, and pytest's prepend import mode puts the test directory
  there instead of the repo root, so `shared` imported fine under plain `python`
  (cwd on path) but failed under pytest. This fixes
  `labs/00-setup/tests/test_setup.py::test_shared_package_importable`.
- **`Taskfile.yml`** — `lab:test` now skips `nbmake` when a lab has no `lab.ipynb`
  (e.g. `00-setup` is tests-only) instead of failing with "file not found", and
  guards the `tests/` step on the directory existing. `task lab:test LAB=00-setup`
  now passes 4/4.
- **`pyproject.toml`** — added `[tool.pytest.ini_options] addopts =
  "--import-mode=importlib"` so each lab's identically-named `tests/test_lab.py`
  can be collected in one run without a module-name clash. Labs 00+01+02 run
  together: **18 passed**.

### Changed — Pulumi runs in local mode

- **`Taskfile.yml`** — added a top-level `env:` block pointing Pulumi at a
  project-local filesystem backend (`PULUMI_BACKEND_URL=file://{{.ROOT_DIR}}/infra/.pulumi`)
  with a default dev `PULUMI_CONFIG_PASSPHRASE`. No Pulumi Cloud account or
  `pulumi login` is required; the env-var backend overrides any existing cloud
  login for task-invoked commands. `infra:init` now `mkdir -p .pulumi` first (the
  file backend won't create its own bucket directory). Verified: `task infra:init`
  creates the `dev` stack in `file:///…/infra/.pulumi` (user `trey`, local).
- **`Taskfile.yml`** — `infra:init` is now idempotent (`pulumi stack select dev
  2>/dev/null || pulumi stack init dev`), removing the noisy "stack already
  exists" error on re-runs of `task setup`.
- **`Taskfile.yml`** — infra uv calls now run through a `UVX = 'env -u VIRTUAL_ENV
  uv'` var so an activated shell venv no longer triggers uv's "VIRTUAL_ENV does
  not match the project environment path" warning (Task's own `env:` cannot fix
  this — it does not override an already-exported variable). `task setup` now runs
  clean under an activated venv.
- **`.env`** — removed a stray bare Pulumi Cloud access token (`pul-…`) on line 1
  that had no `KEY=` and broke the dotenv parser for every `task`. Unneeded in
  local mode.
- **`.env.example`** — documented the local backend and the
  `PULUMI_CONFIG_PASSPHRASE` override.
- State directory `infra/.pulumi/` is covered by the existing `.pulumi/`
  `.gitignore` rule.

### Added — Lab 01: RAG Fundamentals (fully authored)

- **`labs/01-rag-fundamentals/data/generate.py`** — deterministic, self-validating
  generator that writes:
  - `corpus.json` — 50 hand-authored NVIDIA dev-blog / NIM-deployment excerpts
    (`d01`–`d50`), realistic technical prose.
  - `eval.json` — 12 labeled questions (`q01`–`q12`), each with a verbatim
    `answer_span` that drives the retrieval@K hit-rate metric.
- **`labs/01-rag-fundamentals/lab.ipynb`** — learner notebook with TODO stubs,
  following the CLAUDE.md cell structure (scenario → setup → exercises →
  deliberate failures → challenge → takeaways).
- **`solutions/01-rag-fundamentals/lab.ipynb`** — complete, nbmake-clean notebook
  (verified to execute top-to-bottom in mock mode).
- **`labs/01-rag-fundamentals/conftest.py`** + **`tests/test_lab.py`** — 5 tests
  (data invariants, cosine-space scores, the fixed-vs-sentence chunking claim, and
  full solution-notebook execution). Includes an in-process mock NIM fixture so
  tests run with no GPU and no `NVIDIA_API_KEY`.

- **Instructional content** — every exercise section now has a **Concept**
  explanation (what chunking/embeddings/cosine search/hit-rate/sentence chunking
  are and why they matter) plus a **Walkthrough** that traces the algorithm on a
  tiny worked example (e.g. `fixed_size_chunks("ABCDEFG", 4)`, a sentence-packing
  table) and then gives numbered implementation steps — teaching the concept and
  enabling the task without revealing the solution code.
- **`labs/01-rag-fundamentals/build_lab.py`** — the notebook generator is now
  committed (was a throwaway script). It emits the learner and solution notebooks
  from one source so their shared markdown and paired exercise cells stay in sync;
  regenerate with `uv run python labs/01-rag-fundamentals/build_lab.py`.

Verified narrative (mock embedder, MiniLM 384-dim):

- Retrieval@3 hit-rate **50% (fixed 140-char chunks) → 75% (sentence-aware)**;
  `q08` flips from miss to hit.
- Three deliberate failures land as designed: (1) omitted `input_type` on
  NV-EmbedQA (silent under the symmetric mock), (2) chunk-boundary split (`q08`),
  (3) cosine-but-wrong trap — *"What makes TTFT slow on a warm NIM?"* ranks the
  cold-start doc `d21` #1 over the correct `d10`, motivating the Lab 02 reranker.

### Fixed

- **`shared/vector_store.py`** — ChromaDB collection now created with
  `metadata={"hnsw:space": "cosine"}` so `score = 1 - distance` is true cosine
  similarity. Previously defaulted to squared-L2, which does not equal cosine for
  the normalized vectors NV-EmbedQA and the mock embedder return.

### Changed

- **`pyproject.toml`** — added `[tool.ruff] extend-exclude = ["*.ipynb"]`.
  Teaching notebooks legitimately trip `E402` (the repo-root `sys.path` bootstrap
  runs before imports) and `F841` (stub starter variables the learner consumes);
  ruff now lints the `.py` sources, not authored notebooks.
- **`labs/01-rag-fundamentals/README.md`** — removed the "scaffold only" status;
  updated the "You'll measure" line to reflect retrieval@K hit-rate + latency,
  with RAGAS faithfulness deferred to Lab 02.

### Decisions

- **Lab vs. solution nbmake.** CLAUDE.md says both "no solution code in lab
  notebooks" and "notebooks pass nbmake cleanly" — contradictory for a stubbed
  notebook. Resolution: the committed `lab.ipynb` keeps TODO stubs (so
  `task lab:test LAB=01` is red until a learner completes it — the first stub
  raises `AssertionError: Complete <fn> before continuing`), and the **solution**
  notebook is the nbmake-validated artifact, executed by `test_lab.py` via
  `nbclient`.
- **Notebook generation.** Both notebooks are emitted from a single committed
  `nbformat` builder (`labs/<lab>/build_lab.py`) so the stub and solution stay in
  sync. Always edit the builder and regenerate — never hand-edit the `.ipynb`
  files.
- **Metric scope.** Lab 01 measures retrieval@K hit-rate + per-call latency;
  rigorous RAGAS faithfulness moves to Lab 02 (where the cross-encoder reranker
  lives). Generation is guarded on `NVIDIA_API_KEY` so CI stays offline.

### Known / deferred

- Pre-existing `shared/{utils,mock_nim}.py` have a `ruff format` drift (missing
  blank line after the module docstring) that predates this work; left untouched.
  (`shared/nim_client.py` and `shared/vector_store.py` were edited this session and
  are now formatted.)
- `task lab:test:all` runs `nbmake` over the learner notebooks, which keep TODO
  stubs by design and therefore fail until completed; per-lab `tests/` (which
  execute the solution notebooks) are the green signal.
