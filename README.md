# NVIDIA Partner Lab Series

A hands-on curriculum that teaches MLOps, RAG, NIM deployment, GPU architecture,
agents, and observability to startup engineering teams and NVIDIA partners.
Real-world scenarios first — concepts emerge from the problem, not the other
way around.

---

## What

Seven progressively harder labs (00–06), each a self-contained Jupyter notebook
plus tests:

| Lab | Topic | Scenario hook |
|-----|-------|----------------|
| 00 | Setup & validation | Get a green light before Lab 01 |
| 01 | RAG fundamentals | Startup ships basic RAG; agents get wrong answers |
| 02 | RAG reranking | Fintech migrated FAQ → PDFs; quality collapsed |
| 03 | NIM deployment & troubleshooting | ISV ships first on-prem NIM for an air-gapped federal client |
| 04 | GPU architecture for inference | ML engineer sizes multi-tenant inference infra |
| 05 | Agents & orchestration | Multi-agent customer service is non-deterministic in prod |
| 06 | MLOps & platform | MLOps team monitors a NIM cluster under production load |

The shared substrate:

- `shared/` — `nim_client`, `vector_store`, `mock_nim`, timing/display helpers
- `infra/` — Pulumi (Python) for `dev` (local mock) / `staging` (K8s) / `prod` (GPU)
- `Taskfile.yml` — every operation has a `task` target; no shell scripts, no Makefile

## Why

Most NIM/RAG tutorials show the happy path on a clean dataset and stop there.
That doesn't prepare engineers for the calls they actually take:

- "p99 latency spiked from 800ms to 4.2s after we added the reranker — demo is at 4pm"
- "CSAT dropped 12% the week we shipped RAG — retrieval is returning adjacent chunks"
- "Three NIMs on one H100 PCIe, KV cache exhaustion, air-gapped federal customer"

Each lab opens with a named failure mode, asks the learner to form a hypothesis,
then introduces the NVIDIA-stack concept that explains and fixes it. Every NIM
call is timed. Every fix is quantified (RAGAS delta, latency Δ, rank Δ).

The goal: when a learner finishes, they can debug a NIM RAG pipeline in
production, not just stand one up.

## How

### Prereqs

| Tool | Why | Install |
|------|-----|---------|
| `uv` | Python env + lockfile manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `task` | All automation lives in `Taskfile.yml` | `brew install go-task` |
| `pulumi` | Infra for staging/prod (optional for labs 01–02 + 04–05) | `brew install pulumi` |

A GPU is **not required**. A mock NIM server backed by real local models
(MiniLM embed + cross-encoder rerank + API-Catalog LLM proxy) ships in
`shared/mock_nim.py`.

### First-time setup

```bash
task setup                        # uv sync (root + infra), copy .env, init Pulumi dev stack
cp .env.example .env              # then edit: set NVIDIA_API_KEY or USE_MOCK_NIM=true
task lab:test LAB=00-setup        # validate environment
```

### Day-to-day

```bash
task mock:start                   # local mock NIM on :8099 (no GPU needed)
task lab:run LAB=01-rag-fundamentals
task lab:test LAB=01-rag-fundamentals
task lab:clean LAB=01-rag-fundamentals    # clear outputs before committing
```

### Working against real NIMs

```bash
task infra:secrets:set            # store NVIDIA_API_KEY in Pulumi
task infra:up STACK=staging       # provision K8s NIM cluster + monitoring
task infra:env STACK=staging      # pull stack outputs into root .env
task nim:health                   # confirm all endpoints reachable
```

`STACK=dev` provisions nothing — it just sets the mock-NIM endpoints in `.env`.
`STACK=staging` deploys real NIM Helm releases. `STACK=prod` adds full GPU
sizing and monitoring (DGX Cloud or AWS p4d).

### Repo layout

```
nvidia-partner-labs/
├── pyproject.toml         # root uv project
├── Taskfile.yml           # all automation
├── shared/                # importable helpers used by every lab
├── labs/00-setup ... 06-mlops-platform/
│   ├── README.md          # scenario / what-why-how
│   ├── lab.ipynb          # the lab notebook
│   ├── data/              # corpora + generate.py
│   └── tests/             # pytest + nbmake validation
├── solutions/             # instructor copies (gitignored on learner branch)
└── infra/                 # separate uv + Pulumi project
    ├── components/        # NimCluster, MonitoringStack, VectorStore, Networking
    └── Pulumi.{dev,staging,prod}.yaml
```

### Conventions in one screen

- **uv for everything**: never `pip`, never `python -m`. Run scripts with `uv run`.
- **task for everything**: never raw shell. New automation? Add a target.
- **NIM access via `shared.nim_client`** only. Notebooks never instantiate `openai.OpenAI` directly.
- **Latency on every call**: `result, ms = timed_call(client.embeddings.create, ...)`.
- **Real data, not toys**: HuggingFace datasets, synthetic data with a committed `generate.py`, or attributed NVIDIA docs.
- **Failure cells are required**: each lab contains at least one deliberate, explained failure.

Full conventions: see `CLAUDE.md` (not checked in by default).

### Troubleshooting

- `task setup` failed → check `uv --version` (need 0.4+) and `task --version` (need 3.40+).
- `task nim:health` shows all `false` → either set `NVIDIA_API_KEY` in `.env` or start `task mock:start`.
- Pulumi `dev` won't init → run `cd infra && uv run pulumi login --local` once.

### Next

Start with `labs/00-setup/README.md`, then `labs/01-rag-fundamentals/README.md`.
