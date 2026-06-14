"""Author the Lab 02 notebooks from a single source.

Emits two notebooks that stay in sync:
  - labs/02-rag-reranking/lab.ipynb        — learner copy (exercise cells stubbed)
  - solutions/02-rag-reranking/lab.ipynb   — completed, nbmake-clean copy

Each `code(solution, stub)` cell carries both forms; markdown is shared. Edit the
cells below and regenerate:

    uv run python labs/02-rag-reranking/build_lab.py

Same pattern as Lab 01: edit this file, never the .ipynb files directly.
"""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/02-rag-reranking/lab.ipynb"
SOL = ROOT / "solutions/02-rag-reranking/lab.ipynb"

CELLS = []
def md(s): CELLS.append(("md", s))
def code(sol, stub=None): CELLS.append(("code", (sol, stub)))

# ── Scenario ─────────────────────────────────────────────────────────────────
md("""# Lab 02 — RAG Reranking

## Scenario

A fintech startup moved its customer-support knowledge base from a flat FAQ to a
set of **PDF policy documents** — account agreements, fee schedules, dispute
procedures. Answer quality collapsed. The retriever finds the right *document*, but
it keeps surfacing a neighbouring sentence ("the monthly fee is $12…") instead of
the one that actually answers the question ("…waived with a $1,500 minimum
balance"). Support agents get *adjacent* context, not answers — the exact symptom
you diagnosed in Lab 01, now at the ranking level rather than the chunking level.

You'll build the production fix: a **two-stage retrieve-then-rerank** pipeline.
Stage 1 over-retrieves cheap candidates with the bi-encoder (NV-EmbedQA). Stage 2
re-scores them with a **cross-encoder** reranker (NV-RerankQA-Mistral-4B-v3) that
reads each query/passage *pair together*. Everything runs through
`shared/nim_client` against the mock NIM or the NVIDIA API Catalog.""")

# ── Setup ────────────────────────────────────────────────────────────────────
md("""## Setup

Loads the financial-policy corpus and labeled eval set, wires up the embedding +
reranking clients, and defines the `RankedChunk` model that carries a chunk through
both stages so you can see how its rank changes.""")

code(r'''import json
import os
import re
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "shared").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.nim_client import get_embed_client, rerank
from shared.utils import timed_call, display_ranked_chunks, display_metrics_table
from shared.vector_store import VectorStore

load_dotenv()

DATA = REPO_ROOT / "labs/02-rag-reranking/data"
CORPUS = json.loads((DATA / "corpus.json").read_text())
EVAL = json.loads((DATA / "eval.json").read_text())

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
RERANK_MODEL = "nvidia/nv-rerankqa-mistral-4b-v3"

SENTENCE_MAX_CHARS = 320     # sentence-aware chunking budget (from Lab 01)
OVER_RETRIEVE_K1 = 20        # stage 1: cheap candidates from the bi-encoder
RERANK_K2 = 5                # stage 2: keep this many after reranking

embed_client = get_embed_client()
USE_MOCK = os.environ.get("USE_MOCK_NIM", "false").lower() == "true"


class RankedChunk(BaseModel):
    """A candidate chunk carried through both retrieval stages."""
    text: str
    doc_id: str
    embed_score: float          # stage-1 cosine similarity
    embed_rank: int             # stage-1 position (1 = best)
    rerank_score: float | None = None   # stage-2 cross-encoder logit
    rerank_rank: int | None = None       # stage-2 position (1 = best)

    @property
    def rank_delta(self) -> int | None:
        """How far the chunk moved: positive = promoted by the reranker."""
        if self.rerank_rank is None:
            return None
        return self.embed_rank - self.rerank_rank


print(f"corpus: {len(CORPUS)} docs | eval: {len(EVAL)} questions | mock={USE_MOCK}")''')

# ── 1. Recap: bi-encoder index ───────────────────────────────────────────────
md("""## 1 · Recap — the bi-encoder index

### Concept

Stage 1 is exactly the retriever you built in Lab 01: sentence-aware chunks, each
embedded as a `passage` with NV-EmbedQA and stored for cosine search; the query is
embedded as a `query`. These helpers are given so you can focus on what's new —
reranking. Run the cell to chunk the policy corpus and build the index.""")

code(r'''def sentence_chunks(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            chunks.append(current.strip())
            current = s
        else:
            current = (current + " " + s).strip()
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_corpus(corpus, chunker):
    out = []
    for d in corpus:
        for j, piece in enumerate(chunker(d["text"])):
            out.append({"id": f'{d["id"]}#{j}', "doc_id": d["id"], "text": piece})
    return out


def embed_texts(texts, input_type):
    result, ms = timed_call(
        embed_client.embeddings.create,
        model=EMBED_MODEL, input=texts, extra_body={"input_type": input_type},
    )
    return [d.embedding for d in result.data]


PERSIST = tempfile.mkdtemp(prefix="chroma_lab02_")
chunks = chunk_corpus(CORPUS, lambda t: sentence_chunks(t, SENTENCE_MAX_CHARS))
index = VectorStore(collection="lab02", persist_dir=PERSIST)
index.upsert(
    ids=[c["id"] for c in chunks],
    texts=[c["text"] for c in chunks],
    embeddings=embed_texts([c["text"] for c in chunks], "passage"),
    metadatas=[{"doc_id": c["doc_id"]} for c in chunks],
)


def retrieve(query: str, k: int):
    q_emb = embed_texts([query], "query")[0]
    return index.search(q_emb, top_k=k)


print(f"{len(CORPUS)} docs -> {len(chunks)} chunks indexed")''')

md("""**Expected output:**
```
10 docs -> 44 chunks indexed
```""")

# ── bi-encoder weakness ───────────────────────────────────────────────────────
md("""### See the bi-encoder's blind spot

Take **q09**: *"What is the difference between the APR and the interest rate on a
personal loan?"* The answer is one specific sentence in the loan policy. Look at
where the bi-encoder ranks it.

**Predict first:** the loan document has several sentences that all mention "APR"
and "interest rate". Which one will cosine rank highest — the one that *defines the
difference*, or just the one with the most word overlap?""")

code(r'''q09 = next(it for it in EVAL if it["id"] == "q09")
print("Q:", q09["question"])
print("answer span:", repr(q09["answer_span"]), "\n")
for rank, r in enumerate(retrieve(q09["question"], 5), 1):
    here = "  <-- the answer" if q09["answer_span"].lower() in r.text.lower() else ""
    print(f"  bi-rank {rank} | cos={r.score:.3f} | {r.metadata['doc_id']} | {r.text[:60]}{here}")''')

md("""**Expected output** (the answer chunk is *not* rank 1 — sibling chunks that
share "APR"/"interest rate" outrank it):
```
bi-rank 1 | cos=0.810 | p09 | A personal loan is an unsecured installment loan repaid in f
bi-rank 2 | cos=0.653 | p09 | Your APR is fixed for the life of the loan and is determined
bi-rank 3 | cos=0.557 | p09 | Missing payments can lower your credit score and, after nine  <-- the answer
...
```
The right *document* is found, but the answer sits at **rank 3** behind adjacent
chunks. (Chunks are multi-sentence, 320-char units as in Lab 01, so the answer
sentence — "…the APR also includes any origination fee…" — lives *inside* the
rank-3 chunk, past the 60-character preview.) Concatenate only the top-1 into a
prompt and the model never sees the answer.""")

# ── 2. bi vs cross ────────────────────────────────────────────────────────────
md("""## 2 · Bi-encoder vs cross-encoder

### Concept

The two model types trade off speed against accuracy:

- A **bi-encoder** (the embedding model) encodes the query and each passage
  *independently* into vectors, then compares them with cosine. The passage vectors
  are computed once and indexed, so search is fast and scales to millions of chunks
  — but the query and passage never "see" each other, so the score is blind to
  nuances like *which* of two similar sentences actually answers the question.
- A **cross-encoder** (the reranker) feeds the query and one passage through the
  model *together* and outputs a single relevance score. Because the two texts
  interact at every layer, it's far more accurate — but it must run once per
  (query, passage) pair and nothing can be precomputed, so it's far too expensive to
  run over a whole corpus.

The production answer is to **combine them**: let the cheap bi-encoder narrow
millions of chunks to a handful of candidates, then let the expensive cross-encoder
re-score just those. That's the two-stage pipeline.

### Exercise — call the reranker

`shared.nim_client.rerank(query, passages, top_n=...)` POSTs to the reranking NIM
and returns a list of `{"index": i, "logit": score}` dicts, **sorted most- to
least-relevant**, where `index` points back into the `passages` list you passed and
a higher `logit` means more relevant.

**Walkthrough.** Suppose you pass three passages and get back
`[{"index": 2, "logit": 4.1}, {"index": 0, "logit": -2.3}, {"index": 1, "logit": -9.0}]`.
That means `passages[2]` is the best match, then `passages[0]`, then `passages[1]`.
The list order *is* the new ranking; the `index` is how you map each entry back to
the passage it scored.

Implement `rerank_candidates`:

1. Call `rerank(query, passages, top_n=top_n)` and return its result unchanged.

(That's the whole call — the mapping back to chunks happens in the next exercise.)""")

code(r'''def rerank_candidates(query: str, passages: list[str], top_n: int) -> list[dict]:
    """Return the reranker's [{index, logit}, ...] for `passages`, best first."""
    return rerank(query, passages, model=RERANK_MODEL, top_n=top_n)


demo = rerank_candidates(q09["question"], [r.text for r in retrieve(q09["question"], 5)], top_n=5)
print("rankings (index into the 5 candidates, best first):")
for j, row in enumerate(demo, 1):
    print(f"  rerank {j} | candidate #{row['index']} | logit={row['logit']:.3f}")''',
r'''def rerank_candidates(query: str, passages: list[str], top_n: int) -> list[dict]:
    """Return the reranker's [{index, logit}, ...] for `passages`, best first."""
    # TODO: Call shared.nim_client.rerank(query, passages, model=RERANK_MODEL,
    # top_n=top_n) and return the rankings list it produces.
    rankings = None  # replace this line
    assert rankings is not None, "Complete rerank_candidates before continuing"
    return rankings


demo = rerank_candidates(q09["question"], [r.text for r in retrieve(q09["question"], 5)], top_n=5)
print("rankings (index into the 5 candidates, best first):")
for j, row in enumerate(demo, 1):
    print(f"  rerank {j} | candidate #{row['index']} | logit={row['logit']:.3f}")''')

md("""**Expected output** (the candidate that was bi-rank 3 — the real answer — now
scores highest; logits are unbounded, often negative for poor matches):
```
rankings (index into the 5 candidates, best first):
  rerank 1 | candidate #2 | logit=1.xx
  rerank 2 | candidate #0 | logit=-3.xx
  ...
```""")

# ── 3. two-stage / RankedChunk ────────────────────────────────────────────────
md("""## 3 · Assemble the two-stage pipeline

### Concept

Now stitch the stages together and keep a record of how each chunk moved. The
`RankedChunk` model (defined in Setup) stores both ranks, and its `rank_delta`
property reports `embed_rank - rerank_rank` — **positive means the reranker promoted
the chunk**. This is the number that makes the reranker's effect visible.

### Walkthrough — `to_ranked_chunks`

You're handed two things:

- `candidates`: the stage-1 results **in bi-encoder order**. `candidates[0]` is
  embed-rank 1, `candidates[1]` is embed-rank 2, and so on. Each has `.text`,
  `.score` (cosine), and `.metadata["doc_id"]`.
- `rankings`: the reranker output `[{index, logit}, ...]` in **rerank order**.
  `rankings[0]` is rerank-rank 1; its `index` says which candidate it is.

Walk the `rankings` list with an enumerate starting at 1 — that loop position *is*
the rerank rank. For each entry, pull the matching candidate with
`candidates[entry["index"]]`, and build a `RankedChunk` carrying both ranks:

| rerank pos `j` | `entry` | candidate = `candidates[entry["index"]]` | embed_rank | rerank_rank |
|----------------|---------|-------------------------------------------|------------|-------------|
| 1 | `{"index": 2, "logit": 1.3}` | the 3rd stage-1 candidate | 2+1 = **3** | **1** |
| 2 | `{"index": 0, "logit": -3.1}` | the 1st stage-1 candidate | 0+1 = **1** | **2** |

The first row is the win: a chunk the bi-encoder ranked 3rd is now 1st
(`rank_delta = 3 - 1 = +2`).

**Step by step** — for each `(j, entry)` from `enumerate(rankings, start=1)`:

1. `cand = candidates[entry["index"]]`.
2. Build a `RankedChunk(text=cand.text, doc_id=cand.metadata["doc_id"],
   embed_score=cand.score, embed_rank=entry["index"] + 1, rerank_score=entry["logit"],
   rerank_rank=j)`.
3. Append it to the output list and return the first `top_k` of them.""")

code(r'''def to_ranked_chunks(candidates, rankings, top_k):
    """Map reranker output back onto candidates as RankedChunks, best first."""
    ranked = []
    for j, entry in enumerate(rankings, start=1):
        cand = candidates[entry["index"]]
        ranked.append(RankedChunk(
            text=cand.text,
            doc_id=cand.metadata["doc_id"],
            embed_score=cand.score,
            embed_rank=entry["index"] + 1,
            rerank_score=entry["logit"],
            rerank_rank=j,
        ))
    return ranked[:top_k]


def two_stage(query: str, k1: int = OVER_RETRIEVE_K1, k2: int = RERANK_K2):
    candidates = retrieve(query, k1)                                   # stage 1
    rankings = rerank_candidates(query, [c.text for c in candidates], top_n=k1)  # stage 2
    return to_ranked_chunks(candidates, rankings, k2)


ranked = two_stage(q09["question"])
display_ranked_chunks(ranked, query=q09["question"], top_n=RERANK_K2)''',
r'''def to_ranked_chunks(candidates, rankings, top_k):
    """Map reranker output back onto candidates as RankedChunks, best first."""
    ranked = []
    for j, entry in enumerate(rankings, start=1):
        # TODO: pull the candidate this ranking refers to, then build a RankedChunk
        # carrying BOTH ranks. embed_rank is the candidate's 1-based stage-1
        # position (entry["index"] + 1); rerank_rank is j; rerank_score is
        # entry["logit"]; embed_score and text/doc_id come from the candidate.
        ranked_chunk = None  # replace this line
        assert ranked_chunk is not None, "Complete to_ranked_chunks before continuing"
        ranked.append(ranked_chunk)
    return ranked[:top_k]


def two_stage(query: str, k1: int = OVER_RETRIEVE_K1, k2: int = RERANK_K2):
    candidates = retrieve(query, k1)                                   # stage 1
    rankings = rerank_candidates(query, [c.text for c in candidates], top_n=k1)  # stage 2
    return to_ranked_chunks(candidates, rankings, k2)


ranked = two_stage(q09["question"])
display_ranked_chunks(ranked, query=q09["question"], top_n=RERANK_K2)''')

md("""**Expected output** (the `Δ` column shows the rank change; the chunk holding
the answer — bi-rank 3 — jumps to rank 1, `Δ = +2`):
```
                       Top 5 chunks for: '...APR and the interest rate...'
┏━━━━━━┳━━━━┳━━━━━━━┳━━━━━━━━┳━ ... ━┓
┃ Rank ┃  Δ ┃ Embed ┃ Rerank ┃ Text  ┃
┡━━━━━━╇━━━━╇━━━━━━━╇━━━━━━━━╇━ ... ━┩
│    1 │ +2 │ 0.557 │  8.045 │ Missing payments can lower your credit score and, after ninety… │
│    2 │ -1 │ 0.810 │  7.353 │ A personal loan is an unsecured installment loan repaid in fix… │
│  ... │    │       │        │ …                                                              │
└──────┴────┴───────┴────────┴─ ... ─┘
```
The reranker's logits are unbounded (here ~8 for strong matches, negative for poor
ones). The answer chunk that cosine buried at rank 3 is now rank 1.""")

# ── 4. measure ────────────────────────────────────────────────────────────────
md("""## 4 · Measure — does reranking actually help?

### Concept

To prove the pipeline helps (and by how much), score it on the whole eval set with
**Mean Reciprocal Rank (MRR)**. For each question, find the rank of the first chunk
that contains the labeled `answer_span`; its *reciprocal rank* is `1 / rank` (rank 1
→ 1.0, rank 3 → 0.33, not found → 0). MRR is the average across questions — a single
number that rewards putting the answer near the top. We compute it for the
bi-encoder alone and for the two-stage pipeline and compare.

### Walkthrough — `mrr`

`rank_fn(question)` returns an ordered list of chunk-like objects (each with `.text`),
best first. For one question:

- Walk the results with `enumerate(..., start=1)`; the position is the rank.
- The first result whose `.text` contains `answer_span` (case-insensitive) gives
  reciprocal rank `1 / position`; stop there.
- If no result contains the span, this question contributes `0`.

Average those reciprocal ranks over all of `eval_items`.

**Step by step:**

1. For each `item`, call `rank_fn(item["question"])`.
2. Loop the results with `enumerate(results, start=1)`; on the first
   `item["answer_span"].lower() in r.text.lower()`, add `1 / rank` to a running total
   and `break`.
3. Return `total / len(eval_items)`.""")

code(r'''def mrr(eval_items, rank_fn) -> float:
    """Mean reciprocal rank of the answer_span across eval_items."""
    total = 0.0
    for item in eval_items:
        for rank, r in enumerate(rank_fn(item["question"]), start=1):
            if item["answer_span"].lower() in r.text.lower():
                total += 1.0 / rank
                break
    return total / len(eval_items)


bi_only = lambda q: retrieve(q, RERANK_K2)              # stage 1 only, top-5
two_stage_fn = lambda q: two_stage(q, OVER_RETRIEVE_K1, RERANK_K2)

display_metrics_table({
    "MRR — bi-encoder only": mrr(EVAL, bi_only),
    "MRR — two-stage rerank": mrr(EVAL, two_stage_fn),
}, title="Retrieval quality: reranking off vs on")

deltas = [c.rank_delta for q in EVAL for c in two_stage_fn(q["question"]) if c.rank_delta]
print("max rank delta across queries:", max(deltas) if deltas else 0)''',
r'''def mrr(eval_items, rank_fn) -> float:
    """Mean reciprocal rank of the answer_span across eval_items."""
    total = 0.0
    for item in eval_items:
        # TODO: walk rank_fn(item["question"]) with enumerate(start=1); on the first
        # result whose .text contains item["answer_span"] (lowercased), add 1/rank
        # to `total` and stop. Questions with no hit contribute 0.
        raise NotImplementedError("Complete mrr before continuing")
    return total / len(eval_items)


bi_only = lambda q: retrieve(q, RERANK_K2)              # stage 1 only, top-5
two_stage_fn = lambda q: two_stage(q, OVER_RETRIEVE_K1, RERANK_K2)

display_metrics_table({
    "MRR — bi-encoder only": mrr(EVAL, bi_only),
    "MRR — two-stage rerank": mrr(EVAL, two_stage_fn),
}, title="Retrieval quality: reranking off vs on")

deltas = [c.rank_delta for q in EVAL for c in two_stage_fn(q["question"]) if c.rank_delta]
print("max rank delta across queries:", max(deltas) if deltas else 0)''')

md("""**Expected output:**
```
        Retrieval quality: reranking off vs on
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric                    ┃  Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ MRR — bi-encoder only     │ 0.8819 │
│ MRR — two-stage rerank    │ 0.9583 │
└───────────────────────────┴────────┘
max rank delta across queries: 7
```
MRR climbs and no correct answer is demoted; the largest single move is a chunk the
reranker promoted **7 places** into the top-5. The aggregate gain looks modest
because the **mock reranker is a tiny ~22M-param cross-encoder**; production
NV-RerankQA-Mistral-4B-v3 is a 4B model and widens the gap substantially — the
*pipeline* is identical either way.""")

# ── failure 1: k1 too small ───────────────────────────────────────────────────
md("""### ⚠️ Failure 1 — over-retrieving too little

Stage 2 can only reorder what stage 1 handed it. If you over-retrieve too few
candidates, an answer the bi-encoder ranked just outside that window is gone before
the reranker ever sees it. Watch q04, whose answer the bi-encoder puts at rank 4.""")

code(r'''q04 = next(it for it in EVAL if it["id"] == "q04")
span = q04["answer_span"].lower()

for k1 in (3, 20):
    ranked = two_stage(q04["question"], k1=k1, k2=RERANK_K2)
    found = next((c.rerank_rank for c in ranked if span in c.text.lower()), None)
    print(f"k1={k1:>2} -> answer rerank position: {found}  "
          f"({'recovered' if found else 'LOST — never entered stage 2'})")''')

md("""**Expected output:**
```
k1= 3 -> answer rerank position: None  (LOST — never entered stage 2)
k1=20 -> answer rerank position: 2  (recovered)
```
**Root cause.** With `k1=3` the answer (bi-rank 4) is outside the candidate set, so
no reranker can save it — the failure is a *recall* ceiling set by stage 1. The
standard rule of thumb is to over-retrieve generously (4×–10× your final K) so the
reranker has the real answer to promote.""")

# ── failure 2: cost ───────────────────────────────────────────────────────────
md("""### ⚠️ Failure 2 — why not just rerank everything?

If the cross-encoder is more accurate, why not skip stage 1 and rerank the whole
corpus? Because its cost is **linear in the number of passages** — one model pass
per (query, passage) pair, nothing cacheable. Measure the stage-2 time for 20
candidates versus the entire corpus.""")

code(r'''cand_texts = [c.text for c in retrieve(q09["question"], OVER_RETRIEVE_K1)]
all_texts = [c["text"] for c in chunks]

_, ms_few = timed_call(rerank, q09["question"], cand_texts, model=RERANK_MODEL)
_, ms_all = timed_call(rerank, q09["question"], all_texts, model=RERANK_MODEL)

print(f"rerank {len(cand_texts):>3} candidates : {ms_few:6.0f} ms")
print(f"rerank {len(all_texts):>3} chunks (all): {ms_all:6.0f} ms")
print(f"this corpus is tiny; at 1M chunks reranking everything is ~{1_000_000 // len(all_texts):,}x this — infeasible per query")''')

md("""**Expected output** (absolute numbers vary; the point is the ratio and the
extrapolation):
```
rerank  20 candidates :    ... ms
rerank  44 chunks (all):    ... ms
this corpus is tiny; at 1M chunks reranking everything is ~22,727x this — infeasible per query
```
**Root cause.** Reranking is O(n) in candidates with no precomputation. Two-stage
keeps stage 2 bounded to `k1` regardless of corpus size — that's the whole point of
over-retrieve-then-rerank.""")

# ── optional RAGAS ────────────────────────────────────────────────────────────
md("""## 5 · (Optional) Faithfulness — does better retrieval mean better answers?

Reranking is a means to an end: a more *faithful* generated answer. A full RAGAS
faithfulness evaluation needs a judge LLM (and an `NVIDIA_API_KEY`), so this cell is
optional and skips cleanly without one. It contrasts the answer the model gives from
bi-encoder-only context versus reranked context for q09.""")

code(r'''def answer_from(query, ranked_or_results):
    from shared.nim_client import get_llm_client
    def _doc(r):  # works for both RankedChunk (.doc_id) and SearchResult (.metadata)
        return r.doc_id if hasattr(r, "doc_id") else r.metadata["doc_id"]
    ctx = "\n\n".join(f"[{_doc(r)}] {r.text}" for r in ranked_or_results)
    msgs = [
        {"role": "system", "content": "Answer ONLY from the context; cite the [doc_id]. "
         "If the answer isn't present, say so."},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"},
    ]
    resp = get_llm_client().chat.completions.create(
        model="meta/llama-3.1-8b-instruct", messages=msgs, temperature=0.0, max_tokens=160)
    return resp.choices[0].message.content


if os.environ.get("NVIDIA_API_KEY"):
    print("— bi-encoder-only context —")
    print(answer_from(q09["question"], retrieve(q09["question"], RERANK_K2)))
    print("\n— reranked context —")
    print(answer_from(q09["question"], two_stage(q09["question"])))
else:
    print("[skipped] set NVIDIA_API_KEY to generate and compare grounded answers.")''')

md("""**Expected output** (with a key): the reranked-context answer states the APR
includes the origination fee — grounded in the chunk reranking lifted to the top —
while the bi-encoder-only answer is vaguer or hedges, because its top chunk was an
adjacent sentence. Without a key the cell prints a skip message.""")

# ── challenge ─────────────────────────────────────────────────────────────────
md("""## Challenge

1. **Over-retrieve sweep.** Plot two-stage MRR as `k1` goes 3 → 40 (keep `k2=5`).
   Where does MRR plateau? That knee is your cheapest safe over-retrieve ratio.
2. **Latency budget.** Measure stage-1 (`retrieve`) vs stage-2 (`rerank`) latency
   separately across the eval set. If your SLA is 800 ms, what's the largest `k1`
   you can afford?
3. **Find the wins.** Print the per-query `rank_delta` for the top reranked chunk.
   Which questions does reranking help most, and what do they have in common with the
   "adjacent chunk" failure from the scenario?""")

md("""## Key takeaways

- **Bi-encoders retrieve, cross-encoders rerank.** The bi-encoder is fast and
  scalable but scores query and passage in isolation; the cross-encoder reads them
  together and is far more accurate but O(n) and uncacheable.
- **Two-stage = over-retrieve then rerank.** Stage 1 narrows the corpus cheaply;
  stage 2 re-scores a bounded candidate set. It buys cross-encoder accuracy at
  bi-encoder scale.
- **Stage 2 can't fix stage 1's misses.** Over-retrieve generously (4×–10× your
  final K) or the answer never enters the rerank window.
- **`rank_delta` makes the win measurable.** Here MRR rose 0.88 → 0.96 with a tiny
  mock reranker; production NV-RerankQA-Mistral-4B-v3 widens the gap.

**References**
- NeMo Retriever — reranking: https://docs.nvidia.com/nemo/retriever/
- NV-RerankQA-Mistral-4B-v3: https://build.nvidia.com/explore/retrieval
- RAGAS (faithfulness): https://docs.ragas.io/""")

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
