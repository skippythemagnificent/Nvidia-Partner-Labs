"""Author the Lab 01 notebooks from a single source.

Emits two notebooks that stay in sync:
  - labs/01-rag-fundamentals/lab.ipynb        — learner copy (exercise cells stubbed)
  - solutions/01-rag-fundamentals/lab.ipynb   — completed, nbmake-clean copy

Each `code(solution, stub)` cell carries both forms; markdown is shared. Edit the
cells below and regenerate:

    uv run python labs/01-rag-fundamentals/build_lab.py

Why a generator instead of editing the .ipynb files directly: the learner and
solution notebooks differ only in their exercise cells, and hand-editing two JSON
notebooks in lockstep drifts. Keep this file as the source of truth.
"""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# Repo root relative to this file: labs/01-rag-fundamentals/build_lab.py -> root
ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/01-rag-fundamentals/lab.ipynb"
SOL = ROOT / "solutions/01-rag-fundamentals/lab.ipynb"

# (kind, payload). For "code": (solution_src, stub_src_or_None).
CELLS = []
def md(s): CELLS.append(("md", s))
def code(sol, stub=None): CELLS.append(("code", (sol, stub)))

# ── 1. Scenario ────────────────────────────────────────────────────────────
md("""# Lab 01 — RAG Fundamentals

## Scenario

A Series A developer-tools startup shipped a customer-support RAG over their
product docs **last Monday**. CSAT dropped 12% in the first week. Support agents
say the assistant keeps returning *adjacent* passages — close to the answer, but
not the answer — and occasionally answers a question the user never asked. You're
the NVIDIA solutions engineer on the 9am call. This lab rebuilds their pipeline
from scratch — chunk, embed, index, retrieve, assemble — so you can see exactly
where a basic RAG cracks, and measure each fix.

You'll work against **NV-EmbedQA-E5-v5** (NeMo Retriever) through the NVIDIA API
Catalog, or the local mock NIM if you have no GPU/key. Everything goes through
`shared/nim_client` — never a raw `openai.OpenAI()`.""")

# ── 2. Setup ──────────────────────────────────────────────────────────────
md("""## Setup

Loads the corpus and the labeled evaluation set, initializes the embedding client,
and detects whether you're on the API Catalog or the mock NIM. Runs in under a few
seconds.""")

code(r'''import json
import os
import re
import sys
import tempfile
from pathlib import Path

# Put the repo root (the dir holding `shared/`) on the path so `import shared`
# works no matter where this notebook is launched from.
REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "shared").is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from shared.nim_client import get_embed_client, get_llm_client
from shared.utils import timed_call, display_metrics_table
from shared.vector_store import VectorStore

load_dotenv()

DATA = REPO_ROOT / "labs/01-rag-fundamentals/data"

CORPUS = json.loads((DATA / "corpus.json").read_text())
EVAL = json.loads((DATA / "eval.json").read_text())

# NV-EmbedQA on the API Catalog; the mock ignores the model name.
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
LLM_MODEL = "meta/llama-3.1-8b-instruct"

# Retrieval knobs (named constants, not magic numbers).
FIXED_CHUNK_CHARS = 140      # naive baseline: split every N characters
SENTENCE_MAX_CHARS = 320     # sentence-aware budget
RETRIEVAL_TOP_K = 3          # chunks concatenated into the context

embed_client = get_embed_client()
USE_MOCK = os.environ.get("USE_MOCK_NIM", "false").lower() == "true"

print(f"corpus: {len(CORPUS)} docs | eval: {len(EVAL)} questions")
print(f"embed endpoint: {embed_client.base_url}")
print(f"mock mode: {USE_MOCK}")''')

# ── 3. Chunking ─────────────────────────────────────────────────────────────
md("""## 1 · Chunking — choosing the retrieval unit

### Concept

Retrieval works on *chunks*, not whole documents. Before anything is indexed, each
document is sliced into pieces, and a piece — a chunk — is **the smallest unit the
retriever can return.** That makes chunking a retrieval decision, not a formatting
detail: if the sentence that answers a question is split across two chunks, then no
single result can ever contain the whole answer, no matter how good the embedding
model is.

The simplest strategy is **fixed-size chunking**: cut the text every *N* characters
(or tokens), ignoring sentence and paragraph boundaries. It's trivial to implement
and perfectly even, which is why it's the default people reach for first — and it's
what the startup used. Its weakness is exactly that it ignores meaning: it will
happily cut through the middle of the sentence that holds the answer. We implement
it now precisely so we can reproduce that bug and measure it later.

An optional `overlap` repeats the last few characters of one chunk at the start of
the next, softening boundary effects. We default it to `0` here.

### Walkthrough

Picture a window of `size` characters sliding left-to-right across the text, jumping
forward by a fixed *step* each time. With `overlap=0`, step equals `size`, so the
windows are back-to-back with no gap and no repeat.

Trace it on `"ABCDEFG"` with `size=4, overlap=0` (step = 4):

| iteration | `i` | slice `text[i:i+4]` | next `i` |
|-----------|-----|---------------------|----------|
| 1         | 0   | `"ABCD"`            | 4        |
| 2         | 4   | `"EFG"`  (runs short at the end — fine) | 8 |
| stop      | 8   | `8 >= len("ABCDEFG")` → done | — |

Result: `["ABCD", "EFG"]`. Now set `overlap=1` (step = 3): you'd get
`["ABCD", "DEFG", "G"]` — each window re-reads the previous window's last character.
That repeated tail is what cushions a fact sitting right on a boundary.

**Step by step:**

1. Start at index `i = 0` and an empty `chunks` list.
2. Append the slice `text[i : i + size]` to `chunks`.
3. Advance `i` by `size - overlap` (the step).
4. Repeat from step 2 while `i < len(text)`; stop when `i` reaches the end.
5. Return `chunks`.

A `while i < len(text):` loop is the natural shape. The trailing chunk being shorter
than `size` is expected — don't pad it.""")

code(r'''def fixed_size_chunks(text: str, size: int, overlap: int = 0) -> list[str]:
    """Split `text` into windows of `size` characters, stepping `size - overlap`."""
    chunks: list[str] = []
    step = size - overlap
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += step
    return chunks


def chunk_corpus(corpus: list[dict], chunker) -> list[dict]:
    """Apply `chunker` to every doc; return chunk records with stable ids."""
    out: list[dict] = []
    for d in corpus:
        for j, piece in enumerate(chunker(d["text"])):
            out.append({"id": f'{d["id"]}#{j}', "doc_id": d["id"], "text": piece})
    return out


fixed_chunks = chunk_corpus(CORPUS, lambda t: fixed_size_chunks(t, FIXED_CHUNK_CHARS))
print(f"{len(CORPUS)} docs -> {len(fixed_chunks)} fixed-size chunks "
      f"({FIXED_CHUNK_CHARS} chars each)")
print("sample:", fixed_chunks[0]["text"])''',
r'''def fixed_size_chunks(text: str, size: int, overlap: int = 0) -> list[str]:
    """Split `text` into windows of `size` characters, stepping `size - overlap`."""
    # TODO: Walk `text` in steps of (size - overlap), slicing `size` chars each
    # time, until you reach the end. Append each slice to `chunks`.
    chunks: list[str] = []
    chunks = None  # replace this line
    assert isinstance(chunks, list), "Complete fixed_size_chunks before continuing"
    return chunks


def chunk_corpus(corpus: list[dict], chunker) -> list[dict]:
    """Apply `chunker` to every doc; return chunk records with stable ids."""
    out: list[dict] = []
    for d in corpus:
        for j, piece in enumerate(chunker(d["text"])):
            out.append({"id": f'{d["id"]}#{j}', "doc_id": d["id"], "text": piece})
    return out


fixed_chunks = chunk_corpus(CORPUS, lambda t: fixed_size_chunks(t, FIXED_CHUNK_CHARS))
print(f"{len(CORPUS)} docs -> {len(fixed_chunks)} fixed-size chunks "
      f"({FIXED_CHUNK_CHARS} chars each)")
print("sample:", fixed_chunks[0]["text"])''')

md("""**Expected output:**
```
50 docs -> 150 fixed-size chunks (140 chars each)
sample: NVIDIA NIM is a set of containerized inference microservices that wrap an optimized model engine behind a stable, OpenAI-compatible HTTP API
```""")

# ── 4. Embedding ────────────────────────────────────────────────────────────
md("""## 2 · Embedding the corpus

### Concept

An **embedding** turns text into a vector so that semantically similar texts land
near each other in vector space. We embed every chunk once at index time, then at
query time embed the question and find the nearest chunk vectors. The model here is
NeMo Retriever's **NV-EmbedQA-E5-v5**, served through a NIM behind an
OpenAI-compatible API.

NV-EmbedQA is **asymmetric**: questions and passages are encoded by different
"sides" of the model, so you must tell it which role each input plays with
`input_type` — `"passage"` for documents you index, `"query"` for the user's
question. Skipping it puts both into the same subspace and quietly wrecks retrieval
(you'll trigger that failure on purpose in a moment). Because `input_type` is a
NVIDIA extension to the standard OpenAI schema, it rides along in `extra_body`
rather than as a top-level argument.

We wrap the call in `timed_call` (from `shared.utils`) so every NIM call prints its
latency — a habit this curriculum keeps everywhere.

### Walkthrough

You're wrapping one HTTP call to the embedding NIM. Three things have to happen:
send the texts with the right role, time the call, and unpack the vectors.

Trace the data as it flows:

- **Send.** `embed_client.embeddings.create(model=EMBED_MODEL, input=texts,
  extra_body={"input_type": input_type})`. `input` accepts a list, so all the chunks
  go in one batched request (far faster than one call each). `input_type` is the
  asymmetric role from the Concept above.
- **Time.** Don't call it directly — pass it to `timed_call`, which runs the
  function and hands back `(result, elapsed_ms)`:
  `result, ms = timed_call(embed_client.embeddings.create, model=EMBED_MODEL, input=texts, extra_body={"input_type": input_type})`.
  Note you pass the function *and its arguments* to `timed_call`, not the result of
  calling it.
- **Unpack.** The response copies the OpenAI shape: `result.data` is a list aligned
  one-to-one with `texts`, and each element has an `.embedding` attribute holding a
  `list[float]`. So `result.data[0].embedding` is the vector for `texts[0]`.

**Step by step:**

1. Build the `timed_call(...)` line above, assigning `result, ms`.
2. Return `[d.embedding for d in result.data]` — one vector per input text.

The `print(...)` line and the `dim = len(result.data[0].embedding)` read are already
written for you just below the TODO, so you only need those two lines.""")

code(r'''def embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed `texts` with NV-EmbedQA. `input_type` is 'query' or 'passage'."""
    result, ms = timed_call(
        embed_client.embeddings.create,
        model=EMBED_MODEL,
        input=texts,
        extra_body={"input_type": input_type},
    )
    dim = len(result.data[0].embedding)
    print(f"[embed] {ms:.0f}ms | n={len(texts)} | type={input_type} | dim={dim}")
    return [d.embedding for d in result.data]


_sample = embed_texts([c["text"] for c in fixed_chunks[:8]], "passage")
print("first vector preview:", [round(x, 3) for x in _sample[0][:5]], "...")''',
r'''def embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed `texts` with NV-EmbedQA. `input_type` is 'query' or 'passage'."""
    # TODO: Call embed_client.embeddings.create wrapped in timed_call so latency
    # is visible. Pass model=EMBED_MODEL, input=texts, and
    # extra_body={"input_type": input_type}. Return a list[list[float]] of the
    # embeddings (one per input text).
    result, ms = None, 0.0  # replace this line
    assert result is not None, "Complete embed_texts before continuing"
    dim = len(result.data[0].embedding)
    print(f"[embed] {ms:.0f}ms | n={len(texts)} | type={input_type} | dim={dim}")
    return [d.embedding for d in result.data]


_sample = embed_texts([c["text"] for c in fixed_chunks[:8]], "passage")
print("first vector preview:", [round(x, 3) for x in _sample[0][:5]], "...")''')

md("""**Expected output** (dim is 1024 on the real NV-EmbedQA, 384 on the mock; the
*first* mock call is slow — several seconds — because it lazily loads the local
embedding model, after which calls are tens of milliseconds):
```
[embed] 121ms | n=8 | type=passage | dim=384
first vector preview: [-0.041, 0.012, -0.068, 0.033, 0.05] ...
```""")

# ── 5. Index + search ───────────────────────────────────────────────────────
md("""## 3 · Index and cosine search

### Concept

A **vector store** indexes the chunk embeddings and answers nearest-neighbour
queries fast. We use ChromaDB locally (the same interface swaps to pgvector in
Lab 06). Similarity is measured by **cosine similarity** — the cosine of the angle
between two vectors — which compares *direction* and ignores magnitude. For the
L2-normalized vectors NV-EmbedQA returns, that's the right metric; the store is
configured for cosine space so `VectorStore.search` can report
`score = 1 - cosine_distance`, i.e. cosine similarity in roughly the 0.4–0.8 range
for relevant hits.

The two helpers below are written for you, but read them — they encode the core
asymmetric-retrieval rule from section 2:

- `build_index` embeds every chunk as a **`passage`** and upserts it.
- `retrieve` embeds the incoming question as a **`query`**, then returns the
  top-K nearest chunks.

This is the whole first stage of RAG: embed-as-passage to index, embed-as-query to
search.""")

code(r'''PERSIST = tempfile.mkdtemp(prefix="chroma_lab01_")

def build_index(chunks: list[dict], name: str) -> VectorStore:
    store = VectorStore(collection=name, persist_dir=PERSIST)
    embs = embed_texts([c["text"] for c in chunks], "passage")
    store.upsert(
        ids=[c["id"] for c in chunks],
        texts=[c["text"] for c in chunks],
        embeddings=embs,
        metadatas=[{"doc_id": c["doc_id"]} for c in chunks],
    )
    return store


def retrieve(store: VectorStore, query: str, k: int):
    q_emb = embed_texts([query], "query")[0]   # note: query role, not passage
    return store.search(q_emb, top_k=k)


fixed_index = build_index(fixed_chunks, "lab01-fixed")
hits = retrieve(fixed_index, "How do I authenticate Docker to pull NIM images?", 3)
for r in hits:
    print(f"{r.metadata['doc_id']} | cos={r.score:.3f} | {r.text[:70]}")''')

md("""**Expected output** (scores are cosine similarities in roughly 0.4–0.8):
```
[embed] ... | type=query | dim=...
d03 | cos=0.62 | NIM container images live in the NVIDIA NGC private registry a
d03 | cos=0.55 | ...
...
```""")

# ── 6. Failure 1: input_type ─────────────────────────────────────────────────
md("""### ⚠️ Deliberate failure 1 — the asymmetric embedder

What happens if you forget `input_type`? On the **real NV-EmbedQA** the call is
rejected — the field is required. The dangerous case is the *silent* one: a
symmetric model (or the mock) happily returns a vector of the right shape, and the
only symptom is bad search results. Run it and read the explanation.""")

code(r'''try:
    bad = embed_client.embeddings.create(model=EMBED_MODEL, input=["test query"])
    print(f"[no error] returned a {len(bad.data[0].embedding)}-dim vector — "
          "but is it in the right subspace?")
    print()
    print("Root cause: NV-EmbedQA-E5-v5 is ASYMMETRIC. Query and passage vectors")
    print("live in different subspaces. Omitting input_type either (a) is rejected")
    print("by the real NIM, or (b) silently encodes the query with the wrong role,")
    print("collapsing cosine similarity into noise. The failure is invisible until")
    print("recall craters. Always pass input_type explicitly on every call.")
except Exception as e:
    print(f"[EXPECTED FAILURE] {type(e).__name__}: {e}")
    print()
    print("Root cause: NV-EmbedQA requires input_type. Set it to 'query' for the")
    print("user's question and 'passage' for indexed documents — always explicit.")''')

md("""See corpus docs `d11` and `d12` — they describe exactly this asymmetry. The
startup's first bug was here: they embedded queries and passages identically.""")

# ── 7. Baseline hit-rate ─────────────────────────────────────────────────────
md("""## 4 · Measure the baseline — retrieval@K hit-rate

### Concept

You can't improve what you don't measure, and RAG quality starts at *retrieval*:
the generator can only be as good as the context it's handed. So before touching
the generation step, we measure the retriever directly.

**Retrieval@K hit-rate** is the fraction of questions for which the answer actually
appears in the top-K retrieved chunks. Our evaluation set (`eval.json`) makes this
checkable: each question carries a verbatim `answer_span` — the exact text that
must be present for the question to be answerable. We retrieve the top-K chunks,
concatenate their text, and check whether the span is in there. It's a strict,
deterministic proxy for "could the model possibly have answered this?" — no LLM and
no API key required, so it's perfect for a baseline you'll re-run after each change.

### Walkthrough

The metric loops over every labeled question and asks one yes/no question: *did the
answer text make it into the retrieved context?*

Trace a single item — say `q01`, whose `answer_span` is `"$oauthtoken"`:

1. `results = retrieve(store, it["question"], k)` → the top-K chunks for that
   question. Each result has a `.text`.
2. Glue their texts into one string and lowercase it:
   `context = " ".join(r.text for r in results).lower()`. (Joining *then* checking
   means the span counts as found even if it's in the 2nd or 3rd chunk, not just the
   1st.)
3. Test membership: `"$oauthtoken" in context`? If yes → it's a **hit**
   (`hits += 1`). If no → record the miss (`misses.append(it["id"])`).

Do that for all 12 questions and the hit-rate is `hits / len(eval_items)`.

**Step by step:**

1. Inside the loop, build the lowercased `context` from `results` (step 2 above) —
   this is the single line the TODO asks for.
2. The `if it["answer_span"].lower() in context:` check, the counters, and the
   `return (hits / len(eval_items), misses)` are already scaffolded for you.

Because the check is exact-substring and deterministic, you can trust the number and
re-run it after every change with no API key.""")

code(r'''def hit_rate(store: VectorStore, eval_items: list[dict], k: int):
    """Return (hit_rate, list_of_missed_question_ids)."""
    hits = 0
    misses: list[str] = []
    for it in eval_items:
        results = retrieve(store, it["question"], k)
        context = " ".join(r.text for r in results).lower()
        if it["answer_span"].lower() in context:
            hits += 1
        else:
            misses.append(it["id"])
    return hits / len(eval_items), misses


fixed_hr, fixed_misses = hit_rate(fixed_index, EVAL, RETRIEVAL_TOP_K)
print(f"baseline (fixed {FIXED_CHUNK_CHARS}-char chunks) hit-rate@{RETRIEVAL_TOP_K}: "
      f"{fixed_hr:.0%}")
print("missed:", fixed_misses)''',
r'''def hit_rate(store: VectorStore, eval_items: list[dict], k: int):
    """Return (hit_rate, list_of_missed_question_ids)."""
    hits = 0
    misses: list[str] = []
    for it in eval_items:
        results = retrieve(store, it["question"], k)
        # TODO: Join the retrieved chunk texts into one lowercased `context`
        # string, then count a hit when it["answer_span"].lower() is a substring
        # of it. Otherwise append it["id"] to `misses`.
        context = None  # replace this line
        assert context is not None, "Complete hit_rate before continuing"
        if it["answer_span"].lower() in context:
            hits += 1
        else:
            misses.append(it["id"])
    return hits / len(eval_items), misses


fixed_hr, fixed_misses = hit_rate(fixed_index, EVAL, RETRIEVAL_TOP_K)
print(f"baseline (fixed {FIXED_CHUNK_CHARS}-char chunks) hit-rate@{RETRIEVAL_TOP_K}: "
      f"{fixed_hr:.0%}")
print("missed:", fixed_misses)''')

md("""**Expected output:**
```
baseline (fixed 140-char chunks) hit-rate@3: 50%
missed: ['q01', 'q04', 'q05', 'q06', 'q07', 'q08']
```
Half the questions can't be answered — the retriever never surfaces the span.""")

# ── 8. Failure 2: boundary split ─────────────────────────────────────────────
md("""### ⚠️ Deliberate failure 2 — a chunk boundary hides the answer

Look closely at **q08**: *"Why is my very first NIM request so slow when the GPU is
sitting idle?"* The answer lives in doc `d21`. Watch what fixed-size chunking did
to it.

**Before you run the next cell — write your hypothesis:** the right document *is*
in the corpus and clearly relevant. Why would the retriever still miss the answer?""")

code(r'''q08 = next(it for it in EVAL if it["id"] == "q08")
print("Q:", q08["question"])
print("answer span needed:", repr(q08["answer_span"]))
print()
print("Top retrieved chunks (fixed chunking):")
for r in retrieve(fixed_index, q08["question"], RETRIEVAL_TOP_K):
    has = q08["answer_span"].lower() in r.text.lower()
    print(f"  {r.metadata['doc_id']} cos={r.score:.3f} span_here={has} | {r.text[:60]}")

print()
print("How d21 got shredded by 140-char windows:")
for i, piece in enumerate(fixed_size_chunks(
        next(d for d in CORPUS if d["id"] == "d21")["text"], FIXED_CHUNK_CHARS)):
    marker = "  <-- holds the answer" if q08["answer_span"].lower() in piece.lower() else ""
    print(f"  d21#{i}: {piece!r}{marker}")''')

md("""**Root cause.** Fixed-size windows split `d21` mid-thought. The lexically
*strongest* fragment (`d21#0`, "The first request ... is far slower ...") wins the
cosine match and gets retrieved — that's the **adjacent chunk** the support agents
complained about. But the sentence that actually answers the question ("the
latency you see on request one is dominated by model load") landed in a *different*
fragment that scored below the top-K cutoff. The answer was in the corpus the whole
time; chunking buried it.""")

# ── 9. Fix: sentence chunking ────────────────────────────────────────────────
md("""## 5 · Fix — sentence-aware chunking

### Concept

The diagnosis from failure 2 was that fixed-size windows cut *through* sentences,
stranding the answer in a low-scoring fragment. The fix is to chunk on meaning
instead of character count: **sentence-aware chunking** never splits mid-sentence.
It packs whole sentences into a chunk until adding the next one would exceed a size
budget (`max_chars`), then starts a fresh chunk. A complete thought — and the
sentence that answers a question — stays intact and retrievable. (Going further and
splitting where the *topic* shifts is "semantic chunking"; the sentence-aware
version here captures most of the benefit for a fraction of the complexity.)

Note we keep the **same embedding model and the same K** — only the chunking
changes — so any movement in hit-rate is attributable to chunking alone.

### Walkthrough

The split is given — `re.split(r"(?<=[.!?])\\s+", text.strip())` returns a list of
whole sentences (it splits *after* `.`, `!`, or `?`). Your job is to greedily pack
those sentences into chunks that never exceed `max_chars`, and crucially **never cut
a sentence in half**.

Trace three sentences `["AAAA.", "BBBB.", "CCCC."]` (each 5 chars) with
`max_chars=12`, accumulating into `current`:

| sentence | `current` before | fits? (`len(current)+len(s)+1 <= 12`) | action | `chunks` |
|----------|------------------|----------------------------------------|--------|----------|
| `AAAA.`  | `""`             | current empty → just take it           | `current="AAAA."` | `[]` |
| `BBBB.`  | `"AAAA."` (5)    | 5+5+1=11 ≤ 12 → yes                     | `current="AAAA. BBBB."` | `[]` |
| `CCCC.`  | `"AAAA. BBBB."` (11) | 11+5+1=17 > 12 → no                 | flush, then `current="CCCC."` | `["AAAA. BBBB."]` |
| (end)    | `"CCCC."`        | —                                      | flush leftover | `["AAAA. BBBB.", "CCCC."]` |

Two whole sentences landed together; the third started a new chunk rather than being
split. That "+1" accounts for the space you add between sentences.

**Step by step:**

1. Start `current = ""`.
2. For each sentence `s`: if `current` is non-empty **and**
   `len(current) + len(s) + 1 > max_chars`, append `current` to `chunks` and reset
   `current = s`. Otherwise grow it: `current = (current + " " + s).strip()`.
3. After the loop, append any leftover non-empty `current`.
4. Return `chunks`.

Use `.strip()` so you don't accumulate leading/trailing whitespace. Then re-index and
re-measure — watch q08 flip from miss to hit.""")

code(r'''def sentence_chunks(text: str, max_chars: int) -> list[str]:
    """Group consecutive sentences into chunks of at most `max_chars` chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            chunks.append(current.strip())
            current = s
        else:
            current = (current + " " + s).strip()
    if current:
        chunks.append(current.strip())
    return chunks


sent_chunks = chunk_corpus(CORPUS, lambda t: sentence_chunks(t, SENTENCE_MAX_CHARS))
sentence_index = build_index(sent_chunks, "lab01-sentence")
sent_hr, sent_misses = hit_rate(sentence_index, EVAL, RETRIEVAL_TOP_K)

display_metrics_table({
    "fixed-chunk hit-rate": fixed_hr,
    "sentence-chunk hit-rate": sent_hr,
    "absolute gain": sent_hr - fixed_hr,
}, title=f"Retrieval@{RETRIEVAL_TOP_K}: fixed vs sentence chunking")
print("sentence-chunk misses:", sent_misses)
print("q08 now answered:", "q08" not in sent_misses)''',
r'''def sentence_chunks(text: str, max_chars: int) -> list[str]:
    """Group consecutive sentences into chunks of at most `max_chars` chars."""
    # TODO: Split `text` into sentences (regex below splits after . ! or ?).
    # Walk the sentences, accumulating them into `current`; when adding the next
    # one would exceed max_chars, flush `current` to `chunks` and start over.
    # Don't forget the final non-empty `current`.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    chunks = None  # replace this line
    assert isinstance(chunks, list), "Complete sentence_chunks before continuing"
    return chunks


sent_chunks = chunk_corpus(CORPUS, lambda t: sentence_chunks(t, SENTENCE_MAX_CHARS))
sentence_index = build_index(sent_chunks, "lab01-sentence")
sent_hr, sent_misses = hit_rate(sentence_index, EVAL, RETRIEVAL_TOP_K)

display_metrics_table({
    "fixed-chunk hit-rate": fixed_hr,
    "sentence-chunk hit-rate": sent_hr,
    "absolute gain": sent_hr - fixed_hr,
}, title=f"Retrieval@{RETRIEVAL_TOP_K}: fixed vs sentence chunking")
print("sentence-chunk misses:", sent_misses)
print("q08 now answered:", "q08" not in sent_misses)''')

md("""**Expected output:**
```
        Retrieval@3: fixed vs sentence chunking
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric                    ┃  Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ fixed-chunk hit-rate      │ 0.5000 │
│ sentence-chunk hit-rate   │ 0.7500 │
│ absolute gain             │ 0.2500 │
└───────────────────────────┴────────┘
sentence-chunk misses: ['q04', 'q06', 'q10']
q08 now answered: True
```
One chunking change recovered q01, q05, q07, **and q08** — a 25-point jump, no
model change. (Some questions, like q10, still miss — that's the reranking problem
Lab 02 takes on.)""")

# ── 10. Failure 3: cosine-but-wrong ──────────────────────────────────────────
md("""### ⚠️ Deliberate failure 3 — cosine rewards surface overlap

Sentence chunking fixed *recall*, but first-stage cosine still has no notion of
the question being *asked*. Try a query whose wording overlaps a confidently-wrong
document.

**Hypothesis first:** the query below says **warm**. Which document do you expect
cosine to rank #1, and is it the one that actually answers the question?""")

code(r'''trap = "What makes time-to-first-token slow on a warm NIM?"
print("Q:", trap, "\n")
results = retrieve(sentence_index, trap, 4)
for rank, r in enumerate(results, 1):
    title = next(d["title"] for d in CORPUS if d["id"] == r.metadata["doc_id"])
    print(f"  #{rank} {r.metadata['doc_id']} cos={r.score:.3f} | {title}")
print()
print("Top hit is d21 — 'Why the first NIM request is slow (COLD start)'. But the")
print("question says WARM. The correct answer is d10 ('Prefill versus decode':")
print("prefill dominates TTFT), which cosine ranks BELOW d21 because d21 shares the")
print("surface words 'first', 'slow', 'NIM'. Cosine measures closeness in embedding")
print("space, not whether a passage answers the question.")''')

md("""**Root cause + the Lab 02 hook.** A bi-encoder scores the query and each
passage *separately*, so it can't tell that "warm" rules out the cold-start doc. A
**cross-encoder reranker** reads the query and passage *together* and would
demote `d21`. That two-stage *retrieve-then-rerank* pattern is exactly what
**Lab 02** builds. See corpus docs `d36`–`d38`.""")

# ── 11. Assemble + generate ──────────────────────────────────────────────────
md("""## 6 · Assemble context and generate a grounded answer

Finally, assemble the top-K sentence chunks into a grounded prompt and call the LLM
NIM. Generation needs an `NVIDIA_API_KEY` (the mock proxies to the API Catalog), so
this cell skips cleanly if no key is configured.""")

code(r'''def build_prompt(query: str, results) -> list[dict]:
    context = "\n\n".join(
        f"[{r.metadata['doc_id']}] {r.text}" for r in results
    )
    system = (
        "You are a support assistant. Answer ONLY from the provided context. "
        "If the answer is not in the context, say so. Cite the [doc_id] you used."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]


question = "What value enables prefix KV-cache reuse in NIM?"
ctx = retrieve(sentence_index, question, RETRIEVAL_TOP_K)

if os.environ.get("NVIDIA_API_KEY"):
    llm = get_llm_client()
    resp, ms = timed_call(
        llm.chat.completions.create,
        model=LLM_MODEL,
        messages=build_prompt(question, ctx),
        temperature=0.0,
        max_tokens=160,
    )
    print(f"[llm] {ms:.0f}ms")
    print(resp.choices[0].message.content)
else:
    print("[skipped] No NVIDIA_API_KEY set — retrieval context that would be sent:")
    for r in ctx:
        print(f"  [{r.metadata['doc_id']}] {r.text[:70]}")''')

md("""**Expected output** (with a key set):
```
[llm] 540ms
Set NIM_ENABLE_KV_CACHE_REUSE to 1 to turn on prefix KV-cache reuse; it is off by
default because it trades GPU memory for latency. [d07]
```
The answer is grounded in `d07` and cites it — because sentence chunking kept the
variable name and its value in the same retrievable chunk.""")

# ── 12. Challenge ────────────────────────────────────────────────────────────
md("""## Challenge

Pick one and measure the effect on `hit_rate`:

1. **Overlap.** Add a 1-sentence overlap to `sentence_chunks` (repeat the last
   sentence at the start of the next chunk). Does it recover any of `q04`, `q06`,
   `q10`? What does it cost in index size?
2. **K sweep.** Plot hit-rate as `RETRIEVAL_TOP_K` goes 1→10 for both chunkers.
   Where does fixed chunking finally catch up, and what does a large K cost the
   generator (corpus doc `d19`)?
3. **Hybrid.** Combine cosine rank with a keyword (BM25-style) score and re-rank
   the top-20. How close does it get to the reranker preview in failure 3?""")

md("""## Key takeaways

- **Chunking is a retrieval decision.** A chunk is the smallest retrievable unit;
  if the answer spans a boundary, no result contains it. Sentence-aware chunking
  took hit-rate@3 from 50% → 75% with no model change.
- **NV-EmbedQA is asymmetric.** Always pass `input_type` (`query` vs `passage`).
  The failure is silent — it shows up as bad recall, not an error.
- **Configure cosine space explicitly.** A vector store left on its Euclidean
  default ranks differently for normalized embeddings.
- **Cosine rewards surface overlap.** First-stage retrieval can rank a
  surface-similar but wrong passage #1. Reading query and passage *together* — a
  cross-encoder reranker — is the fix. → **Lab 02.**

**References**
- NeMo Retriever (NV-EmbedQA): https://docs.nvidia.com/nemo/retriever/
- NVIDIA API Catalog (retrieval): https://build.nvidia.com/explore/retrieval
- RAGAS (faithfulness, measured rigorously in Lab 02): https://docs.ragas.io/""")

# ── build both notebooks ─────────────────────────────────────────────────────
def make(use_stub: bool) -> nbf.NotebookNode:
    nb = new_notebook()
    cells = []
    for kind, payload in CELLS:
        if kind == "md":
            cells.append(new_markdown_cell(payload))
        else:
            sol, stub = payload
            src = stub if (use_stub and stub) else sol
            cells.append(new_code_cell(src))
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
