# Lab 01 — RAG Fundamentals

## What

A from-scratch RAG over the NVIDIA developer blog. You chunk, embed with
NV-EmbedQA-E5-v5, store in ChromaDB, retrieve by cosine similarity, and
assemble context for an LLM call. Then you break it on purpose:

- A chunk-boundary split that hides the answer
- A query that scores well in cosine but is semantically wrong
- An omitted `input_type` on the asymmetric embedder

For each break, you form a hypothesis before the fix.

## Why

A startup shipped a basic RAG over their support docs last Monday. CSAT
dropped 12% the first week. Support agents are getting adjacent chunks back
instead of answers. You're the SE on the 9am call — this lab reproduces their
pipeline so you understand what they actually built and where it cracks.

## How

```bash
task mock:start                                 # optional: no GPU needed
task lab:run LAB=01-rag-fundamentals            # opens lab.ipynb in Jupyter
task lab:test LAB=01-rag-fundamentals           # nbmake + pytest validation
```

**Stack:** NV-EmbedQA-E5-v5 via API Catalog (default) or mock. ChromaDB
local. Llama 3.1 8B Instruct for generation.

**You'll measure:** retrieval@K hit-rate (50% → 75% just by switching from
fixed-size to sentence-aware chunking) and per-call latency. Answer-level
faithfulness (RAGAS) is previewed here and measured rigorously in Lab 02, where
the cross-encoder reranker that fixes the cosine-but-wrong failure lives.
