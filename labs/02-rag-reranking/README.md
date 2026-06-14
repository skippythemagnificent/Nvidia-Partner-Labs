# Lab 02 — RAG Reranking

## What

A two-stage retrieval pipeline. Stage 1 over-retrieves with the bi-encoder
(top-20 by cosine). Stage 2 reranks with NV-RerankQA-Mistral-4B-v3 down to
top-5. You compare ordered results, quantify rank deltas, and run RAGAS
faithfulness before vs after.

## Why

A fintech startup migrated their support corpus from a flat FAQ to PDF
policy documents. Answer quality collapsed — the bi-encoder retrieves the
right neighborhood but ranks adjacent chunks above the actual answer. A
cross-encoder reranker, scoring each query/passage pair jointly, recovers
the right order. This lab makes that intuition visible with real rank-delta
output, not hand-waving.

## How

```bash
task lab:data:generate LAB=02-rag-reranking     # writes data/corpus.json + eval.json
task mock:start                                 # optional: no GPU needed
task lab:run LAB=02-rag-reranking
task lab:test LAB=02-rag-reranking
```

**Stack:** NV-EmbedQA-E5-v5 + NV-RerankQA-Mistral-4B-v3 (the mock uses a compact
cross-encoder, so the quality gain is real but smaller than production).

**You'll measure:** max rank delta across queries, MRR before vs after reranking
(0.88 → 0.96 on the mock), top-1 rerank score, and the stage-1 + stage-2 latency
budget. RAGAS-style faithfulness is an optional, key-gated comparison.
