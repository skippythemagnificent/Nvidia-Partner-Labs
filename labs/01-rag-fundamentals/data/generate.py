# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Generate the Lab 01 corpus and evaluation set.

Run with:  uv run python labs/01-rag-fundamentals/data/generate.py

Writes two files next to this script:

  - corpus.json : 50 short NVIDIA developer-blog / NIM-deployment excerpts.
                  Each is realistic technical prose (no lorem, no "Acme Corp").
  - eval.json   : 12 labeled questions. Each names the doc that answers it and
                  the verbatim `answer_span` that must appear in the retrieved
                  context for the question to count as a hit. The labels drive
                  the retrieval@K hit-rate the notebook measures, so they are
                  committed alongside the corpus rather than re-derived at run
                  time.

The content is hand-authored rather than LLM-generated so the corpus is stable
across runs and the boundary-split / semantic-mismatch failure cases the lab
relies on stay reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
BLOG = "https://developer.nvidia.com/blog"

# Each doc: id, title, source_url, text.
# Docs d07 (KV-cache reuse) and d21 (cold start) are the anchors for the two
# core failure demos — see eval.json entries q05 and q08.
CORPUS: list[dict[str, str]] = [
    {
        "id": "d01",
        "title": "What NVIDIA NIM packages for you",
        "text": (
            "NVIDIA NIM is a set of containerized inference microservices that wrap an "
            "optimized model engine behind a stable, OpenAI-compatible HTTP API. Each "
            "container ships the model weights, a tuned TensorRT-LLM or Triton backend, "
            "and the runtime needed to serve it, so a team can pull one image from NGC "
            "and get a production endpoint without assembling the stack by hand."
        ),
    },
    {
        "id": "d02",
        "title": "NIM exposes an OpenAI-compatible API",
        "text": (
            "Because every NIM speaks the OpenAI REST dialect, existing clients work "
            "unchanged: point the OpenAI SDK at the NIM base URL and call "
            "/v1/chat/completions, /v1/completions, or /v1/embeddings. This means you "
            "can swap a hosted model for a self-hosted NIM by changing only the base_url "
            "and api_key, with no rewrite of application code."
        ),
    },
    {
        "id": "d03",
        "title": "Pulling NIM images from NGC",
        "text": (
            "NIM container images live in the NVIDIA NGC private registry at nvcr.io. To "
            "pull them you authenticate Docker with your NGC API key, using the literal "
            "username $oauthtoken and the key as the password. In Kubernetes the same "
            "credential is stored as an image pull secret of type "
            "kubernetes.io/dockerconfigjson so the kubelet can fetch the image."
        ),
    },
    {
        "id": "d04",
        "title": "Deploying a NIM with the official Helm chart",
        "text": (
            "NVIDIA publishes a Helm chart for NIM at the NGC Helm repository. The chart "
            "templates a Deployment, a Service, and a PersistentVolumeClaim for the model "
            "cache, and it wires in the NGC pull secret. You set the model image, the GPU "
            "count under resources.limits, and persistence.size for the cache volume, then "
            "run helm install against your cluster."
        ),
    },
    {
        "id": "d05",
        "title": "TensorRT-LLM profile selection at startup",
        "text": (
            "On startup a NIM inspects the GPUs it has been scheduled onto and selects a "
            "matching TensorRT-LLM engine profile — for example an H100 FP8 profile versus "
            "an A100 FP16 profile. The selected profile is printed in the container logs. If "
            "no prebuilt profile matches the detected hardware, the NIM falls back to "
            "building an engine locally, which adds minutes to the first startup."
        ),
    },
    {
        "id": "d06",
        "title": "Reading the profile NIM selected",
        "text": (
            "To confirm which optimization profile a running NIM chose, query its model "
            "metadata endpoint or grep the startup logs for the profile hash. Mismatched "
            "profiles are a common cause of lower-than-expected throughput: a NIM that "
            "silently fell back to a generic profile will serve correct answers but at a "
            "fraction of the tokens-per-second the hardware can sustain."
        ),
    },
    {
        "id": "d07",
        "title": "Prefix KV-cache reuse in NIM",
        "text": (
            "NIM can reuse the attention key/value cache for prompt prefixes that recur "
            "across requests, such as a shared system prompt, so those tokens are not "
            "recomputed every call. The feature is controlled by the environment variable "
            "NIM_ENABLE_KV_CACHE_REUSE. It is off by default because caching trades GPU "
            "memory for latency, and that trade-off should be measured per workload. To "
            "turn it on in production you set NIM_ENABLE_KV_CACHE_REUSE to 1."
        ),
    },
    {
        "id": "d08",
        "title": "KV cache exhaustion under load",
        "text": (
            "The KV cache is the dominant consumer of GPU memory during decode. When too "
            "many concurrent sequences are in flight the cache fills, and the scheduler "
            "either queues or preempts requests, which shows up as a sudden latency cliff "
            "rather than a gradual slowdown. Reducing max sequence length or lowering "
            "concurrency relieves the pressure."
        ),
    },
    {
        "id": "d09",
        "title": "Continuous batching keeps GPUs busy",
        "text": (
            "Rather than waiting for a fixed batch to fill, TensorRT-LLM uses continuous "
            "(in-flight) batching: it admits new requests into the running batch as soon as "
            "slots free up when other sequences finish. This keeps the GPU saturated under "
            "bursty traffic and is the main reason a single NIM can serve many concurrent "
            "users without a linear collapse in throughput."
        ),
    },
    {
        "id": "d10",
        "title": "Prefill versus decode",
        "text": (
            "An LLM request has two phases with very different hardware profiles. Prefill "
            "processes the whole prompt in parallel and is compute-bound, so it scales with "
            "GPU FLOPs. Decode generates one token at a time and is memory-bandwidth-bound, "
            "so it scales with HBM bandwidth. Time-to-first-token is dominated by prefill; "
            "inter-token latency is dominated by decode."
        ),
    },
    {
        "id": "d11",
        "title": "NV-EmbedQA is an asymmetric embedding model",
        "text": (
            "NV-EmbedQA-E5-v5 is trained for asymmetric retrieval: a question and the "
            "passage that answers it are encoded differently. Callers must declare the role "
            "of each input with input_type, set to query for the user's question and to "
            "passage for indexed documents. Omitting input_type collapses the two roles and "
            "produces vectors that do not compare meaningfully."
        ),
    },
    {
        "id": "d12",
        "title": "Why input_type matters for retrieval quality",
        "text": (
            "Encoding a query with the passage role (or vice versa) places it in the wrong "
            "subspace, so cosine similarity against your index becomes noise and recall "
            "drops sharply. The failure is silent: the API still returns a vector of the "
            "right shape, and the bug only surfaces as bad search results. Always pass "
            "input_type explicitly on every embedding call."
        ),
    },
    {
        "id": "d13",
        "title": "Embedding dimensionality and storage",
        "text": (
            "NV-EmbedQA-E5-v5 returns 1024-dimensional vectors. The dimensionality fixes "
            "the width of every row in your vector index, so it must match between indexing "
            "and query time. Switching embedding models almost always changes the dimension "
            "and therefore requires a full re-index of the corpus."
        ),
    },
    {
        "id": "d14",
        "title": "Cosine similarity for normalized embeddings",
        "text": (
            "Retrieval ranks passages by cosine similarity between the query vector and each "
            "passage vector. When embeddings are L2-normalized, cosine similarity reduces to "
            "the dot product, and cosine distance equals one minus that similarity. Configure "
            "your vector store to use the cosine space explicitly; a store left on its "
            "Euclidean default will rank differently."
        ),
    },
    {
        "id": "d15",
        "title": "Chunking is a retrieval decision, not a formatting one",
        "text": (
            "How you split documents determines what the retriever can possibly return. A "
            "chunk is the smallest unit that can be retrieved, so if the text that answers a "
            "question is spread across two chunks, no single result will contain the whole "
            "answer. Chunk size and boundaries should be chosen for retrieval, then the "
            "context window checked second."
        ),
    },
    {
        "id": "d16",
        "title": "Fixed-size chunking and its failure mode",
        "text": (
            "Fixed-size chunking splits text every N characters or tokens regardless of "
            "meaning. It is simple and predictable, but it routinely cuts a sentence — or the "
            "definition of a setting and its required value — across a boundary. When that "
            "happens the answer is fragmented and retrieval returns an adjacent chunk that "
            "looks relevant but is incomplete."
        ),
    },
    {
        "id": "d17",
        "title": "Sentence-aware and semantic chunking",
        "text": (
            "Sentence-aware chunking respects sentence boundaries and groups consecutive "
            "sentences up to a size budget, so a complete thought stays intact. Semantic "
            "chunking goes further and splits where the topic shifts. Both keep "
            "answer-bearing text together, which raises the chance that a single retrieved "
            "chunk fully answers the question."
        ),
    },
    {
        "id": "d18",
        "title": "Chunk overlap preserves context at the seams",
        "text": (
            "Adding a small overlap between adjacent chunks — repeating the last sentence or "
            "two at the start of the next chunk — softens boundary effects so a fact that "
            "lands near a seam appears in both neighbors. Overlap costs index size and some "
            "duplication in results, so keep it modest, on the order of 10 to 20 percent."
        ),
    },
    {
        "id": "d19",
        "title": "Top-k retrieval and the context budget",
        "text": (
            "Retrieval returns the top-k most similar chunks, which are concatenated into the "
            "prompt as context. Larger k raises recall but spends context tokens and can "
            "bury the answer among distractors. A common pattern is to over-retrieve, then "
            "rerank down to a smaller, higher-precision set before assembling the prompt."
        ),
    },
    {
        "id": "d20",
        "title": "Assembling grounded context for the LLM",
        "text": (
            "After retrieval, the chunks are formatted into the prompt with a clear "
            "instruction to answer only from the provided context and to say when the answer "
            "is not present. Grounding the model this way reduces hallucination, and "
            "including the source of each chunk lets the application cite where an answer "
            "came from."
        ),
    },
    {
        "id": "d21",
        "title": "Why the first NIM request is slow (cold start)",
        "text": (
            "The first request to a freshly started NIM is far slower than later ones even "
            "when the GPU is idle. On cold start the NIM must load model weights into GPU "
            "memory and finalize the TensorRT-LLM engine before it can serve a single token, "
            "so the latency you see on request one is dominated by model load, not by "
            "inference. Once warm, the same request returns in a fraction of the time."
        ),
    },
    {
        "id": "d22",
        "title": "The command that starts a NIM container",
        "text": (
            "To start a NIM locally you run the container with docker run, pass your NGC key "
            "in the environment, request the GPU with the --gpus all flag, and publish the "
            "service port. The container then begins its startup sequence and reports ready "
            "once the model is loaded. This is the quickest way to bring a single NIM up on "
            "a workstation for testing."
        ),
    },
    {
        "id": "d23",
        "title": "Health and readiness endpoints",
        "text": (
            "Every NIM exposes Kubernetes-style probes. /v2/health/live indicates the process "
            "is up, while /v2/health/ready returns 200 only after the model is loaded and the "
            "service can answer requests. Wire readiness — not liveness — into your load "
            "balancer so traffic is held back until the cold start completes."
        ),
    },
    {
        "id": "d24",
        "title": "Autoscaling NIM replicas",
        "text": (
            "Because GPUs are the scarce resource, NIM replicas are scaled on GPU-aware "
            "signals rather than CPU. Teams commonly drive a Horizontal Pod Autoscaler from "
            "queue depth or a custom metric like KV-cache utilization exported by the NIM, "
            "and they account for cold-start time so new replicas are ready before the spike "
            "they were meant to absorb."
        ),
    },
    {
        "id": "d25",
        "title": "Air-gapped NIM deployment",
        "text": (
            "For environments with no internet egress, NIM supports an offline cache: on a "
            "connected host you pre-download the model profile into a cache directory, "
            "transfer that directory across the air gap, and mount it into the container so "
            "the NIM finds the model locally and never reaches out to NGC at runtime."
        ),
    },
    {
        "id": "d26",
        "title": "Multi-Instance GPU partitions an H100",
        "text": (
            "Multi-Instance GPU (MIG) splits one A100 or H100 into as many as seven isolated "
            "instances, each with its own slice of compute and a dedicated, hardware-walled "
            "portion of HBM. MIG suits many small models or strict tenant isolation; it does "
            "not help a single large model that needs the whole device's memory and compute."
        ),
    },
    {
        "id": "d27",
        "title": "Tensor parallelism for large models",
        "text": (
            "When a model's weights and KV cache exceed one GPU's memory, tensor parallelism "
            "shards each layer across several GPUs that exchange activations over NVLink "
            "every step. It enables models that would not otherwise fit, but the per-step "
            "communication means throughput does not scale linearly with the GPU count."
        ),
    },
    {
        "id": "d28",
        "title": "NVLink versus PCIe for multi-GPU inference",
        "text": (
            "NVLink provides far higher GPU-to-GPU bandwidth than PCIe, which matters for the "
            "frequent all-reduce traffic of tensor-parallel inference. An SXM board with "
            "NVLink will sustain tensor-parallel decode that a PCIe board, limited to the "
            "slower bus, throttles. Match the interconnect to whether you intend to run "
            "multi-GPU models."
        ),
    },
    {
        "id": "d29",
        "title": "HBM bandwidth bounds decode throughput",
        "text": (
            "During decode the GPU streams the full model weights and KV cache from HBM for "
            "every token, so inter-token latency is set by memory bandwidth, not raw FLOPs. "
            "This is why the HBM3 bandwidth of an H100 SXM, rather than its compute, is often "
            "the figure that predicts tokens-per-second for a given model."
        ),
    },
    {
        "id": "d30",
        "title": "FP8 on Hopper",
        "text": (
            "Hopper GPUs add hardware FP8, and TensorRT-LLM can quantize weights and "
            "activations to FP8 to roughly halve memory traffic versus FP16. That widens the "
            "decode bottleneck and increases throughput, usually with negligible quality loss "
            "when the quantization is calibrated. FP8 profiles are why H100 NIMs often far "
            "outpace their A100 counterparts."
        ),
    },
    {
        "id": "d31",
        "title": "Speculative decoding",
        "text": (
            "Speculative decoding runs a small draft model to propose several tokens that the "
            "large target model then verifies in a single forward pass. When the draft is "
            "usually right, this amortizes the memory-bound decode step over multiple tokens "
            "and cuts latency, at the cost of running and maintaining a second model."
        ),
    },
    {
        "id": "d32",
        "title": "Triton serves the model under NIM",
        "text": (
            "Inside many NIMs the Triton Inference Server hosts the optimized engine and "
            "handles request scheduling, dynamic batching, and metrics. NIM presents the "
            "friendly OpenAI API on top, but the Triton layer underneath is what exposes the "
            "operational knobs and the Prometheus metrics endpoint that platform teams scrape."
        ),
    },
    {
        "id": "d33",
        "title": "Triton metrics for observability",
        "text": (
            "Triton publishes Prometheus metrics on port 8002 at /metrics, including request "
            "counts, queue time, compute time, and time-to-first-token. Scraping these gives "
            "platform teams the latency breakdown they need to tell a queuing problem (rising "
            "queue time) apart from a compute problem (rising compute time)."
        ),
    },
    {
        "id": "d34",
        "title": "DCGM Exporter for GPU-level metrics",
        "text": (
            "The DCGM Exporter surfaces device-level telemetry — GPU utilization, memory used, "
            "temperature, power, and SM clocks — as Prometheus metrics. Paired with Triton's "
            "serving metrics it lets you correlate a latency spike with, say, a thermal "
            "throttle or a memory-pressure event on the specific GPU serving the request."
        ),
    },
    {
        "id": "d35",
        "title": "NeMo Retriever embedding and reranking NIMs",
        "text": (
            "NeMo Retriever packages the retrieval models as NIMs: an embedding NIM such as "
            "NV-EmbedQA for the first-stage vector search and a reranking NIM such as "
            "NV-RerankQA-Mistral-4B for the second stage. Running both as NIMs gives a RAG "
            "pipeline the same OpenAI-style interface and on-prem deployment story as the "
            "generation model."
        ),
    },
    {
        "id": "d36",
        "title": "Bi-encoders retrieve, cross-encoders rerank",
        "text": (
            "An embedding model is a bi-encoder: it encodes the query and each passage "
            "separately, which is fast and lets you precompute the index, but it never lets "
            "the two texts interact. A cross-encoder reranker reads the query and a passage "
            "together and scores their relevance directly, which is far more accurate but too "
            "expensive to run over the whole corpus."
        ),
    },
    {
        "id": "d37",
        "title": "The two-stage retrieve-then-rerank pattern",
        "text": (
            "Production RAG over-retrieves cheaply with the bi-encoder — say the top 20 by "
            "cosine — then reranks those candidates with the cross-encoder and keeps the top "
            "few. This pairs the bi-encoder's speed with the cross-encoder's precision and "
            "fixes the common case where the truly relevant passage sits a few ranks below a "
            "surface-similar distractor."
        ),
    },
    {
        "id": "d38",
        "title": "Why cosine similarity rewards surface overlap",
        "text": (
            "First-stage embeddings often rank a passage highly because it shares vocabulary "
            "with the query even when it does not answer it. Cosine similarity has no notion "
            "of the question being asked; it measures closeness in embedding space, where "
            "lexical and topical overlap pull vectors together. A reranker that reads both "
            "texts jointly is what separates relevant from merely similar."
        ),
    },
    {
        "id": "d39",
        "title": "Faithfulness measures grounding",
        "text": (
            "RAGAS faithfulness checks whether each claim in the generated answer is supported "
            "by the retrieved context. A low faithfulness score means the model is asserting "
            "things the context does not back — hallucination — and it usually points at "
            "retrieval that failed to surface the answer rather than at the generator itself."
        ),
    },
    {
        "id": "d40",
        "title": "Context precision and recall",
        "text": (
            "Context precision asks how much of the retrieved context is actually relevant, "
            "and context recall asks whether everything needed to answer was retrieved. Low "
            "recall caps the best answer the generator can give no matter how good it is, so "
            "diagnosing RAG quality starts with the retrieval metrics, not the prompt."
        ),
    },
    {
        "id": "d41",
        "title": "Temperature zero for deterministic routing",
        "text": (
            "When an LLM call must be reproducible — a router choosing a tool, or a classifier "
            "labeling an intent — set temperature to 0 so the model takes the most likely "
            "token at each step. Sampling temperature above 0 introduces run-to-run variation "
            "that is desirable for open generation but a liability for control flow."
        ),
    },
    {
        "id": "d42",
        "title": "Structured output with JSON schema",
        "text": (
            "Constraining a NIM to emit JSON that conforms to a schema turns free text into a "
            "parseable contract, which is essential when the output feeds another system. "
            "Schema-guided decoding rejects tokens that would break the structure, so the "
            "response is valid JSON by construction rather than by hopeful prompting."
        ),
    },
    {
        "id": "d43",
        "title": "NeMo Guardrails as a NIM",
        "text": (
            "NeMo Guardrails can run as its own NIM in front of the LLM to enforce input and "
            "output policies — blocking off-topic requests, masking sensitive data, and "
            "checking responses against a set of rails — without baking that logic into every "
            "application that calls the model."
        ),
    },
    {
        "id": "d44",
        "title": "Capacity planning from an SLA",
        "text": (
            "Sizing GPU infrastructure starts from the service-level target: a p99 "
            "time-to-first-token and a sustained requests-per-second. From the model size and "
            "the GPU's HBM bandwidth you estimate per-request decode cost, divide the target "
            "throughput by what one GPU sustains within the latency budget, and add headroom "
            "for traffic bursts and cold starts."
        ),
    },
    {
        "id": "d45",
        "title": "Quantization trades memory for a little quality",
        "text": (
            "Quantizing weights from FP16 to INT8 or FP8 shrinks the memory footprint and the "
            "bandwidth each decode step consumes, which directly raises throughput. Calibrated "
            "post-training quantization usually costs only a small, measurable drop in "
            "accuracy, and that drop should be validated on your own evaluation set before "
            "shipping."
        ),
    },
    {
        "id": "d46",
        "title": "Persisting the model cache in Kubernetes",
        "text": (
            "Mount a PersistentVolumeClaim at the NIM's cache path so the downloaded model and "
            "built engine survive pod restarts. Without it, every restart re-downloads "
            "tens of gigabytes from NGC and rebuilds the engine, turning a quick reschedule "
            "into a multi-minute cold start and wasting egress bandwidth."
        ),
    },
    {
        "id": "d47",
        "title": "GPU Operator and the device plugin",
        "text": (
            "The NVIDIA GPU Operator installs the driver, container toolkit, and device plugin "
            "a cluster needs to schedule GPU workloads. The device plugin advertises "
            "nvidia.com/gpu as a schedulable resource, which is the resource a NIM pod "
            "requests under resources.limits so Kubernetes places it on a GPU node."
        ),
    },
    {
        "id": "d48",
        "title": "Right-sizing GPU memory for a model",
        "text": (
            "A model's GPU memory budget is the sum of its weights, the KV cache for the "
            "concurrency you intend to serve, and activation and framework overhead. The KV "
            "term grows with sequence length and batch size, so a model that fits at low "
            "concurrency can run out of memory once traffic and context lengths climb."
        ),
    },
    {
        "id": "d49",
        "title": "Batch size and the latency-throughput trade-off",
        "text": (
            "Larger batches amortize the memory-bound decode step over more sequences and "
            "raise tokens-per-second, but each individual request waits longer, so per-user "
            "latency rises. The right operating point is the largest batch that still meets "
            "your p99 latency target, which continuous batching approaches automatically."
        ),
    },
    {
        "id": "d50",
        "title": "RAG beats fine-tuning for changing knowledge",
        "text": (
            "When the knowledge a system needs changes often — policies, prices, "
            "documentation — retrieval-augmented generation is usually the better tool than "
            "fine-tuning, because you update an index instead of retraining a model. "
            "Fine-tuning earns its keep for changing behavior and format, not for injecting "
            "facts that go stale."
        ),
    },
]

# Labeled questions. `answer_doc_id` is the doc that answers the question;
# `answer_span` is a verbatim, lowercase-comparable substring that must appear in
# the concatenated top-k context for the retrieval to count as a hit.
EVAL: list[dict[str, str]] = [
    {
        "id": "q01",
        "question": "What username do I use to authenticate Docker against NGC to pull NIM images?",
        "answer_doc_id": "d03",
        "answer_span": "$oauthtoken",
    },
    {
        "id": "q02",
        "question": "Which input_type should I pass when embedding a user's question with NV-EmbedQA?",
        "answer_doc_id": "d11",
        "answer_span": "set to query for the user's question",
    },
    {
        "id": "q03",
        "question": "What dimensionality do NV-EmbedQA-E5-v5 vectors have?",
        "answer_doc_id": "d13",
        "answer_span": "1024-dimensional vectors",
    },
    {
        "id": "q04",
        "question": "How does continuous batching keep the GPU busy under bursty traffic?",
        "answer_doc_id": "d09",
        "answer_span": "admits new requests into the running batch",
    },
    {
        # Boundary-split anchor: the variable name and its required value sit in
        # different sentences of d07, so a small fixed-size chunk can separate them.
        "id": "q05",
        "question": "What value enables prefix KV-cache reuse in NIM?",
        "answer_doc_id": "d07",
        "answer_span": "set nim_enable_kv_cache_reuse to 1",
    },
    {
        "id": "q06",
        "question": "Which endpoint should a load balancer use to know a NIM can serve traffic?",
        "answer_doc_id": "d23",
        "answer_span": "/v2/health/ready returns 200 only after the model is loaded",
    },
    {
        "id": "q07",
        "question": "How do I deploy a NIM offline with no internet access?",
        "answer_doc_id": "d25",
        "answer_span": "pre-download the model profile into a cache directory",
    },
    {
        # Semantic-mismatch anchor: phrased to share surface words ("slow", "start",
        # "GPU") with the container-start doc d22, while the real answer is d21.
        "id": "q08",
        "question": "Why is my very first NIM request so slow when the GPU is sitting idle?",
        "answer_doc_id": "d21",
        "answer_span": "latency you see on request one is dominated by model load",
    },
    {
        "id": "q09",
        "question": "On which port does Triton expose Prometheus metrics?",
        "answer_doc_id": "d33",
        "answer_span": "prometheus metrics on port 8002",
    },
    {
        "id": "q10",
        "question": "What is the difference between a bi-encoder and a cross-encoder?",
        "answer_doc_id": "d36",
        "answer_span": "cross-encoder reranker reads the query and a passage together",
    },
    {
        "id": "q11",
        "question": "How many instances can MIG split an H100 into?",
        "answer_doc_id": "d26",
        "answer_span": "as many as seven isolated",
    },
    {
        "id": "q12",
        "question": "What does RAGAS faithfulness measure?",
        "answer_doc_id": "d39",
        "answer_span": "each claim in the generated answer is supported by the retrieved context",
    },
]


def _validate() -> None:
    """Fail loudly if the corpus/eval drift out of sync — these invariants are
    what the notebook and tests rely on."""
    ids = [d["id"] for d in CORPUS]
    assert len(ids) == len(set(ids)), "duplicate doc id in CORPUS"
    assert len(CORPUS) == 50, f"expected 50 docs, got {len(CORPUS)}"
    by_id = {d["id"]: d for d in CORPUS}
    for item in EVAL:
        doc = by_id.get(item["answer_doc_id"])
        assert doc is not None, (
            f"{item['id']} points at missing doc {item['answer_doc_id']}"
        )
        assert item["answer_span"].lower() in doc["text"].lower(), (
            f"{item['id']} answer_span is not a verbatim substring of {doc['id']}"
        )


def main() -> None:
    _validate()
    for d in CORPUS:
        d.setdefault("source_url", f"{BLOG}/{d['id']}")

    corpus_path = HERE / "corpus.json"
    eval_path = HERE / "eval.json"
    corpus_path.write_text(json.dumps(CORPUS, indent=2) + "\n")
    eval_path.write_text(json.dumps(EVAL, indent=2) + "\n")
    print(f"[ok] wrote {len(CORPUS)} docs  -> {corpus_path}")
    print(f"[ok] wrote {len(EVAL)} queries -> {eval_path}")


if __name__ == "__main__":
    main()
