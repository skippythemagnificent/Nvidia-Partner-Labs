# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Generate the Lab 03 deployment artifacts: NIM log samples + a profile manifest.

Run with:  uv run python labs/03-nim-deployment/data/generate.py
       or:  task lab:data:generate LAB=03-nim-deployment

Lab 03 is an *offline* troubleshooting lab — no GPU and no NIM calls. The learner
parses and reasons about real-shaped artifacts that a NIM emits on a cluster:

  logs/startup_healthy.log     A clean NIM 1.x startup on 1x H100: GPU detection,
                               TRT-LLM profile auto-selection, KV-cache allocation,
                               Triton + Uvicorn ready, first-token latency. The
                               "what good looks like" reference for time-to-ready.
  logs/ngc_auth_failure.log    `kubectl describe pod` for an ImagePullBackOff — the
                               NGC pull secret is missing/invalid (401 from nvcr.io).
  logs/kv_cache_exhaustion.log A healthy NIM that starts rejecting requests under
                               concurrent load: KV-cache blocks exhausted, HTTP 503.
  logs/airgapped_offline.log   An air-gapped A100 target failing to load because the
                               cache was prepped on an H100 dev box — wrong profile
                               artifacts on the offline media, NGC unreachable.

  profiles.json                The model's TRT-LLM profile manifest, as printed by
                               `nim list-model-profiles`: one entry per GPU SKU /
                               precision / tensor-parallel size, with the cache
                               artifacts each profile needs.
  cache_manifest.json          The artifacts actually present on the air-gapped
                               media (the H100/fp8 set someone copied by mistake).
  deployment.json              The Helm values + the GPU the air-gapped node detected.

Everything is hand-authored so the timestamps, token counts, and profile IDs stay
fixed and the parsing/diagnosis exercises remain reproducible. Content is illustrative;
log lines are representative of NIM/Triton output, not copied from a live system.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
LOGS = HERE / "logs"

MODEL = "meta/llama-3.1-8b-instruct"
NIM_VERSION = "1.3.0"

# ── Log samples ──────────────────────────────────────────────────────────────
# A clean startup on a single H100. Time-to-ready is the gap between the first
# line (09:14:02.001) and "Uvicorn running" (09:16:30.044) = 148.043 s. The
# KV-cache line reports the total token budget the capacity exercise reuses.
STARTUP_HEALTHY = f"""\
INFO 2025-11-03 09:14:02.001 launch.py:38] NIM for LLMs {NIM_VERSION} starting for model "{MODEL}"
INFO 2025-11-03 09:14:02.044 ngc_injector.py:152] Detected 1 GPU(s) on the host:
INFO 2025-11-03 09:14:02.045 ngc_injector.py:153]   GPU 0: NVIDIA H100 80GB HBM3 | compute capability 9.0 | 79.6 GiB free
INFO 2025-11-03 09:14:02.310 ngc_injector.py:201] Inspecting 6 model profile(s) in the manifest for compatible hardware
INFO 2025-11-03 09:14:02.522 ngc_injector.py:233] Selected profile: 8835c31752fd (tensorrt_llm-h100-fp8-tp1-throughput)
INFO 2025-11-03 09:14:02.523 ngc_injector.py:234]   backend=tensorrt_llm precision=fp8 tp=1 pp=1 gpu=h100 kind=throughput
INFO 2025-11-03 09:14:02.524 ngc_injector.py:240] Reason: most-optimized profile compatible with 1x H100 (sm90 supports fp8); override with NIM_MODEL_PROFILE
INFO 2025-11-03 09:14:03.900 cache.py:88] Profile artifacts found in /opt/nim/.cache; skipping NGC download
INFO 2025-11-03 09:14:31.220 trtllm.py:144] Loading prebuilt TRT-LLM engine for profile 8835c31752fd (no compilation required)
INFO 2025-11-03 09:15:58.402 trtllm.py:201] KV cache allocated: 17.9 GiB across 7340 blocks of 16 tokens (max 117440 tokens); gpu_memory_utilization=0.90
INFO 2025-11-03 09:15:58.404 trtllm.py:205] Runtime limits: max_num_seqs=48 max_model_len=4096
INFO 2025-11-03 09:16:10.110 server.py:412] Triton inference server ready (grpc :8001, http :8000, metrics :8002)
INFO 2025-11-03 09:16:30.044 api_server.py:88] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO 2025-11-03 09:16:31.500 api_server.py:120] First inference OK: TTFT=243 ms, generated 128 tokens at 71.4 tok/s
"""

# `kubectl describe pod` for a pod that never starts: the image cannot be pulled
# because the NGC pull secret is missing or holds a bad key (401 from nvcr.io).
NGC_AUTH_FAILURE = f"""\
$ kubectl -n nvidia-labs-staging get pods
NAME                          READY   STATUS             RESTARTS   AGE
nim-llm-7c9d8b6f4-2xktp       0/1     ImagePullBackOff   0          3m12s

$ kubectl -n nvidia-labs-staging describe pod nim-llm-7c9d8b6f4-2xktp
Name:             nim-llm-7c9d8b6f4-2xktp
Namespace:        nvidia-labs-staging
Containers:
  nim-llm:
    Image:        nvcr.io/nim/{MODEL}:{NIM_VERSION}
    State:        Waiting
      Reason:     ImagePullBackOff
Volumes:
  ngc-pull-secret:
    Type:         Secret (a volume populated by a Secret)
    SecretName:   ngc-secret
    Optional:     false
Events:
  Type     Reason     Age                  From     Message
  ----     ------     ----                 ----     -------
  Normal   Scheduled  3m20s                default-scheduler  Successfully assigned nvidia-labs-staging/nim-llm-7c9d8b6f4-2xktp to gpu-node-03
  Normal   Pulling    3m18s                kubelet  Pulling image "nvcr.io/nim/{MODEL}:{NIM_VERSION}"
  Warning  Failed     3m02s                kubelet  Failed to pull image "nvcr.io/nim/{MODEL}:{NIM_VERSION}": rpc error: code = Unknown desc = failed to pull and unpack image: failed to resolve reference "nvcr.io/nim/{MODEL}:{NIM_VERSION}": pulling from host nvcr.io failed with status code [manifests {NIM_VERSION}]: 401 Unauthorized
  Warning  Failed     3m02s                kubelet  Error: ErrImagePull
  Normal   BackOff    2m49s (x4 over 3m1s) kubelet  Back-off pulling image "nvcr.io/nim/{MODEL}:{NIM_VERSION}"
  Warning  Failed     2m49s (x4 over 3m1s) kubelet  Error: ImagePullBackOff
"""

# A NIM that started fine, then buckles under concurrency: every available KV-cache
# block is in use, the scheduler preempts and finally rejects requests with 503.
KV_CACHE_EXHAUSTION = """\
INFO 2025-11-03 11:08:55.012 server.py:412] Triton inference server ready; accepting traffic
INFO 2025-11-03 11:41:03.220 executor.py:298] Load ramp: 64 concurrent client streams, mean prompt 1200 tokens, max_tokens=512
WARNING 2025-11-03 11:41:07.880 executor.py:332] KV cache utilization 0.97 (7120/7340 blocks); scheduler queue depth 41
WARNING 2025-11-03 11:41:08.119 executor.py:355] Preempting 3 in-flight request(s) to reclaim KV cache blocks; affected sequences will be recomputed
ERROR 2025-11-03 11:41:08.450 scheduler.py:511] Request req-3f2a91 rejected: needs 4096 tokens but 0 free KV cache blocks (max_num_seqs=48 in flight, max_model_len=4096)
ERROR 2025-11-03 11:41:08.451 api_server.py:233] HTTP 503 for req-3f2a91: "no available kv cache blocks; server overloaded, retry later"
WARNING 2025-11-03 11:41:08.460 executor.py:332] KV cache utilization 1.00 (7340/7340 blocks); scheduler queue depth 47
ERROR 2025-11-03 11:41:09.002 scheduler.py:511] Request req-3f2b04 rejected: needs 4096 tokens but 0 free KV cache blocks (max_num_seqs=48 in flight, max_model_len=4096)
ERROR 2025-11-03 11:41:09.003 api_server.py:233] HTTP 503 for req-3f2b04: "no available kv cache blocks; server overloaded, retry later"
INFO 2025-11-03 11:41:14.700 executor.py:298] Load drained to 22 concurrent streams; KV cache utilization 0.61; no rejections
"""

# Air-gapped A100 node. NGC is unreachable, so the engine must come from the copied
# cache — but the cache holds the H100/fp8 profile, not the A100/fp16 one this GPU
# resolves to. The engine artifacts for the needed profile are absent.
AIRGAPPED_OFFLINE = f"""\
INFO 2025-11-03 15:02:10.500 launch.py:38] NIM for LLMs {NIM_VERSION} starting for model "{MODEL}"
INFO 2025-11-03 15:02:10.540 cache.py:51] NIM_CACHE_PATH=/opt/nim/.cache; NGC connectivity probe failed, entering offline mode (NIM_OFFLINE=1)
INFO 2025-11-03 15:02:10.700 ngc_injector.py:152] Detected 1 GPU(s) on the host:
INFO 2025-11-03 15:02:10.701 ngc_injector.py:153]   GPU 0: NVIDIA A100 80GB PCIe | compute capability 8.0 | 79.1 GiB free
INFO 2025-11-03 15:02:10.980 ngc_injector.py:233] Selected profile: 6f1ac2d40b77 (tensorrt_llm-a100-fp16-tp1-throughput)
INFO 2025-11-03 15:02:10.981 ngc_injector.py:240] Reason: A100 (sm80) does not support fp8; fp16 is the most-optimized compatible precision
ERROR 2025-11-03 15:02:11.220 cache.py:142] Required artifact models/6f1ac2d40b77/rank0.engine for profile 6f1ac2d40b77 not found in /opt/nim/.cache and NGC is unreachable (NIM_OFFLINE=1)
ERROR 2025-11-03 15:02:11.221 cache.py:148] Cached profiles on this host: [8835c31752fd (h100-fp8-tp1)]; requested profile: 6f1ac2d40b77 (a100-fp16-tp1)
CRITICAL 2025-11-03 15:02:11.300 launch.py:96] Cannot materialize engine offline. Run `nim download-to-cache --profile 6f1ac2d40b77` on a connected host and copy /opt/nim/.cache to this node, or set NIM_MODEL_PROFILE to a cached profile.
"""

LOG_FILES = {
    "startup_healthy.log": STARTUP_HEALTHY,
    "ngc_auth_failure.log": NGC_AUTH_FAILURE,
    "kv_cache_exhaustion.log": KV_CACHE_EXHAUSTION,
    "airgapped_offline.log": AIRGAPPED_OFFLINE,
}

# ── Profile manifest (`nim list-model-profiles`) ─────────────────────────────
# Six profiles for the 8B model across GPU SKUs / precisions / tensor-parallel
# sizes, plus a portable vLLM fallback. fp8 profiles require compute capability
# >= 8.9 (Hopper/Ada); A100 (sm80) cannot run them.
_PROFILE_SPECS = [
    ("8835c31752fd", "tensorrt_llm", "h100", "9.0", "fp8", 1, "throughput"),
    ("a47e9b13c205", "tensorrt_llm", "h100", "9.0", "fp16", 1, "latency"),
    ("6f1ac2d40b77", "tensorrt_llm", "a100", "8.0", "fp16", 1, "throughput"),
    ("b2d8e6740199", "tensorrt_llm", "a100", "8.0", "fp16", 2, "throughput"),
    ("c90af5e21b6d", "tensorrt_llm", "l40s", "8.9", "fp8", 1, "throughput"),
    ("0f4e7a9c8db3", "vllm", "any", None, "fp16", 1, "generic"),
]

# Artifacts shared by every profile of this model, plus per-profile engine files.
SHARED_ARTIFACTS = ["config.json", "tokenizer.json", "tokenizer_config.json"]


def _artifacts(profile_id: str, backend: str, tp: int) -> list[str]:
    items = list(SHARED_ARTIFACTS)
    if backend == "tensorrt_llm":
        items.append(f"models/{profile_id}/config.json")
        items += [f"models/{profile_id}/rank{r}.engine" for r in range(tp)]
    else:  # vllm loads HF weights directly
        items.append("model.safetensors")
    return items


def _profiles() -> list[dict]:
    out = []
    for pid, backend, gpu, cc, precision, tp, kind in _PROFILE_SPECS:
        out.append({
            "id": pid,
            "backend": backend,
            "gpu": gpu,
            "compute_capability": cc,
            "precision": precision,
            "tp": tp,
            "pp": 1,
            "kind": kind,
            "artifacts": _artifacts(pid, backend, tp),
        })
    return out


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    for name, text in LOG_FILES.items():
        (LOGS / name).write_text(text)

    profiles = _profiles()
    manifest = {"model": MODEL, "nim_version": NIM_VERSION, "profiles": profiles}
    (HERE / "profiles.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # The air-gapped media: whoever prepped it ran download-to-cache on an H100
    # dev box, so only the H100/fp8 profile (plus shared files) made it across.
    h100 = next(p for p in profiles if p["id"] == "8835c31752fd")
    cache = {
        "prepared_on": "NVIDIA H100 80GB HBM3 (dev workstation)",
        "present_artifacts": sorted(set(h100["artifacts"])),
    }
    (HERE / "cache_manifest.json").write_text(json.dumps(cache, indent=2) + "\n")

    deployment = {
        "model": MODEL,
        "nim_version": NIM_VERSION,
        "helm_chart": "nim-llm",
        "namespace": "nvidia-labs-staging",
        "image_pull_secret": "ngc-secret",
        "airgapped_target_gpu": {"gpu": "a100", "compute_capability": "8.0", "count": 1},
        "dev_box_gpu": {"gpu": "h100", "compute_capability": "9.0", "count": 1},
    }
    (HERE / "deployment.json").write_text(json.dumps(deployment, indent=2) + "\n")

    print(f"[ok] wrote {len(LOG_FILES)} log samples -> {LOGS}")
    print(f"[ok] wrote {len(profiles)} profiles     -> {HERE / 'profiles.json'}")
    print(f"[ok] wrote cache manifest + deployment  -> {HERE}")


if __name__ == "__main__":
    main()
