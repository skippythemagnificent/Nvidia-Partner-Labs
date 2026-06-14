# Lab 03 — NIM Deployment & Troubleshooting

## What

An offline troubleshooting lab — no GPU, no live NIM. You read real-shaped
NIM, Triton, and Kubernetes log samples (`data/logs/`) and write small
functions that turn them into diagnoses:

- `parse_startup` — read a clean NIM startup: GPU detection, TRT-LLM profile
  auto-selection, KV-cache allocation, and **time-to-ready** + first-token
  latency.
- `select_profile` — reproduce the `ngc_injector`'s profile pick (GPU family,
  `tp ≤ GPU count`, fp8 needs compute capability ≥ 8.9, most-optimized wins).
- `diagnose` — match a log against a failure playbook and return a structured
  `Diagnosis`. You point it at the three classic first-deployment failures:
  1. **NGC auth** — `ImagePullBackOff` / 401 from `nvcr.io` (bad pull secret).
  2. **KV-cache exhaustion** — HTTP 503 under load once the token budget fills.
  3. **Air-gapped cache miss** — the offline cache was prepped on the wrong GPU.

## Why

An ISV is shipping its first on-prem NIM to an air-gapped federal customer:
no internet, no runtime NGC access, hard go-live date. Standing up the Helm
chart is the easy part. What sinks the deal is everything around it — a pull
secret that 401s, a server that looks healthy until real traffic exhausts its
KV cache, and an offline cache that was burned against the dev box's H100
instead of the customer's A100. This lab is the on-call runbook for those.

## How

```bash
task lab:data:generate LAB=03-nim-deployment    # writes data/logs/*.log + profiles.json
task lab:run LAB=03-nim-deployment              # works offline; logs are sample data
task lab:test LAB=03-nim-deployment
# Optional: real K8s deploy if you have a cluster
task infra:up STACK=staging
task nim:profile                                 # which TRT-LLM profile got picked?
```

**Stack:** nim-llm Helm chart, NGC pull secret, Triton, the TRT-LLM profile
system, `nim list-model-profiles` / `download-to-cache`.

**You'll measure:** time-to-ready (148 s on the sample H100 startup), first-token
latency, the profile each GPU SKU resolves to, the KV-cache concurrency ceiling
(`kv_cache_tokens // max_model_len` = 28 here, vs 64 offered → 503s), and exactly
which cache artifacts the air-gapped node is missing.
