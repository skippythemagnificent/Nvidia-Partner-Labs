# Lab 00 — Setup & Validation

**Status:** ready.

## What

A no-notebook lab. Runs four assertions that prove your environment can
execute the rest of the series:

1. The `shared/` package imports cleanly.
2. `.env` (or `.env.example`) is present.
3. Either `NVIDIA_API_KEY` is set or `USE_MOCK_NIM=true`.
4. `infra/` is initialized (skipped if `pulumi` isn't installed).

## Why

The labs assume a working uv project, a populated `.env`, and at least one
reachable NIM endpoint. Surfacing a missing prereq here saves you from
debugging a "why won't the embed client return anything" Heisenbug in Lab 01.

## How

```bash
# from repo root
task setup                        # installs deps, copies .env.example → .env
# edit .env: set NVIDIA_API_KEY=nvapi-... OR USE_MOCK_NIM=true
task lab:test LAB=00-setup
```

Expected: `4 passed`. If `test_nvidia_api_key_or_mock` fails, your `.env`
isn't populated.
