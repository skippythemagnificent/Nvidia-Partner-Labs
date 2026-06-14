"""Pulumi stack outputs → root .env.

Run with: `uv run python scripts/export_env.py --stack staging --out ../.env`
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

OUTPUT_MAP = {
    "nim_embed_url": "NIM_EMBED_URL",
    "nim_rerank_url": "NIM_RERANK_URL",
    "nim_llm_url": "NIM_LLM_URL",
    "vector_db_url": "VECTOR_DB_URL",
    "prometheus_url": "PROMETHEUS_URL",
    "grafana_url": "GRAFANA_URL",
    "use_mock_nim": "USE_MOCK_NIM",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = subprocess.run(
        ["uv", "run", "pulumi", "stack", "output", "--json", "--stack", args.stack],
        capture_output=True,
        text=True,
        check=True,
    )
    outputs = json.loads(result.stdout)

    env_path = Path(args.out)
    existing = env_path.read_text() if env_path.exists() else ""

    for pulumi_key, env_key in OUTPUT_MAP.items():
        if pulumi_key not in outputs:
            continue
        value = outputs[pulumi_key]
        pattern = rf"^{env_key}=.*$"
        line = f"{env_key}={value}"
        if re.search(pattern, existing, re.MULTILINE):
            existing = re.sub(pattern, line, existing, flags=re.MULTILINE)
        else:
            existing += f"\n{line}"

    env_path.write_text(existing.strip() + "\n")
    print(f"[ok] Wrote outputs to {args.out} from stack '{args.stack}'")


if __name__ == "__main__":
    main()
