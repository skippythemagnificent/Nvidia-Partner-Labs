"""Typed config loader for Pulumi stacks."""
from __future__ import annotations

import pulumi


class LabConfig:
    """Reads stack config + secrets into typed attributes."""

    def __init__(self) -> None:
        cfg = pulumi.Config()
        k8s_cfg = pulumi.Config("kubernetes")

        # Secrets — set with: task infra:secrets:set
        self.nvidia_api_key = cfg.require_secret("nvidia_api_key")
        self.ngc_api_key = cfg.get_secret("ngc_api_key") or self.nvidia_api_key

        # Plain config
        self.nim_embed_model: str = cfg.get("nim_embed_model") or "nvidia/nv-embedqa-e5-v5"
        self.nim_rerank_model: str = (
            cfg.get("nim_rerank_model") or "nvidia/nv-rerankqa-mistral-4b-v3"
        )
        self.nim_llm_model: str = cfg.get("nim_llm_model") or "meta/llama-3.1-8b-instruct"
        self.gpu_count: int = cfg.get_int("gpu_count") or 1
        self.namespace: str = cfg.get("k8s_namespace") or "nvidia-labs"
        self.kubeconfig: str | None = k8s_cfg.get("kubeconfig")
