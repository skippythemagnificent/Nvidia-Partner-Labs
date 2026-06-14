"""Pulumi entry point — dispatches by active stack."""
from __future__ import annotations

import pulumi

from config import LabConfig

stack = pulumi.get_stack()

if stack == "dev":
    # Dev: mock-NIM endpoints only — no cloud resources provisioned.
    pulumi.export("nim_embed_url", "http://localhost:8099/v1")
    pulumi.export("nim_rerank_url", "http://localhost:8099/v1")
    pulumi.export("nim_llm_url", "http://localhost:8099/v1")
    pulumi.export("use_mock_nim", "true")
elif stack in ("staging", "prod"):
    cfg = LabConfig()
    from components.nim_cluster import NimCluster
    from components.monitoring import MonitoringStack
    from components.vector_store import VectorStore

    vector_store = VectorStore("vector-store", cfg)
    nim = NimCluster("nim-cluster", cfg, vector_store)
    monitoring = MonitoringStack("monitoring", cfg, nim)

    pulumi.export("nim_embed_url", nim.embed_url)
    pulumi.export("nim_rerank_url", nim.rerank_url)
    pulumi.export("nim_llm_url", nim.llm_url)
    pulumi.export("vector_db_url", vector_store.connection_url)
    pulumi.export("prometheus_url", monitoring.prometheus_url)
    pulumi.export("grafana_url", monitoring.grafana_url)
    pulumi.export("use_mock_nim", "false")
else:
    raise ValueError(f"Unknown stack: {stack!r}. Expected dev | staging | prod.")
