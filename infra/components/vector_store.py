"""Vector store component — scaffold (pgvector wiring lands in Lab 06)."""
from __future__ import annotations

import pulumi


class VectorStore(pulumi.ComponentResource):
    def __init__(self, name: str, cfg, opts=None) -> None:
        super().__init__("nvidia-labs:infra:VectorStore", name, {}, opts)
        # TODO Lab 06: provision pgvector on RDS (prod) or in-cluster (staging).
        self.connection_url = pulumi.Output.from_input(
            f"postgresql://placeholder/{cfg.namespace}"
        )
        self.register_outputs({"connection_url": self.connection_url})
