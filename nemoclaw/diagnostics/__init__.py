"""Importable diagnostic logic powering the NeMoClaw MCP tools.

Each module ports (or re-exports) the *verified* solution logic from one lab, so the
MCP tools and the lab notebooks share a single source of truth. The test suite pins
these to the labs' canonical numbers.
"""

from nemoclaw.diagnostics import (  # noqa: F401
    fixtures,
    lab03_nim,
    lab04_gpu,
    lab05_agents,
    lab06_mlops,
)

# lab0102_rag pulls in sentence-transformers; import it lazily where needed rather
# than eagerly here so the rest of the diagnostics stay import-light.

__all__ = ["fixtures", "lab03_nim", "lab04_gpu", "lab05_agents", "lab06_mlops"]
