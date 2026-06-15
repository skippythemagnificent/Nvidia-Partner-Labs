"""NeMoClaw — an NVIDIA-solutions-engineer troubleshooting agent.

NeMoClaw exposes the *verified diagnostic logic* from the NVIDIA Partner Lab series
(Labs 00-06) as Model Context Protocol (MCP) tools, then drives them with an agent
built on the NVIDIA NeMo Agent Toolkit using an LLM NIM as its reasoning brain.

- ``nemoclaw.diagnostics`` — importable ports/re-exports of the labs' diagnostic
  functions (the single source of truth for the tool behaviour).
- ``nemoclaw.scenarios`` — the named real-world scenarios each lab dramatizes.
- ``nemoclaw.server`` — the FastMCP server (``python -m nemoclaw.server``).
- ``nemoclaw/agent`` — the NeMo Agent Toolkit workflow that consumes the server.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
