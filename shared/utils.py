"""Timing + rich display helpers used by every lab notebook."""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from rich.console import Console
from rich.table import Table

T = TypeVar("T")

_console = Console()


def timed_call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    """Run `fn(*args, **kwargs)` and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def display_ranked_chunks(
    chunks: Sequence[Any],
    query: str,
    top_n: int = 5,
    text_width: int = 80,
) -> None:
    """Render a ranked-chunk table with rank deltas after reranking.

    Each chunk is expected to expose: `text`, `embed_rank`, `rerank_rank`,
    `embed_score`, `rerank_score`, `rank_delta`.
    """
    table = Table(title=f"Top {top_n} chunks for: {query!r}", show_lines=False)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Δ", justify="right")
    table.add_column("Embed", justify="right")
    table.add_column("Rerank", justify="right")
    table.add_column("Text")

    for i, c in enumerate(chunks[:top_n], start=1):
        delta = getattr(c, "rank_delta", None)
        delta_str = "—" if delta is None else (f"+{delta}" if delta > 0 else str(delta))
        embed = f"{getattr(c, 'embed_score', 0.0):.3f}"
        rerank_raw = getattr(c, "rerank_score", None)
        rerank = "—" if rerank_raw is None else f"{rerank_raw:.3f}"
        text = (c.text[:text_width] + "…") if len(c.text) > text_width else c.text
        table.add_row(str(i), delta_str, embed, rerank, text)

    _console.print(table)


def display_metrics_table(metrics: dict[str, float], title: str = "Metrics") -> None:
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    _console.print(table)
