from __future__ import annotations

from . import get_key_points, query_graph, read_passages, search_chunks
from .registry import (
    ToolContext,
    ToolError,
    ToolOutcome,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


def build_registry(*, graph_enabled: bool = False) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(search_chunks.SPEC)
    registry.register(read_passages.SPEC)
    registry.register(get_key_points.SPEC)
    if graph_enabled:
        registry.register(query_graph.SPEC)
    return registry


__all__ = [
    "ToolContext",
    "ToolError",
    "ToolOutcome",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "build_registry",
]
