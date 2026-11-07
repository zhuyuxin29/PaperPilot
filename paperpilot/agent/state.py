"""Agent 状态：一次问答执行过程中需要保存的内容。"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..evidence.registry import EvidenceRegistry


@dataclass
class AgentState:
    question: str
    paper_ids: tuple[str, ...]
    evidence: EvidenceRegistry = field(default_factory=EvidenceRegistry)
    trace: list[dict] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    step_count: int = 0
    tool_call_count: int = 0
    status: str = "RUNNING"
    stop_reason: str | None = None
