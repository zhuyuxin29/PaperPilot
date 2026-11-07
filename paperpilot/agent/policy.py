"""预算与停止条件（模块 9）。"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings


@dataclass(frozen=True)
class Budget:
    max_steps: int = 8
    max_tool_calls: int = 12
    max_seconds: float = 120.0

    @classmethod
    def from_settings(cls, settings: Settings) -> "Budget":
        return cls(
            max_steps=settings.max_steps,
            max_tool_calls=settings.max_tool_calls,
            max_seconds=float(settings.timeout_seconds),
        )

    def can_start_step(self, steps_done: int) -> bool:
        return steps_done < self.max_steps

    def remaining_tool_calls(self, used: int) -> int:
        return max(0, self.max_tool_calls - used)
