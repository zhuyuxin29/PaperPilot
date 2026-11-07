"""工具注册表（模块 9）。

规则：
- 只有注册过的工具才能被模型调用（白名单），未知工具返回 ``UNKNOWN_TOOL``；
- 参数用 Pydantic 校验，失败返回 ``BAD_ARGUMENTS``；
- 每个结果都带 ``tool_call_id``，成功/失败使用同一结构，便于回填给模型；
- 工具只能通过 ``ToolContext`` 访问数据库、embedding 与证据集合，论文范围由程序注入。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..evidence.registry import EvidenceRegistry
from ..llm import EmbeddingProvider, ToolCall


class ToolError(RuntimeError):
    """工具执行失败；``code`` 会原样返回给上层。"""

    def __init__(
        self, message: str, code: str = "TOOL_ERROR", retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable


@dataclass
class ToolContext:
    conn: sqlite3.Connection
    settings: Settings
    provider: EmbeddingProvider
    evidence: EvidenceRegistry
    allowed_paper_ids: tuple[str, ...]
    graph: object | None = None


@dataclass(frozen=True)
class ToolOutcome:
    data: dict[str, Any]
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    name: str
    status: str
    data: Any
    evidence_ids: tuple[str, ...]
    error: dict[str, Any] | None
    duration_ms: int

    def as_tool_message(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "status": self.status,
            "data": self.data,
            "evidence_ids": list(self.evidence_ids),
            "error": self.error,
        }
        return payload


Handler = Callable[[BaseModel, ToolContext], ToolOutcome]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Handler

    def openai_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class ToolRegistry:
    """白名单工具注册表。"""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"工具重复注册：{spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def openai_tools(self) -> list[dict[str, Any]]:
        return [spec.openai_schema() for spec in self._specs.values()]

    def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        spec = self._specs.get(call.name)
        if spec is None:
            return self._error(
                call, "UNKNOWN_TOOL", f"未注册的工具：{call.name}", started
            )
        try:
            args = spec.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            return self._error(
                call, "BAD_ARGUMENTS", f"工具参数不合法：{exc}", started, retryable=True
            )

        try:
            outcome = spec.handler(args, ctx)
        except ToolError as exc:
            return self._error(
                call, exc.code, exc.message, started, retryable=exc.retryable
            )
        except Exception as exc:  # 检索/证据/embedding 错误统一映射
            code = getattr(exc, "code", "TOOL_ERROR")
            retryable = code in {"EMBEDDING_NETWORK", "INDEX_STALE", "RETRIEVAL_ERROR"}
            return self._error(call, code, str(exc), started, retryable=retryable)

        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            status="ok",
            data=outcome.data,
            evidence_ids=outcome.evidence_ids,
            error=None,
            duration_ms=_elapsed_ms(started),
        )

    def _error(
        self,
        call: ToolCall,
        code: str,
        message: str,
        started: float,
        *,
        retryable: bool = False,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            status="error",
            data=None,
            evidence_ids=(),
            error={"code": code, "message": message[:500], "retryable": retryable},
            duration_ms=_elapsed_ms(started),
        )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))
