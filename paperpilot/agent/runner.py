from __future__ import annotations

import json
import copy
from pydantic import ValidationError
from .output import FinalAnswer, OUTPUT_INSTRUCTION
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field

from ..config import Settings
from ..evidence.registry import EvidenceRecord
from ..evidence.validation import EvidenceError, assert_evidence_in_scope
from ..llm import ChatError, ChatMessage, ChatProvider, EmbeddingProvider
from ..tools.registry import ToolContext, ToolRegistry
from .policy import Budget
from .state import AgentState
from . import runs
from ..services import document_lock

_CITATION_RE = re.compile(r"\bev_\d{3,}\b")

SYSTEM_PROMPT = """你是 PaperPilot 文献阅读助手，只能基于用户选中的论文回答问题。

规则：
1. 你可以使用 query_graph、search_chunks、read_passages 等工具，由你根据问题自行决定调用顺序：
   - 关系型问题（某方法用于哪些实验、实验产生什么结果、哪些依据支持某结论）通常更适合先用 query_graph；
   - 事实/术语/定义/总结类问题通常用 search_chunks 就够。
2. query_graph 只返回关系声明和 chunk_id，不是原文；要引用必须再用 read_passages 回读。
3. search_chunks 返回的是候选片段，不能直接引用。
4. 要引用原文时先调用 read_passages 回读；只有它返回的 evidence_id 才能写进回答。
   检索到候选后，至少对最相关的 1~2 个候选执行 read_passages，再判断是否“未找到”；
   read_passages 默认带相邻片段，优先查看数字、比例、结论附近的上下文。
5. 如果 search_chunks 返回 recommended_chunk_ids，必须先 read_passages 读取第一个；
   读完仍找不到证据，才允许回答“当前文献中未找到”。
6. 如果已读片段包含与问题相关的数字/事实，必须基于它作答或说明缺口，不得直接说未找到。
7. 引用格式为 [ev_001]；没有证据支持时明确说明"当前文献中未找到"，不要编造。
8. 不要编造页码、原文或作者观点，不要超出所选论文范围。
9. 预算：最多 {max_steps} 轮决策、{max_tool_calls} 次工具调用；证据足够就尽快回答。
10. 用中文回答，先给结论，再给依据。
11. 不要把预算、轮次或工具调用次数写进回答；用户只关心结论和依据。"""


@dataclass(frozen=True)
class AgentResult:
    answer: str
    citations: tuple[EvidenceRecord, ...]
    status: str
    stop_reason: str | None
    trace: tuple[dict, ...]
    steps: int
    tool_calls: int
    usage: dict[str, int] = field(default_factory=dict)
    invalid_citations: tuple[str, ...] = ()
    repair_attempts: int = 0
    answer_kind: str | None = None
    run_id: str | None = None


def _run_agent(
    *,
    question: str,
    paper_ids: tuple[str, ...] | list[str],
    conn: sqlite3.Connection,
    settings: Settings,
    embedding: EmbeddingProvider,
    chat: ChatProvider,
    tools: ToolRegistry,
    budget: Budget | None = None,
    system_prompt: str | None = None,
    graph: object | None = None,
    cancel_event: threading.Event | None = None,
    run_id: str | None = None,
) -> AgentResult:
    budget = budget or Budget.from_settings(settings)
    state = AgentState(question=question, paper_ids=tuple(paper_ids))
    context = ToolContext(
        conn=conn,
        settings=settings,
        provider=embedding,
        evidence=state.evidence,
        allowed_paper_ids=state.paper_ids,
        graph=graph,
    )
    system = (system_prompt or SYSTEM_PROMPT).format(
        max_steps=budget.max_steps, max_tool_calls=budget.max_tool_calls
    )
    system += OUTPUT_INSTRUCTION
    system += "\n当前已启用工具：" + ", ".join(tools.names()) + "。不得调用未启用的工具。"
    system += "\n当前允许的论文 ID：" + ", ".join(state.paper_ids)
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=question),
    ]

    started = time.monotonic()
    deadline = started + budget.max_seconds

    def stopped():
        if cancel_event is not None and cancel_event.is_set():
            return "CANCELLED"
        if time.monotonic() >= deadline:
            return "TIMEOUT"
        return None

    last_signature: str | None = None
    repeat_count = 0

    while True:
        if cancel_event is not None and cancel_event.is_set():
            return _finish(state, "STOPPED", "CANCELLED")
        if not budget.can_start_step(state.step_count):
            return _finish(state, "STOPPED", "BUDGET_STEPS")
        if time.monotonic() - started > budget.max_seconds:
            return _finish(state, "STOPPED", "TIMEOUT")

        state.step_count += 1
        try:
            response = chat.chat(
                messages, tools=tools.openai_tools(),
                timeout=max(0.001, deadline - time.monotonic()),
            )
        except ChatError:
            if reason := stopped():
                return _finish(state, "STOPPED", reason)
            raise
        _add_usage(state, response.usage)
        if reason := stopped():
            return _finish(state, "STOPPED", reason)

        assistant = response.message
        messages.append(assistant)

        if assistant.tool_calls:
            for call in assistant.tool_calls:
                if state.tool_call_count >= budget.max_tool_calls:
                    return _finish(state, "STOPPED", "BUDGET_TOOL_CALLS")

                if reason := stopped():
                    return _finish(state, "STOPPED", reason)
                before_ids = {record.evidence_id for record in state.evidence.all()}
                context.provider = copy.copy(embedding)
                if hasattr(context.provider, "timeout"):
                    context.provider.timeout = min(
                        embedding.timeout, max(0.001, deadline - time.monotonic())
                    )
                result = tools.execute(call, context)
                state.tool_call_count += 1
                state.trace.append(
                    {
                        "step": state.step_count,
                        "tool": call.name,
                        "arguments": call.arguments,
                        "status": result.status,
                        "duration_ms": result.duration_ms,
                        "evidence_ids": list(result.evidence_ids),
                        "error": result.error,
                        "graph_claims": (
                            result.data.get("claims")
                            if call.name == "query_graph"
                            and isinstance(result.data, dict)
                            else None
                        ),
                    }
                )
                if run_id:
                    runs.progress(conn, run_id, state.trace)
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=call.id,
                        name=call.name,
                        content=json.dumps(
                            result.as_tool_message(), ensure_ascii=False
                        ),
                    )
                )

                signature = json.dumps(
                    {"name": call.name, "arguments": call.arguments},
                    sort_keys=True,
                    ensure_ascii=False,
                )
                if reason := stopped():
                    return _finish(state, "STOPPED", reason)
                new_ids = {record.evidence_id for record in state.evidence.all()} - before_ids
                if signature == last_signature and not new_ids:
                    repeat_count += 1
                else:
                    repeat_count = 0
                last_signature = signature
                if repeat_count >= 1:
                    return _finish(state, "STOPPED", "REPEATED_CALL")
            continue

        text = (assistant.content or "").strip()
        if not text:
            return _finish(state, "FAILED", "EMPTY_ANSWER")

        records, invalid = _validate_text(state, text)
        repair_attempts = 0
        if invalid:
            if reason := stopped():
                return _finish(state, "STOPPED", reason)
            try:
                repaired = _repair_citations(
                    chat=chat, messages=messages, answer=text, invalid=invalid,
                    state=state, budget=budget,
                    timeout=max(0.001, deadline - time.monotonic()),
                )
            except ChatError:
                if reason := stopped():
                    return _finish(state, "STOPPED", reason)
                raise
            repair_attempts = 1 if repaired is not None else 0
            if repaired:
                text = repaired
                messages.append(ChatMessage(role="assistant", content=text))
                records, invalid = _validate_text(state, text)

        if reason := stopped():
            return _finish(state, "STOPPED", reason)
        if invalid:
            return AgentResult(
                answer="",
                citations=(),
                status="FAILED",
                stop_reason="INVALID_ANSWER" if "INVALID_FORMAT" in invalid else "INVALID_CITATION",
                trace=tuple(state.trace),
                steps=state.step_count,
                tool_calls=state.tool_call_count,
                usage=dict(state.usage),
                invalid_citations=invalid,
                repair_attempts=repair_attempts,
            )

        return AgentResult(
            answer=_render_answer(text),
            answer_kind=FinalAnswer.model_validate_json(text).status,
            citations=tuple(records),
            status="COMPLETED",
            stop_reason=None,
            trace=tuple(state.trace),
            steps=state.step_count,
            tool_calls=state.tool_call_count,
            usage=dict(state.usage),
            repair_attempts=repair_attempts,
        )


def run_agent(**kwargs) -> AgentResult:
    from dataclasses import asdict, replace
    from .. import storage
    conn = kwargs["conn"]
    settings = kwargs["settings"]


    with document_lock(settings):
        if settings.graph_mode == "on":
            from ..services import graph_ready
            from ..graph.client import GraphError
            if kwargs.get("graph") is None or not graph_ready(conn, kwargs["paper_ids"]):
                raise GraphError("当前论文修订的图谱未就绪，请先构建图谱", "GRAPH_NOT_READY")
        revisions = {pid: paper.revision for pid in kwargs["paper_ids"]
                     if (paper := storage.get_paper(conn, pid)) is not None}
        run_id = runs.start(conn, kwargs["question"], kwargs["paper_ids"])

    try:
        result = _run_agent(**kwargs, run_id=run_id)
    except Exception as exc:
        with document_lock(settings):
            runs.finish(conn, run_id, "FAILED", {"error_type": type(exc).__name__, "code": getattr(exc, "code", "RUN_ERROR")})
        raise

    with document_lock(settings):
        if any((paper := storage.get_paper(conn, pid)) is None or paper.revision != revision
               for pid, revision in revisions.items()):
            result = replace(result, status="FAILED", stop_reason="REVISION_CHANGED", answer="", citations=())
        result = replace(result, run_id=run_id)
        runs.finish(conn, run_id, result.status, {**asdict(result), "source_revisions": revisions})
    return result


def _validate_text(
    state: AgentState, text: str
) -> tuple[tuple[EvidenceRecord, ...], tuple[str, ...]]:
    try:
        final = FinalAnswer.model_validate_json(text)
    except (ValidationError, ValueError):
        return (), ("INVALID_FORMAT",)
    cited = tuple(dict.fromkeys(
        identifier for claim in final.claims for identifier in claim.evidence_ids
    ))

    if any(_cited_ids(claim.text) for claim in final.claims):
        return (), ("INVALID_FORMAT",)
    try:
        records = assert_evidence_in_scope(
            state.evidence, cited, allowed_paper_ids=set(state.paper_ids)
        )
        return tuple(records), ()
    except EvidenceError:
        return (), tuple(cited)


def _repair_citations(
    *,
    chat: ChatProvider,
    messages: list[ChatMessage],
    answer: str,
    invalid: tuple[str, ...],
    state: AgentState,
    budget: Budget,
    timeout: float,
) -> str | None:
    if state.step_count >= budget.max_steps:
        return None

    evidence_lines = "\n".join(
        f"[{record.evidence_id}] 第{record.page_number}页：{record.quote[:180]}"
        for record in state.evidence.all()[:12]
    )
    instruction = (
        f"上一条最终输出未通过结构或引用检查：{', '.join(invalid)}。\n"
        f"本次运行可用的证据只有：\n{evidence_lines or '（无）'}\n\n"
        "请重写回答：只保留能被上述证据支持的结论，并只引用这些 evidence_id；"
        "没有证据支持的内容必须明确写“当前文献中未找到”。"
        "不要新增未提供的信息，不要编造新的 evidence_id。"
    )
    repair_messages = [
        *messages,
        ChatMessage(role="user", content=instruction),
    ]

    started = time.perf_counter()
    response = chat.chat(repair_messages, timeout=timeout)
    _add_usage(state, response.usage)
    state.step_count += 1
    repaired = (response.message.content or "").strip()

    _, still_invalid = _validate_text(state, repaired) if repaired else ((), ("EMPTY",))
    state.trace.append(
        {
            "step": state.step_count,
            "tool": "citation_repair",
            "arguments": {"invalid_citations": list(invalid)},
            "status": "ok" if not still_invalid else "failed",
            "duration_ms": max(0, int((time.perf_counter() - started) * 1000)),
            "evidence_ids": [],
            "error": (
                None
                if not still_invalid
                else {
                    "code": "INVALID_CITATION",
                    "message": f"重写后仍引用无效证据：{', '.join(still_invalid)}",
                    "retryable": False,
                }
            ),
        }
    )
    return repaired or None


def _render_answer(raw: str) -> str:
    final = FinalAnswer.model_validate_json(raw)
    if final.status == "insufficient_evidence":
        return "当前文献中未找到足够证据，无法可靠回答这个问题。"
    if final.status == "needs_clarification":
        return "请明确要询问的方法、实验或结论，以便在当前文献中查证。"
    return "\n\n".join(
        claim.text + " " + "".join(f"[{identifier}]" for identifier in claim.evidence_ids)
        for claim in final.claims
    )


def _cited_ids(text: str) -> tuple[str, ...]:
    seen: list[str] = []
    for evidence_id in _CITATION_RE.findall(text):
        if evidence_id not in seen:
            seen.append(evidence_id)
    return tuple(seen)


def _add_usage(state: AgentState, usage: dict[str, int] | None) -> None:
    if not usage:
        return
    for key, value in usage.items():
        if isinstance(value, int):
            state.usage[key] = state.usage.get(key, 0) + value


def _finish(state: AgentState, status: str, reason: str) -> AgentResult:
    return AgentResult(
        answer="",
        citations=(),
        status=status,
        stop_reason=reason,
        trace=tuple(state.trace),
        steps=state.step_count,
        tool_calls=state.tool_call_count,
        usage=dict(state.usage),
    )
