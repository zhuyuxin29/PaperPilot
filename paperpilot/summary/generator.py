from __future__ import annotations

import json
import hashlib
import os
import uuid
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .. import storage
from ..config import Settings
from ..evidence.passages import read_passages
from ..evidence.registry import EvidenceRegistry
from ..llm import ChatMessage, ChatProvider
from ..schemas import (
    SUMMARY_KEYS,
    SectionStatus,
    StatementKind,
    Summary,
    SummaryCoverage,
    SummaryItem,
    SummarySection,
)
from .validation import SummaryValidationError, validate_summary

SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "summary-v2"
DEFAULT_MAX_CHARS = 40_000
_MAX_REPAIR_ATTEMPTS = 2

SUMMARY_LABELS: dict[str, str] = {
    "background": "研究背景",
    "research_question": "研究问题",
    "methods": "研究方法",
    "experiments": "实验设计",
    "conclusions": "主要结论",
    "contributions": "创新与贡献",
    "limitations": "局限性",
}

SYSTEM_PROMPT = """你是学术文献阅读助手，任务是把一篇论文整理成固定七类要点。

只输出一个 JSON 对象，不要输出解释，不要用 Markdown 代码块。结构必须严格如下：
{"sections": {
  "background": {"status": "supported|not_found|insufficient_evidence", "items": [{"text": "...", "statement_kind": "reported_by_paper|assistant_analysis", "evidence_ids": ["ev_001"]}], "note": "可选说明"},
  "research_question": {...},
  "methods": {...},
  "experiments": {...},
  "conclusions": {...},
  "contributions": {...},
  "limitations": {...}
}}

规则：
1. 七个键必须全部出现，不能多也不能少。
2. status=supported 时必须至少一条 items，每条都要带 evidence_ids；evidence_ids 只能使用材料中出现的编号。
3. 论文确实没有写的内容：status=not_found，items 为空，note 说明检查范围。
4. 材料不足或无法确认：status=insufficient_evidence，items 为空，note 说明缺口。
5. 区分 reported_by_paper（作者明确陈述）和 assistant_analysis（基于证据的分析）；后者不能伪装成作者观点。
6. 不要编造页码、引用或论文没有的内容，不要为了填满字段而生成内容。
7. 条目用中文，简洁准确，每条 1~3 句。
8. 如果材料末尾附有“图谱关系线索”，它只用于帮助你组织要点，不能替代原文证据；
   引用与结论仍必须以上面的原文材料为准。"""


class SummaryError(RuntimeError):
    def __init__(self, message: str, code: str = "SUMMARY_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class GenerationResult:
    summary: Summary
    path: Path
    reused: bool


def current_summary_model(settings: Settings) -> str:
    provider = (settings.chat_provider or "").strip().lower()
    if provider in ("", "deepseek"):
        return settings.chat_model or "deepseek-chat"
    return settings.chat_model or provider


def summary_path(
    settings: Settings, paper_id: str, revision: int, model_id: str
) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
    return settings.summaries_dir / (
        f"{paper_id}.r{revision}.{SCHEMA_VERSION}.{PROMPT_VERSION}.{safe_model}.json"
    )


def load_summary(
    settings: Settings,
    path: Path,
    *,
    paper_id: str,
    revision: int,
    model_id: str,
    graph_digest: str | None = None,
) -> Summary:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = Summary.model_validate(data)
    if (
        summary.paper_id != paper_id
        or summary.revision != revision
        or summary.model_id != model_id
        or summary.prompt_version != PROMPT_VERSION
        or summary.schema_version != SCHEMA_VERSION
        or (graph_digest is not None and summary.graph_digest != graph_digest)
    ):
        raise SummaryError("七类要点缓存已过期（版本、模型或图谱不一致）", "SUMMARY_STALE")
    return summary


def generate_summary(
    conn: sqlite3.Connection,
    settings: Settings,
    chat: ChatProvider,
    paper_id: str,
    *,
    revision: int | None = None,
    force: bool = False,
    max_chars: int = DEFAULT_MAX_CHARS,
    graph: object | None = None,
) -> GenerationResult:
    paper = storage.get_paper(conn, paper_id)
    if paper is None:
        raise SummaryError(f"文献不存在：{paper_id}", "PAPER_NOT_FOUND")
    revision = revision or paper.revision
    model_id = current_summary_model(settings)
    path = summary_path(settings, paper_id, revision, model_id)
    graph_text, graph_digest = _graph_materials(graph, paper_id)

    if not force and path.is_file():
        try:
            return GenerationResult(
                load_summary(
                    settings,
                    path,
                    paper_id=paper_id,
                    revision=revision,
                    model_id=model_id,
                    graph_digest=graph_digest,
                ),
                path,
                True,
            )
        except (SummaryError, ValidationError, ValueError):
            pass

    pages = storage.get_pages(conn, paper_id, revision)
    chunks = storage.get_chunks(conn, paper_id, revision)
    if not chunks:
        raise SummaryError(f"文献没有可用的片段：{paper_id}", "NO_CHUNKS")

    selected = _select_chunks(chunks, max_chars)

    registry = EvidenceRegistry()
    passages = []
    for start in range(0, len(selected), 5):
        batch = [chunk.chunk_id for chunk in selected[start : start + 5]]
        passages.extend(
            read_passages(
                conn, registry, batch, allowed_paper_ids={paper_id}
            )
        )

    coverage = _build_coverage(
        pages=pages,
        chunks=chunks,
        selected=selected,
        processed=passages,
    )
    materials = "\n\n".join(
        f"[{passage.evidence.evidence_id}] 第 {passage.evidence.page_number} 页：\n"
        f"{passage.chunk.text}"
        for passage in passages
    )
    graph_section = (
        "\n\n图谱关系线索（仅辅助组织要点，引用仍必须使用上面的 evidence 编号）：\n"
        f"{graph_text}\n"
        if graph_text
        else ""
    )
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=(
                f"论文标题：{paper.title}\n\n"
                f"以下是回读到的原文材料（带证据编号）：\n\n{materials}\n"
                f"{graph_section}\n"
                "请按系统要求生成七类要点 JSON。"
            ),
        ),
    ]

    payload = _request_json(chat, messages)
    evidence_map = {
        record.evidence_id: record.chunk_id for record in registry.all()
    }
    summary = _build_summary(
        payload,
        paper_id=paper_id,
        revision=revision,
        model_id=model_id,
        coverage=coverage,
        graph_digest=graph_digest,
        evidence_map=evidence_map,
    )
    validate_summary(summary, registry, allowed_paper_ids={paper_id})

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return GenerationResult(summary, path, False)


GRAPH_QUERY_TYPES = (
    "method_experiments",
    "experiment_results",
    "conclusion_support",
)


def _graph_materials(graph: object | None, paper_id: str) -> tuple[str, str]:
    if graph is None:
        return "", ""
    try:
        from ..graph.client import WORKSPACE_ID

        lines: list[str] = []
        for query_type in GRAPH_QUERY_TYPES:
            rows = graph.query_template(
                query_type,
                workspace_id=WORKSPACE_ID,
                paper_ids=[paper_id],
                entity_name=None,
                limit=8,
            )
            for row in rows:
                lines.append(
                    f"- 第{row.get('page_number', '?')}页："
                    f"{row.get('subject', '?')} → {row.get('object', '?')}"
                    f"（{row.get('relation_type', '?')}）"
                )
        if not lines:
            return "", ""
        text = "\n".join(lines)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return text, digest
    except Exception:
        return "", ""


def _request_json(chat: ChatProvider, messages: list[ChatMessage]) -> dict:
    last_error = ""
    for _ in range(_MAX_REPAIR_ATTEMPTS):
        response = chat.chat(messages)
        content = response.message.content or ""
        try:
            return _parse_json(content)
        except ValueError as exc:
            last_error = str(exc)
            messages = [
                *messages,
                ChatMessage(role="assistant", content=content),
                ChatMessage(
                    role="user",
                    content=(
                        f"上一条输出不是合法 JSON（{exc}）。"
                        "请只输出一个 JSON 对象，不要解释，不要代码块。"
                    ),
                ),
            ]
    raise SummaryError(f"模型输出无法解析为 JSON：{last_error}", "BAD_MODEL_JSON")


def _parse_json(content: str) -> dict:
    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("没有找到 JSON 对象")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象")
    return data


def _select_chunks(chunks, max_chars: int):
    if not chunks or max_chars < 1:
        return []
    total = sum(len(chunk.text) for chunk in chunks)
    if total <= max_chars:
        return list(chunks)

    count = min(len(chunks), 32)
    if count <= 1:
        indices = [0]
    else:
        indices = sorted(
            {round(index * (len(chunks) - 1) / (count - 1)) for index in range(count)}
        )

    chosen: set[int] = set()
    used_chars = 0
    for index in ([0] if len(chunks) == 1 else [0, len(chunks) - 1]):
        size = len(chunks[index].text)
        if not chosen or used_chars + size <= max_chars:
            chosen.add(index)
            used_chars += size
    for index in indices:
        if index in chosen:
            continue
        size = len(chunks[index].text)
        if used_chars + size <= max_chars:
            chosen.add(index)
            used_chars += size
    return [chunks[index] for index in sorted(chosen)]


def _build_coverage(pages, chunks, selected, processed) -> SummaryCoverage:
    total_units = len(chunks)
    planned = len(selected)
    done = len(processed)
    unparsed = [page.page_number for page in pages if page.is_blank]
    complete = planned == total_units and done == planned and not unparsed
    return SummaryCoverage(
        planned_units=planned,
        processed_units=done,
        failed_units=(),
        unprocessed_units=(
            tuple(chunk.chunk_id for chunk in chunks if chunk.chunk_id not in {p.chunk.chunk_id for p in processed})
            if not complete
            else ()
        ),
        total_units=total_units,
        processed_ratio=round(done / total_units, 6),
        parsed_pages=len(pages) - len(unparsed),
        unparsed_pages=tuple(unparsed),
        status="complete" if complete else "partial",
        status_reason=(
            None
            if complete
            else f"已处理 {done}/{total_units} 个已解析片段；未解析页：{unparsed or '无'}"
        ),
    )


def _build_summary(
    payload: dict,
    *,
    paper_id: str,
    revision: int,
    model_id: str,
    coverage: SummaryCoverage,
    graph_digest: str = "",
    evidence_map: dict[str, str] | None = None,
) -> Summary:
    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, dict):
        raise SummaryError("模型输出缺少 sections", "BAD_SUMMARY_SHAPE")

    sections: dict[str, SummarySection] = {}
    for key in SUMMARY_KEYS:
        sections[key] = _build_section(key, raw_sections.get(key), coverage)

    return Summary(
        schema_version=SCHEMA_VERSION,
        paper_id=paper_id,
        revision=revision,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        graph_digest=graph_digest,
        evidence_map=evidence_map or {},
        coverage=coverage,
        sections=sections,
    )


def _build_section(
    key: str, raw: Any, coverage: SummaryCoverage
) -> SummarySection:
    if not isinstance(raw, dict):
        return SummarySection(
            status=SectionStatus.INSUFFICIENT_EVIDENCE,
            note=f"模型未返回“{SUMMARY_LABELS[key]}”类目",
        )

    try:
        status = SectionStatus(str(raw.get("status") or "").strip().lower())
    except ValueError:
        return SummarySection(
            status=SectionStatus.INSUFFICIENT_EVIDENCE,
            note="模型返回了未知状态，已标记为证据不足",
        )

    note = str(raw.get("note") or "").strip() or None
    items: list[SummaryItem] = []
    for raw_item in raw.get("items") or []:
        if not isinstance(raw_item, dict):
            continue
        try:
            kind = StatementKind(
                str(raw_item.get("statement_kind") or "reported_by_paper")
                .strip()
                .lower()
            )
        except ValueError:
            kind = StatementKind.REPORTED
        text = str(raw_item.get("text") or "").strip()
        evidence_ids = tuple(
            str(value).strip()
            for value in (raw_item.get("evidence_ids") or [])
            if str(value).strip()
        )
        if not text or not evidence_ids:
            continue
        try:
            items.append(
                SummaryItem(
                    text=text[:2000],
                    statement_kind=kind,
                    evidence_ids=evidence_ids,
                )
            )
        except ValidationError:
            continue

    if status is SectionStatus.SUPPORTED and not items:
        status = SectionStatus.INSUFFICIENT_EVIDENCE
        note = note or "模型声称有内容，但没有给出带引用的有效条目"

    if status is SectionStatus.NOT_FOUND:
        complete = (
            coverage.status == "complete"
            and abs(coverage.processed_ratio - 1.0) <= 1e-9
        )
        if not complete:
            status = SectionStatus.INSUFFICIENT_EVIDENCE
            note = note or "覆盖不完整，无法确认论文未报告该内容"

    if status is not SectionStatus.SUPPORTED:
        items = []
    if status is SectionStatus.INSUFFICIENT_EVIDENCE and not note:
        note = "证据不足或模型未提供该类目"

    try:
        return SummarySection(status=status, items=tuple(items), note=note)
    except ValidationError:
        return SummarySection(
            status=SectionStatus.INSUFFICIENT_EVIDENCE,
            note="模型输出不合法，已标记为证据不足",
        )
