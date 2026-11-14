from __future__ import annotations

import sqlite3
import time
from typing import Sequence
from dataclasses import replace

from paperpilot.agent.policy import Budget
from paperpilot.agent.runner import run_agent
from paperpilot.config import Settings
from paperpilot.evidence.registry import EvidenceRegistry
from paperpilot.graph.client import WORKSPACE_ID, GraphError
from paperpilot.llm import (
    ChatError,
    ChatProvider,
    EmbeddingProvider,
)
from paperpilot.rag import retrieval
from paperpilot.tools import build_registry

from .dataset import Question
from .metrics import (
    evidence_group_coverage,
    hit_at_k,
    looks_like_refusal,
    mean,
    recall_at_k,
)

GRAPH_QUERY_TYPES = (
    "method_experiments",
    "experiment_results",
    "conclusion_support",
)


def run_retrieval_eval(
    conn: sqlite3.Connection,
    settings: Settings,
    provider: EmbeddingProvider,
    questions: Sequence[Question],
    *,
    use_graph: bool = False,
    graph_client=None,
    top_k: int = 8,
) -> dict:
    rows: list[dict] = []
    for question in questions:
        entry = {
            "question_id": question.question_id,
            "question_type": question.question_type,
            "answerable": question.answerable,
            "gold_chunk_ids": list(question.gold_chunk_ids),
            "retrieved": [],
            "recall_at_k": None,
            "hit_at_k": None,
            "group_coverage": None,
            "graph_used": False,
            "error": None,
        }
        if not question.paper_ids:
            entry["error"] = "NO_PAPER"
            rows.append(entry)
            continue
        try:
            candidates = retrieval.search_chunks(
                conn,
                settings,
                question.question,
                list(question.paper_ids),
                provider=provider,
                top_k=top_k,
            )
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(entry)
            continue

        text_ids = [candidate.chunk_id for candidate in candidates]
        graph_ids: list[str] = []
        if use_graph and graph_client is not None:
            try:
                graph_ids = _graph_chunk_ids(
                    graph_client, list(question.paper_ids), question.question
                )
            except GraphError as exc:
                entry["error"] = f"GRAPH: {exc.message}"

        merged = (
            _rrf(text_ids, graph_ids, top_k=top_k)
            if graph_ids
            else text_ids[:top_k]
        )
        entry.update(
            {
                "retrieved": merged,
                "graph_used": bool(graph_ids),
                "recall_at_k": recall_at_k(
                    question.gold_chunk_ids, merged, top_k
                ),
                "hit_at_k": hit_at_k(question.gold_chunk_ids, merged, top_k),
                "group_coverage": evidence_group_coverage(
                    question.gold_evidence_groups, merged, top_k
                ),
            }
        )
        rows.append(entry)

    summary = {
        "questions": len(rows),
        "with_gold": sum(1 for row in rows if row["recall_at_k"] is not None),
        "recall_at_k": mean([row["recall_at_k"] for row in rows]),
        "hit_at_k": mean([row["hit_at_k"] for row in rows]),
        "group_coverage": mean([row["group_coverage"] for row in rows]),
        "graph_used": sum(1 for row in rows if row["graph_used"]),
        "errors": sum(1 for row in rows if row["error"]),
    }
    return {
        "config": "B" if use_graph else "A",
        "kind": "retrieval",
        "top_k": top_k,
        "summary": summary,
        "rows": rows,
    }


def run_agent_eval(
    conn: sqlite3.Connection,
    settings: Settings,
    embedding: EmbeddingProvider,
    chat: ChatProvider,
    questions: Sequence[Question],
    *,
    use_graph: bool = False,
    graph_client=None,
    budget: Budget | None = None,
) -> dict:
    if use_graph and graph_client is None:
        raise ValueError("配置 D 必须提供可用的图谱连接")
    settings = replace(settings, graph_mode="on" if use_graph else "off")
    from paperpilot.tools.registry import ToolRegistry
    base = build_registry(graph_enabled=use_graph)
    tools = ToolRegistry()

    for name in base.names():
        if name != "get_key_points":
            tools.register(base.get(name))
    rows: list[dict] = []
    for question in questions:
        started = time.perf_counter()
        entry = {
            "question_id": question.question_id,
            "question_type": question.question_type,
            "answerable": question.answerable,
            "status": None,
            "stop_reason": None,
            "steps": 0,
            "tool_calls": 0,
            "latency_ms": 0,
            "citations": [],
            "invalid_citations": [],
            "answer": "",
            "refused": False,
            "gold_covered": 0 if question.gold_chunk_ids else None,
            "cited_chunk_ids": [],
            "graph_used": False,
            "group_coverage": None,
            "repair_attempts": 0,
            "error": None,
        }
        try:
            result = None
            for attempt in range(2):
                try:
                    result = run_agent(
                        question=question.question,
                        paper_ids=question.paper_ids,
                        conn=conn,
                        settings=settings,
                        embedding=embedding,
                        chat=chat,
                        tools=tools,
                        budget=budget,
                        graph=graph_client if use_graph else None,
                    )
                    break
                except ChatError:
                    if attempt == 1:
                        raise
            assert result is not None
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        else:
            cited_chunks = [citation.chunk_id for citation in result.citations]
            entry.update(
                {
                    "status": result.status,
                    "stop_reason": result.stop_reason,
                    "steps": result.steps,
                    "tool_calls": result.tool_calls,
                    "citations": [
                        citation.evidence_id for citation in result.citations
                    ],
                    "invalid_citations": list(result.invalid_citations),
                    "answer": result.answer,
                    "refused": result.status == "COMPLETED" and result.answer_kind == "insufficient_evidence",
                    "cited_chunk_ids": cited_chunks,
                    "trace": list(result.trace),
                    "graph_used": any(t.get("tool") == "query_graph" and t.get("status") == "ok" for t in result.trace),
                    "group_coverage": evidence_group_coverage(question.gold_evidence_groups, cited_chunks, len(cited_chunks)),
                    "repair_attempts": result.repair_attempts,
                }
            )
            if question.gold_chunk_ids:
                entry["gold_covered"] = (
                    1
                    if set(question.gold_chunk_ids).intersection(cited_chunks)
                    else 0
                )
        entry["latency_ms"] = int((time.perf_counter() - started) * 1000)
        rows.append(entry)

    completed = [row for row in rows if row["status"] == "COMPLETED"]
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    summary = {
        "config": "D" if use_graph else "C",
        "questions": len(rows),
        "completed": len(completed),
        "completion_rate": (len(completed) / len(rows)) if rows else None,
        "errors": sum(1 for row in rows if row["error"]),
        "avg_steps": mean([row["steps"] for row in rows]),
        "avg_tool_calls": mean([row["tool_calls"] for row in rows]),
        "citation_question_rate": (
            sum(1 for row in answerable if row["status"] == "COMPLETED" and row["citations"]) / len(answerable)
            if answerable
            else None
        ),
        "refusal_rate_unanswerable": (
            sum(1 for row in unanswerable if row["refused"]) / len(unanswerable)
            if unanswerable
            else None
        ),
        "graph_used_questions": sum(1 for row in rows if row["graph_used"]),
        "evidence_group_coverage": mean([row["group_coverage"] for row in answerable]),
        "gold_covered_rate": mean([row["gold_covered"] for row in answerable]),
        "avg_repair_attempts": mean(
            [row["repair_attempts"] for row in rows]
        ),
        "repaired_questions": sum(
            1 for row in rows if row["repair_attempts"]
        ),
        "avg_latency_ms": mean([row["latency_ms"] for row in rows]),
    }
    return {
        "config": "D" if use_graph else "C",
        "kind": "agent",
        "summary": summary,
        "rows": rows,
    }


def _graph_chunk_ids(graph_client, paper_ids: list[str], question: str) -> list[str]:
    seen: list[str] = []
    for query_type in GRAPH_QUERY_TYPES:
        rows = graph_client.query_template(
            query_type,
            workspace_id=WORKSPACE_ID,
            paper_ids=paper_ids,
            entity_name=None,
            question=question,
            limit=20,
        )
        for row in rows:
            chunk_id = row.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.append(chunk_id)
    return seen


def _rrf(
    text_ids: list[str], graph_ids: list[str], *, top_k: int, k: int = 60
) -> list[str]:
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(text_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for rank, chunk_id in enumerate(graph_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [chunk_id for chunk_id, _ in ordered[:top_k]]
