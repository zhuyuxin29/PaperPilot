"""图谱构建（模块 11）：抽取 → 校验 → 幂等写入 Neo4j → 记录状态。

边界：
- 只写“关系声明 + 片段指针”，不写全文；实体 ID 由程序生成，模型不决定主键；
- 证据对不上的声明写为 ``pending_review``，查询只返回 ``validated``；
- 图谱失败不影响普通检索与问答（GRAPH_MODE=off 时完全不连 Neo4j）。
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .. import storage
from ..config import Settings
from ..llm import ChatProvider
from . import extractor
from .client import (
    EXTRACTOR_VERSION,
    WORKSPACE_ID,
    GraphError,
    Neo4jClient,
)


@dataclass(frozen=True)
class GraphBuildResult:
    status: str
    chunks_processed: int
    claims_total: int
    claims_validated: int
    claims_pending: int
    errors: tuple[str, ...]


def build_paper_graph(
    conn: sqlite3.Connection,
    settings: Settings,
    chat: ChatProvider,
    client: Neo4jClient,
    paper_id: str,
    *,
    revision: int | None = None,
    max_chunks: int | None = None,
    batch_size: int = 5,
) -> GraphBuildResult:
    paper = storage.get_paper(conn, paper_id)
    if paper is None:
        raise GraphError(f"文献不存在：{paper_id}", "PAPER_NOT_FOUND")
    revision = revision or paper.revision
    chunks = storage.get_chunks(conn, paper_id, revision)
    if not chunks:
        raise GraphError(f"文献没有片段：{paper_id}", "NO_CHUNKS")
    if max_chunks is not None:
        chunks = chunks[: max(1, max_chunks)]

    client.ensure_schema()
    claims = extractor.extract_claims(
        chat,
        chunks,
        batch_size=batch_size,
        cache_dir=settings.data_dir / "graph_cache",
    )
    # 抽取成功后再清空旧图，避免抽取失败把已有图谱删掉
    if hasattr(client, "clear_paper_graph"):
        client.clear_paper_graph(WORKSPACE_ID, paper_id)

    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    records: list[dict] = []
    for claim in claims:
        chunk = chunk_map.get(claim.chunk_id)
        if chunk is None:
            continue
        subject_id = entity_id(WORKSPACE_ID, claim.subject_type, claim.subject_name)
        object_id = entity_id(WORKSPACE_ID, claim.object_type, claim.object_name)
        identifier = claim_id(
            paper_id,
            revision,
            subject_id,
            claim.relation_type,
            object_id,
            chunk.chunk_id,
            claim.evidence_text,
        )
        status = "validated" if claim.evidence_aligned else "pending_review"
        records.append({
            "workspace_id": WORKSPACE_ID,
            "paper_id": paper_id,
            "revision": revision,
            "title": paper.title,
            "chunk_id": chunk.chunk_id,
            "page_number": chunk.page_number,
            "text_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:16],
            "claim_id": identifier,
            "relation_type": claim.relation_type,
            "subject_id": subject_id,
            "subject_type": claim.subject_type,
            "subject_name": claim.subject_name,
            "object_id": object_id,
            "object_type": claim.object_type,
            "object_name": claim.object_name,
            "evidence_text": claim.evidence_text,
            "status": status,
            "qualifiers_json": _json_dumps(claim.qualifiers),
            "extractor_version": EXTRACTOR_VERSION,
            "normalization_version": EXTRACTOR_VERSION,
        })

    validated = 0
    pending = 0
    errors: list[str] = []
    entities: set[str] = set()

    # 批量写入：真实客户端用 UNWIND，单事务多 claim；测试/自定义客户端逐条回退。
    batch_size = 40
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        try:
            if hasattr(client, "write_claims"):
                client.write_claims(batch)
            else:
                for record in batch:
                    client.write_claim(record)
        except GraphError as exc:
            errors.append(f"图谱写入失败（{len(batch)} 条）：{exc.message}")
            continue
        for record in batch:
            entities.update({record["subject_id"], record["object_id"]})
            if record["status"] == "validated":
                validated += 1
            else:
                pending += 1

    total = validated + pending
    if errors and total == 0:
        state = "failed"
    elif errors:
        state = "partial"
    else:
        state = "ready"
    storage.save_graph_state(
        conn,
        paper_id,
        revision,
        state,
        entity_count=len(entities),
        claim_count=total,
        error="; ".join(errors[:3]) or None,
    )
    return GraphBuildResult(
        status=state,
        chunks_processed=len(chunks),
        claims_total=total,
        claims_validated=validated,
        claims_pending=pending,
        errors=tuple(errors),
    )


def entity_id(workspace_id: str, entity_type: str, name: str) -> str:
    digest = hashlib.sha256(
        f"{workspace_id}|{entity_type}|{_normalize(name)}".encode("utf-8")
    ).hexdigest()[:16]
    return f"ent_{digest}"


def claim_id(
    paper_id: str,
    revision: int,
    subject_id: str,
    relation_type: str,
    object_id: str,
    chunk_id: str,
    evidence_text: str,
) -> str:
    digest = hashlib.sha256(
        f"{paper_id}|{revision}|{subject_id}|{relation_type}|{object_id}|"
        f"{chunk_id}|{_normalize(evidence_text)}".encode("utf-8")
    ).hexdigest()[:16]
    return f"clm_{digest}"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _json_dumps(value: dict) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def main() -> None:
    """python -m paperpilot.graph.builder [paper_id] [max_chunks]"""
    import sys

    from ..config import load_settings
    from ..llm import create_chat_provider

    settings = load_settings()
    settings.ensure_dirs()
    conn = storage.connect(settings.db_path)
    storage.ensure_schema(conn)
    client = Neo4jClient.connect(settings)
    try:
        paper_ids = [sys.argv[1]] if len(sys.argv) > 1 else [
            paper.id for paper in storage.list_papers(conn)
        ]
        max_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else None
        chat = create_chat_provider(settings)
        for paper_id in paper_ids:
            result = build_paper_graph(
                conn, settings, chat, client, paper_id, max_chunks=max_chunks
            )
            print(paper_id, result)
    finally:
        client.close()
        conn.close()


if __name__ == "__main__":
    main()
