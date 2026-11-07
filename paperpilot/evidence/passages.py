"""原文回读与证据登记（模块 8）。

``read_passages`` 是**唯一**能产生 ``evidence_id`` 的入口：

1. 按 chunk_id 取回真实片段（校验存在、修订、论文范围）；
2. 登记到本次运行的证据集合；
3. 可选返回同页相邻片段作为阅读上下文（相邻片段不自动登记为证据）。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable

from .. import storage
from ..schemas import Chunk
from .registry import EvidenceRecord, EvidenceRegistry
from .validation import EvidenceError

DEFAULT_MAX_BATCH = 5
MAX_NEIGHBOR = 2


@dataclass(frozen=True)
class Passage:
    evidence: EvidenceRecord
    chunk: Chunk
    neighbors: tuple[Chunk, ...] = ()


def read_passages(
    conn: sqlite3.Connection,
    registry: EvidenceRegistry,
    chunk_ids: Iterable[str],
    *,
    allowed_paper_ids: Iterable[str] | None = None,
    neighbor: int = 0,
    max_batch: int = DEFAULT_MAX_BATCH,
) -> list[Passage]:
    """回读片段并登记证据；一次最多 ``max_batch`` 个片段。"""
    unique_ids: list[str] = []
    seen: set[str] = set()
    for chunk_id in chunk_ids:
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            unique_ids.append(chunk_id)

    if not unique_ids:
        raise EvidenceError("没有要读取的片段", "EMPTY_BATCH")
    if len(unique_ids) > max_batch:
        raise EvidenceError(
            f"一次最多读取 {max_batch} 个片段，当前 {len(unique_ids)}",
            "BATCH_TOO_LARGE",
        )
    if neighbor < 0 or neighbor > MAX_NEIGHBOR:
        raise EvidenceError(
            f"neighbor 必须在 0~{MAX_NEIGHBOR} 之间", "BAD_NEIGHBOR"
        )

    allowed = set(allowed_paper_ids) if allowed_paper_ids is not None else None
    passages: list[Passage] = []
    for chunk_id in unique_ids:
        chunk = storage.get_chunk(conn, chunk_id)
        if chunk is None:
            raise EvidenceError(f"片段不存在：{chunk_id}", "CHUNK_NOT_FOUND")

        paper = storage.get_paper(conn, chunk.paper_id)
        if paper is None:
            raise EvidenceError(f"文献不存在：{chunk.paper_id}", "PAPER_NOT_FOUND")
        if chunk.revision != paper.revision:
            raise EvidenceError(
                f"片段属于旧修订：{chunk_id}（r{chunk.revision} ≠ r{paper.revision}）",
                "REVISION_MISMATCH",
            )
        if allowed is not None and chunk.paper_id not in allowed:
            raise EvidenceError(
                f"片段不在当前论文范围：{chunk_id}", "OUT_OF_SCOPE"
            )

        neighbors = (
            _neighbors(conn, chunk, neighbor) if neighbor > 0 else ()
        )
        passages.append(
            Passage(
                evidence=registry.register(chunk),
                chunk=chunk,
                neighbors=neighbors,
            )
        )
    return passages


def _neighbors(
    conn: sqlite3.Connection, chunk: Chunk, span: int
) -> tuple[Chunk, ...]:
    """同页相邻片段；只作为阅读上下文返回，不登记为证据。"""
    same_page = storage.get_chunks(
        conn, chunk.paper_id, chunk.revision, page_index=chunk.page_index
    )
    index = next(
        (
            position
            for position, item in enumerate(same_page)
            if item.chunk_id == chunk.chunk_id
        ),
        None,
    )
    if index is None:
        return ()
    start = max(0, index - span)
    return tuple(same_page[start:index] + same_page[index + 1 : index + 1 + span])
