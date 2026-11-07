"""证据登记（模块 8）：run 级证据集合。

约定：
- ``evidence_id`` 只在**本次运行**内有效，不能跨 run 复用；
- 同一片段在一次运行内重复回读，复用同一个 evidence_id；
- 证据只由 ``read_passages`` 产生，检索候选不是证据。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import Chunk


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    chunk_id: str
    paper_id: str
    revision: int
    page_number: int
    quote: str


class EvidenceRegistry:
    """一次运行内的证据集合；按登记顺序生成 ``ev_001`` 形式 ID。"""

    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}
        self._by_chunk: dict[str, str] = {}
        self._next = 1

    def register(
        self, chunk: Chunk, page_number: int | None = None
    ) -> EvidenceRecord:
        existing = self._by_chunk.get(chunk.chunk_id)
        if existing is not None:
            return self._records[existing]

        evidence_id = f"ev_{self._next:03d}"
        self._next += 1
        record = EvidenceRecord(
            evidence_id=evidence_id,
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            revision=chunk.revision,
            page_number=page_number if page_number is not None else chunk.page_number,
            quote=chunk.text,
        )
        self._records[evidence_id] = record
        self._by_chunk[chunk.chunk_id] = evidence_id
        return record

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return self._records.get(evidence_id)

    def all(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records.values())

    def clear(self) -> None:
        self._records.clear()
        self._by_chunk.clear()
        self._next = 1

    def __len__(self) -> int:
        return len(self._records)
