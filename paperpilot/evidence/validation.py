from __future__ import annotations

from typing import Iterable

from .registry import EvidenceRecord, EvidenceRegistry


class EvidenceError(RuntimeError):
    def __init__(self, message: str, code: str = "EVIDENCE_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def assert_evidence_in_scope(
    registry: EvidenceRegistry,
    evidence_ids: Iterable[str],
    allowed_paper_ids: Iterable[str] | None = None,
) -> tuple[EvidenceRecord, ...]:
    allowed = set(allowed_paper_ids) if allowed_paper_ids is not None else None
    records: list[EvidenceRecord] = []
    for evidence_id in evidence_ids:
        record = registry.get(evidence_id)
        if record is None:
            raise EvidenceError(
                f"引用了本次运行未登记的证据：{evidence_id}", "UNKNOWN_EVIDENCE"
            )
        if allowed is not None and record.paper_id not in allowed:
            raise EvidenceError(
                f"证据不属于当前论文范围：{evidence_id}", "OUT_OF_SCOPE"
            )
        records.append(record)
    return tuple(records)


def is_valid_citation(
    registry: EvidenceRegistry,
    evidence_id: str,
    allowed_paper_ids: Iterable[str] | None = None,
) -> bool:
    try:
        assert_evidence_in_scope(registry, [evidence_id], allowed_paper_ids)
    except EvidenceError:
        return False
    return True
