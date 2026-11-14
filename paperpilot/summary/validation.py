from __future__ import annotations

from typing import Iterable

from ..evidence.registry import EvidenceRegistry
from ..evidence.validation import EvidenceError, assert_evidence_in_scope
from ..schemas import SectionStatus, Summary


class SummaryValidationError(RuntimeError):
    def __init__(self, message: str, code: str = "SUMMARY_INVALID") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def validate_summary(
    summary: Summary,
    registry: EvidenceRegistry,
    *,
    allowed_paper_ids: Iterable[str],
) -> None:
    allowed = set(allowed_paper_ids)
    for key, section in summary.sections.items():
        if section.status is SectionStatus.SUPPORTED and not section.items:
            raise SummaryValidationError(
                f"{key} 标记为 supported 但没有条目", "EMPTY_SUPPORTED"
            )
        for item in section.items:
            try:
                assert_evidence_in_scope(registry, item.evidence_ids, allowed)
            except EvidenceError as exc:
                raise SummaryValidationError(
                    f"{key} 的引用无效：{exc}", exc.code
                ) from exc
