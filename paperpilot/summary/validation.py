"""要点校验（模块 10）：引用必须来自本次生成登记的原文证据。

结构约束（七键、状态与条目规则、覆盖度一致性）由 ``schemas.Summary`` 的
Pydantic 校验器保证；这里补充“证据是否真的存在且属于当前论文”的检查。
"""

from __future__ import annotations

from typing import Iterable

from ..evidence.registry import EvidenceRegistry
from ..evidence.validation import EvidenceError, assert_evidence_in_scope
from ..schemas import SectionStatus, Summary


class SummaryValidationError(RuntimeError):
    """要点校验失败；``code`` 供上层映射成提示。"""

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
    """逐条检查要点引用是否有效；不合法时抛 ``SummaryValidationError``。"""
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
