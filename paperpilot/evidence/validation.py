"""证据校验（模块 8）：引用必须属于本次运行，且在允许的论文范围内。

后续 Agent 发布答案前调用这里：模型只能引用本次 ``read_passages`` 登记的
``evidence_id``，不能引用检索候选、历史缓存或其他论文的证据。
"""

from __future__ import annotations

from typing import Iterable

from .registry import EvidenceRecord, EvidenceRegistry


class EvidenceError(RuntimeError):
    """证据缺失或越界；``code`` 供上层映射成用户提示。"""

    def __init__(self, message: str, code: str = "EVIDENCE_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def assert_evidence_in_scope(
    registry: EvidenceRegistry,
    evidence_ids: Iterable[str],
    allowed_paper_ids: Iterable[str] | None = None,
) -> tuple[EvidenceRecord, ...]:
    """校验并返回证据记录；不合法时抛 ``EvidenceError``。"""
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
