from __future__ import annotations

from typing import Sequence

REFUSAL_MARKERS = (
    "未找到",
    "没有找到",
    "证据不足",
    "无法回答",
    "未涉及",
    "未提及",
    "不包含",
    "未提供",
    "没有提供",
    "未报告",
    "没有报告",
    "未给出",
    "没有给出",
    "未开展",
    "没有开展",
    "insufficient",
    "not found",
    "no evidence",
    "cannot answer",
    "does not discuss",
)


def recall_at_k(
    gold_chunk_ids: Sequence[str], retrieved: Sequence[str], k: int
) -> float | None:
    gold = {item for item in gold_chunk_ids if item}
    if not gold:
        return None
    hits = len(gold.intersection(retrieved[:k]))
    return hits / len(gold)


def hit_at_k(
    gold_chunk_ids: Sequence[str], retrieved: Sequence[str], k: int
) -> int | None:
    gold = {item for item in gold_chunk_ids if item}
    if not gold:
        return None
    return 1 if gold.intersection(retrieved[:k]) else 0


def evidence_group_coverage(
    gold_groups: Sequence[Sequence[str]],
    retrieved: Sequence[str],
    k: int,
) -> float | None:
    groups = [set(group) for group in gold_groups if group]
    if not groups:
        return None
    hit_groups = sum(1 for group in groups if group.intersection(retrieved[:k]))
    return hit_groups / len(groups)


def looks_like_refusal(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def mean(values: Sequence[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None
