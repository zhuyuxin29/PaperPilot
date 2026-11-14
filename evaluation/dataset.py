from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

QUESTION_TYPES = (
    "fact",
    "term",
    "method",
    "relation",
    "unanswerable",
)


@dataclass(frozen=True)
class Question:
    question_id: str
    paper_ids: tuple[str, ...]
    question: str
    question_type: str
    answerable: bool
    gold_chunk_ids: tuple[str, ...]
    gold_evidence_groups: tuple[tuple[str, ...], ...]
    reference_note: str = ""


class DatasetError(RuntimeError):
    pass


def load_dataset(path: Path | str) -> list[Question]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DatasetError(f"数据集不存在：{dataset_path}")
    questions: list[Question] = []
    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"第 {line_number} 行不是合法 JSON：{exc}") from exc
        try:
            questions.append(_build_question(payload))
        except (KeyError, ValueError, TypeError) as exc:
            raise DatasetError(f"第 {line_number} 行字段不合法：{exc}") from exc
    if not questions:
        raise DatasetError("数据集为空")
    return questions


def _build_question(payload: dict) -> Question:
    question_id = str(payload["question_id"]).strip()
    question = str(payload["question"]).strip()
    if not question_id or not question:
        raise ValueError("question_id / question 不能为空")

    paper_ids = payload.get("paper_ids") or []
    if not isinstance(paper_ids, list) or not paper_ids:
        raise ValueError("paper_ids 必须是非空列表")

    question_type = str(payload.get("question_type") or "fact").strip()
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"未知题型：{question_type}")

    answerable = bool(payload.get("answerable", True))

    gold = payload.get("gold_chunk_ids") or []
    if not isinstance(gold, list) or not all(isinstance(item, str) for item in gold):
        raise ValueError("gold_chunk_ids 必须是字符串列表")

    groups_raw = payload.get("gold_evidence_groups") or []
    groups: list[tuple[str, ...]] = []
    for group in groups_raw:
        if not isinstance(group, list) or not all(
            isinstance(item, str) for item in group
        ):
            raise ValueError("gold_evidence_groups 必须是字符串列表的列表")
        groups.append(tuple(group))

    return Question(
        question_id=question_id,
        paper_ids=tuple(str(item) for item in paper_ids),
        question=question,
        question_type=question_type,
        answerable=answerable,
        gold_chunk_ids=tuple(gold),
        gold_evidence_groups=tuple(groups),
        reference_note=str(payload.get("reference_note") or ""),
    )
