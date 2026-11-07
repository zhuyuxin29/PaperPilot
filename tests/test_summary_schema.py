"""模块 10（summary schema）验收测试：七键、状态规则、覆盖度约束。"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from paperpilot.schemas import (
    SUMMARY_KEYS,
    SectionStatus,
    StatementKind,
    Summary,
    SummaryCoverage,
    SummaryItem,
    SummarySection,
)


def coverage(**overrides) -> SummaryCoverage:
    data = dict(
        planned_units=3,
        processed_units=3,
        failed_units=(),
        unprocessed_units=(),
        total_units=3,
        processed_ratio=1.0,
        parsed_pages=2,
        unparsed_pages=(),
        status="complete",
        status_reason=None,
    )
    data.update(overrides)
    return SummaryCoverage(**data)


def supported_section() -> SummarySection:
    return SummarySection(
        status=SectionStatus.SUPPORTED,
        items=(
            SummaryItem(
                text="研究背景说明。",
                statement_kind=StatementKind.REPORTED,
                evidence_ids=("ev_001",),
            ),
        ),
    )


def make_summary(**overrides) -> Summary:
    sections = {key: supported_section() for key in SUMMARY_KEYS}
    data = dict(
        schema_version="1.0",
        paper_id="paper_001",
        revision=1,
        model_id="deepseek-chat",
        prompt_version="summary-v1",
        coverage=coverage(),
        sections=sections,
    )
    data.update(overrides)
    return Summary(**data)


class SummarySchemaTest(unittest.TestCase):
    def test_valid_summary(self) -> None:
        summary = make_summary()
        self.assertEqual(set(summary.sections), set(SUMMARY_KEYS))

    def test_missing_key_rejected(self) -> None:
        sections = {key: supported_section() for key in SUMMARY_KEYS[:-1]}
        with self.assertRaises(ValidationError):
            make_summary(sections=sections)

    def test_extra_key_rejected(self) -> None:
        sections = {key: supported_section() for key in SUMMARY_KEYS}
        sections["extra"] = supported_section()
        with self.assertRaises(ValidationError):
            make_summary(sections=sections)

    def test_supported_requires_items(self) -> None:
        with self.assertRaises(ValidationError):
            SummarySection(status=SectionStatus.SUPPORTED, items=())

    def test_item_requires_evidence(self) -> None:
        with self.assertRaises(ValidationError):
            SummaryItem(text="没有引用", evidence_ids=())

    def test_non_supported_requires_empty_items(self) -> None:
        with self.assertRaises(ValidationError):
            SummarySection(
                status=SectionStatus.NOT_FOUND,
                items=supported_section().items,
            )

    def test_insufficient_requires_note(self) -> None:
        with self.assertRaises(ValidationError):
            SummarySection(status=SectionStatus.INSUFFICIENT_EVIDENCE)

    def test_not_found_requires_complete_coverage(self) -> None:
        sections = {key: supported_section() for key in SUMMARY_KEYS}
        sections["limitations"] = SummarySection(status=SectionStatus.NOT_FOUND)
        partial = coverage(
            planned_units=2,
            processed_units=2,
            total_units=3,
            processed_ratio=2 / 3,
            status="partial",
            status_reason="只处理了 2/3",
        )
        with self.assertRaises(ValidationError):
            make_summary(sections=sections, coverage=partial)

    def test_not_found_allowed_when_complete(self) -> None:
        sections = {key: supported_section() for key in SUMMARY_KEYS}
        sections["limitations"] = SummarySection(status=SectionStatus.NOT_FOUND)
        summary = make_summary(sections=sections)
        self.assertIs(
            summary.sections["limitations"].status, SectionStatus.NOT_FOUND
        )


class CoverageTest(unittest.TestCase):
    def test_ratio_must_match(self) -> None:
        with self.assertRaises(ValidationError):
            coverage(processed_ratio=0.5)

    def test_complete_requires_full_ratio(self) -> None:
        with self.assertRaises(ValidationError):
            coverage(planned_units=2, processed_units=2, status="complete")

    def test_complete_rejects_unprocessed(self) -> None:
        with self.assertRaises(ValidationError):
            coverage(unprocessed_units=("chunk_x",))

    def test_processed_cannot_exceed_planned(self) -> None:
        with self.assertRaises(ValidationError):
            coverage(planned_units=2, processed_units=3)


if __name__ == "__main__":
    unittest.main()
