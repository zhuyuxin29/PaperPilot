from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"
CHUNK_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{5,199}$"
_PAPER_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,63}$"

_sha256_re = re.compile(SHA256_PATTERN)


class GraphMode(str, Enum):
    OFF = "off"
    AUTO = "auto"
    ON = "on"


class PaperState(str, Enum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    READY = "ready"
    PARTIAL = "partial"
    FAILED = "failed"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Paper(_Frozen):
    id: str = Field(pattern=_PAPER_ID_PATTERN)
    title: str = Field(min_length=1, max_length=500)
    file_name: str = Field(min_length=1, max_length=255)
    sha256: str = Field(pattern=SHA256_PATTERN)
    revision: int = Field(default=1, ge=1)
    page_count: int = Field(default=0, ge=0)
    state: PaperState = PaperState.UPLOADED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("sha256", mode="before")
    @classmethod
    def _normalize_sha256(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class Page(_Frozen):
    paper_id: str = Field(pattern=_PAPER_ID_PATTERN)
    revision: int = Field(default=1, ge=1)
    page_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    text: str = ""
    is_blank: bool = False
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_invariants(self) -> "Page":
        if self.page_number != self.page_index + 1:
            raise ValueError(
                f"page_number 必须等于 page_index + 1，"
                f"当前 page_index={self.page_index}, page_number={self.page_number}"
            )
        if self.is_blank != (not self.text.strip()):
            raise ValueError(
                "is_blank 必须与 text 是否为空一致：空文本页必须标记 is_blank=True"
            )
        return self


class Chunk(_Frozen):
    chunk_id: str = Field(pattern=CHUNK_ID_PATTERN)
    paper_id: str = Field(pattern=_PAPER_ID_PATTERN)
    revision: int = Field(default=1, ge=1)
    page_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=1)
    token_estimate: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_invariants(self) -> "Chunk":
        if self.page_number != self.page_index + 1:
            raise ValueError(
                f"page_number 必须等于 page_index + 1，"
                f"当前 page_index={self.page_index}, page_number={self.page_number}"
            )
        if self.end_offset <= self.start_offset:
            raise ValueError(
                f"end_offset 必须大于 start_offset："
                f"[{self.start_offset}, {self.end_offset})"
            )
        if not self.text.strip():
            raise ValueError("chunk.text 不能为空白")
        return self


def page_number_for_index(page_index: int) -> int:
    if page_index < 0:
        raise ValueError(f"page_index 不能为负数：{page_index}")
    return page_index + 1


def is_valid_sha256(value: str) -> bool:
    return bool(_sha256_re.fullmatch(value.strip().lower()))


SUMMARY_KEYS: tuple[str, ...] = (
    "background",
    "research_question",
    "methods",
    "experiments",
    "conclusions",
    "contributions",
    "limitations",
)


class SectionStatus(str, Enum):
    SUPPORTED = "supported"
    NOT_FOUND = "not_found"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class StatementKind(str, Enum):
    REPORTED = "reported_by_paper"
    ANALYSIS = "assistant_analysis"


class SummaryItem(_Frozen):
    text: str = Field(min_length=1, max_length=2000)
    statement_kind: StatementKind = StatementKind.REPORTED
    evidence_ids: tuple[str, ...] = Field(min_length=1)


class SummarySection(_Frozen):
    status: SectionStatus
    items: tuple[SummaryItem, ...] = ()
    note: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _check(self) -> "SummarySection":
        if self.status is SectionStatus.SUPPORTED:
            if not self.items:
                raise ValueError("supported 状态必须至少包含一个条目")
        else:
            if self.items:
                raise ValueError(f"{self.status.value} 状态必须为空条目")
            if self.status is SectionStatus.INSUFFICIENT_EVIDENCE and not (
                self.note or ""
            ).strip():
                raise ValueError("insufficient_evidence 必须填写 note 说明缺口")
        return self


class SummaryCoverage(_Frozen):
    planned_units: int = Field(ge=0)
    processed_units: int = Field(ge=0)
    failed_units: tuple[str, ...] = ()
    unprocessed_units: tuple[str, ...] = ()
    total_units: int = Field(ge=1)
    processed_ratio: float = Field(ge=0, le=1)
    parsed_pages: int = Field(ge=0)
    unparsed_pages: tuple[int, ...] = ()
    status: Literal["complete", "partial"]
    status_reason: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _check(self) -> "SummaryCoverage":
        if self.processed_units > self.planned_units:
            raise ValueError("processed_units 不能大于 planned_units")
        if self.planned_units > self.total_units:
            raise ValueError("planned_units 不能大于 total_units")
        expected = self.processed_units / self.total_units
        if abs(self.processed_ratio - expected) > 1e-6:
            raise ValueError(
                f"processed_ratio 与 processed/total 不一致："
                f"{self.processed_ratio} ≠ {expected:.6f}"
            )
        if self.status == "complete":
            if self.unparsed_pages:
                raise ValueError("存在未解析页面时不能声明全文覆盖完整")
            if self.failed_units or self.unprocessed_units:
                raise ValueError("complete 时不能有失败或未处理的单元")
            if self.processed_units != self.planned_units:
                raise ValueError("complete 时 planned 与 processed 必须相等")
            if abs(self.processed_ratio - 1.0) > 1e-9:
                raise ValueError("complete 时文档覆盖比例必须为 1")
        return self


class Summary(_Frozen):
    schema_version: str = Field(default="1.0", min_length=1, max_length=20)
    paper_id: str = Field(pattern=_PAPER_ID_PATTERN)
    revision: int = Field(ge=1)
    model_id: str = Field(min_length=1, max_length=200)
    prompt_version: str = Field(min_length=1, max_length=100)
    graph_digest: str = Field(default="", max_length=32)
    evidence_map: dict[str, str] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    coverage: SummaryCoverage
    sections: dict[str, SummarySection]

    @model_validator(mode="after")
    def _check(self) -> "Summary":
        keys = set(self.sections)
        expected = set(SUMMARY_KEYS)
        missing = expected - keys
        extra = keys - expected
        if missing:
            raise ValueError(f"缺少要点类别：{sorted(missing)}")
        if extra:
            raise ValueError(f"出现未知要点类别：{sorted(extra)}")
        if any(
            section.status is SectionStatus.NOT_FOUND
            for section in self.sections.values()
        ):
            if self.coverage.status != "complete" or abs(
                self.coverage.processed_ratio - 1.0
            ) > 1e-9:
                raise ValueError("只有覆盖完整时才能标记 not_found")
        return self
