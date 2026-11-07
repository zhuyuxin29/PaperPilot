"""API 请求/响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthOut(BaseModel):
    status: str
    graph_mode: str


class PaperOut(BaseModel):
    id: str
    title: str
    file_name: str
    state: str
    revision: int
    page_count: int
    chunk_count: int
    graph_status: str


class CitationOut(BaseModel):
    evidence_id: str
    chunk_id: str
    page_number: int
    quote: str = ""


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ChatOut(BaseModel):
    run_id: str | None = None
    answer_kind: str | None = None
    status: str
    stop_reason: str | None
    answer: str
    citations: list[CitationOut]
    invalid_citations: list[str]
    steps: int
    tool_calls: int
    repair_attempts: int = 0
    trace: list[dict[str, Any]]


class SummaryOut(BaseModel):
    schema_version: str
    paper_id: str
    revision: int
    model_id: str
    prompt_version: str
    graph_digest: str = ""
    evidence_map: dict[str, str] = Field(default_factory=dict)
    generated_at: datetime
    coverage: dict[str, Any]
    sections: dict[str, Any]
