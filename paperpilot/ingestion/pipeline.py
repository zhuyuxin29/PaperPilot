from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .. import storage
from ..config import Settings
from ..schemas import Chunk, Paper, PaperState
from . import chunker, cleaning, parser

MAX_PDF_BYTES = 50 * 1024 * 1024
_PAPER_DIR_RE = re.compile(r"^paper_(\d{3,})$")


class IngestError(Exception):
    code = "INGEST_ERROR"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class IngestValidationError(IngestError):
    code = "INVALID_FILE"


@dataclass(frozen=True)
class IngestResult:
    paper: Paper
    paper_dir: Path
    page_count: int
    chunk_count: int
    blank_page_numbers: tuple[int, ...]
    duplicated: bool


def ingest_pdf(
    data: bytes,
    file_name: str,
    settings: Settings,
    conn: sqlite3.Connection,
    *,
    max_bytes: int = MAX_PDF_BYTES,
    paper_id: str | None = None,
) -> IngestResult:
    _validate_upload(data, max_bytes)
    display_name = _display_name(file_name)
    sha256 = hashlib.sha256(data).hexdigest()

    if paper_id is not None:
        if not _PAPER_DIR_RE.fullmatch(paper_id):
            raise IngestValidationError(f"非法论文编号：{paper_id!r}", "BAD_PAPER_ID")
    else:
        existing = storage.find_paper_by_sha256(conn, sha256)
        if existing is not None and existing.state is not PaperState.FAILED:
            pages = storage.get_pages(conn, existing.id, existing.revision)
            chunks = storage.get_chunks(conn, existing.id, existing.revision)
            return IngestResult(
                paper=existing,
                paper_dir=settings.paper_dir(existing.id),
                page_count=existing.page_count or len(pages),
                chunk_count=len(chunks),
                blank_page_numbers=tuple(
                    page.page_number for page in pages if page.is_blank
                ),
                duplicated=True,
            )

        if existing is not None:
            storage.delete_paper(conn, existing.id)
            paper_id = existing.id
        else:
            paper_id = _next_paper_id(settings, conn)

    working_dir = settings.tmp_dir / paper_id
    final_dir = settings.paper_dir(paper_id)
    moved = False

    try:
        _cleanup(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        (working_dir / "source.pdf").write_bytes(data)

        parsed = parser.parse_pdf(working_dir / "source.pdf", paper_id=paper_id)
        pages = cleaning.strip_repeated_headers(parsed.pages)
        chunks = chunker.chunk_pages(pages)
        if not chunks:
            raise IngestError(
                "PDF 中没有可检索文字，未入库；请先做 OCR 或换一份可复制文字的 PDF",
                code="NO_CHUNKS",
            )
        _write_chunks_snapshot(working_dir / "chunks.jsonl", chunks)

        if final_dir.exists():
            raise IngestError(f"目标目录已存在：{final_dir}", code="TARGET_EXISTS")
        _rename_with_retry(working_dir, final_dir)
        moved = True

        paper = Paper(
            id=paper_id,
            title=_title_from(display_name),
            file_name=display_name,
            sha256=sha256,
            page_count=len(pages),
            state=(
                PaperState.READY
                if _fully_parsed(pages)
                else PaperState.PARTIAL
            ),
        )
        storage.save_document(conn, paper, pages, chunks)
    except Exception:
        _cleanup(working_dir)
        if moved:
            _cleanup(final_dir)
        raise

    return IngestResult(
        paper=paper,
        paper_dir=final_dir,
        page_count=len(pages),
        chunk_count=len(chunks),
        blank_page_numbers=tuple(
            page.page_number for page in pages if page.is_blank
        ),
        duplicated=False,
    )


def _validate_upload(data: bytes, max_bytes: int) -> None:
    if not data:
        raise IngestValidationError("文件为空", code="EMPTY_FILE")
    if len(data) > max_bytes:
        raise IngestValidationError(
            f"文件超过大小上限（{max_bytes} 字节）", code="FILE_TOO_LARGE"
        )
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise IngestValidationError("不是有效的 PDF 文件", code="NOT_PDF")


def _display_name(file_name: str) -> str:
    name = Path(file_name).name.strip()
    return (name or "document.pdf")[:255]


def _title_from(display_name: str) -> str:
    stem = Path(display_name).stem.strip()
    return (stem or "未命名文献")[:500]


def _fully_parsed(pages: tuple) -> bool:
    return all(not page.is_blank and not page.quality_flags for page in pages)


def _next_paper_id(settings: Settings, conn: sqlite3.Connection) -> str:
    highest = storage.get_sequence(conn, "paper_id")
    if settings.data_dir.is_dir():
        for path in settings.data_dir.iterdir():
            match = _PAPER_DIR_RE.fullmatch(path.name)
            if match:
                highest = max(highest, int(match.group(1)))
    for paper in storage.list_papers(conn):
        match = _PAPER_DIR_RE.fullmatch(paper.id)
        if match:
            highest = max(highest, int(match.group(1)))
    number = highest + 1
    storage.set_sequence(conn, "paper_id", number)
    return f"paper_{number:03d}"


def _write_chunks_snapshot(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.chunk_id,
                "page_index": chunk.page_index,
                "page_number": chunk.page_number,
                "ordinal": chunk.ordinal,
                "text": chunk.text,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _cleanup(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)

def _rename_with_retry(source: Path, target: Path, attempts: int = 3) -> None:
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            source.rename(target)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    assert last_error is not None
    raise last_error
