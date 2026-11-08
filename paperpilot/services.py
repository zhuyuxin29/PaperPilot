"""Shared preparation workflows for UI and API; preserve usable revisions on failure."""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager

from . import storage
from .ingestion import pipeline, parser, cleaning, chunker
from .llm import create_embedding_provider
from .rag import indexing
from .schemas import Paper, PaperState

logger = logging.getLogger(__name__)
_BACKGROUND_THREADS: dict[str, threading.Thread] = {}

_locks: dict[str, threading.RLock] = {}
_guard = threading.Lock()


@contextmanager
def document_lock(settings):
    # One local process may have multiple Streamlit sessions / API threads.
    key = str(settings.data_dir.resolve())
    with _guard:
        lock = _locks.setdefault(key, threading.RLock())
    with lock:
        yield


def graph_ready(conn, paper_ids):
    for paper_id in paper_ids:
        paper = storage.get_paper(conn, paper_id)
        state = storage.get_graph_state(conn, paper_id)
        if paper is None or state is None or state["revision"] != paper.revision or state["status"] != "ready":
            return False
    return bool(paper_ids)


def ensure_index(conn, settings, paper_id, provider=None):
    provider = provider or create_embedding_provider(settings)
    paper = storage.get_paper(conn, paper_id)
    if paper is None:
        raise ValueError(f"文献不存在：{paper_id}")
    if not indexing.index_ready(settings, paper.id, paper.revision, provider):
        indexing.build_paper_index(conn, settings, provider, paper.id, paper.revision)


def prepare_paper(data, file_name, settings, conn, *, provider=None):
    """Keep parsed documents when indexing fails; uploading again retries the index."""
    with document_lock(settings):
        result = pipeline.ingest_pdf(data, file_name, settings, conn)
        try:
            ensure_index(conn, settings, result.paper.id, provider)
        except Exception:
            storage.update_paper_state(conn, result.paper.id, PaperState.PARTIAL)
            raise
        pages = storage.get_pages(conn, result.paper.id, result.paper.revision)
        state = PaperState.READY if pipeline._fully_parsed(pages) else PaperState.PARTIAL
        storage.update_paper_state(conn, result.paper.id, state)
        from dataclasses import replace
        return replace(result, paper=storage.get_paper(conn, result.paper.id))


def start_background_ingest(
    conn: sqlite3.Connection,
    settings,
    data: bytes,
    file_name: str,
) -> tuple[Paper, bool]:
    """登记 PARSING 占位记录并提交后台入库；返回（论文, 是否新建任务）。

    已存在且未失败的同一文件直接返回（duplicated）；失败记录复用原 ID 重试。
    """
    with document_lock(settings):
        sha256 = hashlib.sha256(data).hexdigest()
        existing = storage.find_paper_by_sha256(conn, sha256)
        if existing is not None and existing.state is not PaperState.FAILED:
            return existing, False
        if existing is not None:
            storage.delete_paper(conn, existing.id)
            paper_id = existing.id
        else:
            paper_id = pipeline._next_paper_id(settings, conn)
        display_name = pipeline._display_name(file_name)
        paper = Paper(
            id=paper_id,
            title=pipeline._title_from(display_name),
            file_name=display_name,
            sha256=sha256,
            page_count=0,
            state=PaperState.PARSING,
        )
        storage.save_paper(conn, paper)

    thread = threading.Thread(
        target=_background_ingest,
        args=(settings, data, file_name, paper_id),
        name=f"paperpilot-ingest-{paper_id}",
        daemon=True,
    )
    _BACKGROUND_THREADS[paper_id] = thread
    thread.start()
    return paper, True


def wait_for_background(paper_id: str, timeout: float = 10.0) -> None:
    """等待该论文的后台入库线程结束（测试/维护入口使用）。"""
    thread = _BACKGROUND_THREADS.get(paper_id)
    if thread is not None:
        thread.join(timeout)


def _background_ingest(settings, data: bytes, file_name: str, paper_id: str) -> None:
    """后台线程：解析/分块/入库/建索引，失败时把占位记录标为 FAILED。"""
    conn = None
    try:
        conn = storage.connect(settings.db_path)
        storage.ensure_schema(conn)
        with document_lock(settings):
            result = pipeline.ingest_pdf(
                data, file_name, settings, conn, paper_id=paper_id
            )
            try:
                ensure_index(conn, settings, paper_id)
            except Exception:
                logger.exception("后台入库建索引失败：%s", paper_id)
                storage.update_paper_state(conn, paper_id, PaperState.PARTIAL)
                return
            pages = storage.get_pages(conn, paper_id, result.paper.revision)
            state = (
                PaperState.READY
                if pipeline._fully_parsed(pages)
                else PaperState.PARTIAL
            )
            storage.update_paper_state(conn, paper_id, state)
    except Exception:
        logger.exception("后台入库失败：%s", paper_id)
        if conn is not None:
            try:
                storage.update_paper_state(conn, paper_id, PaperState.FAILED)
            except Exception:
                logger.exception("后台入库失败状态写入失败：%s", paper_id)
        shutil.rmtree(settings.paper_dir(paper_id), ignore_errors=True)
    finally:
        if conn is not None:
            conn.close()
        _BACKGROUND_THREADS.pop(paper_id, None)


def reprocess_paper(settings, conn, paper_id, *, provider=None):
    """Build a new revision before atomically switching SQLite's current revision.

    The source PDF and older pages/chunks/summaries remain intact. Derived files
    for an unpublished revision may be retried; they are never queried as current.
    """
    with document_lock(settings):
        paper = storage.get_paper(conn, paper_id)
        if paper is None:
            raise ValueError(f"文献不存在：{paper_id}")
        source = settings.paper_dir(paper_id) / "source.pdf"
        if not source.is_file():
            raise FileNotFoundError("找不到原始 PDF，无法重新处理")
        revision = paper.revision + 1
        parsed = parser.parse_pdf(source, paper_id=paper_id, revision=revision)
        pages = cleaning.strip_repeated_headers(parsed.pages)
        chunks = chunker.chunk_pages(pages)
        if not chunks:
            raise pipeline.IngestError("新修订没有可用片段，已保留原版本", "NO_CHUNKS")
        updated = paper.model_copy(update={
            "revision": revision, "page_count": len(pages),
            "state": PaperState.READY if pipeline._fully_parsed(pages) else PaperState.PARTIAL,
        })
        temporary = storage.connect(":memory:")
        snapshot = settings.paper_dir(paper_id) / f"chunks.r{revision}.jsonl"
        pending = snapshot.with_name(snapshot.name + f".{uuid.uuid4().hex}.tmp")
        try:
            storage.ensure_schema(temporary)
            storage.save_document(temporary, updated, pages, chunks)
            indexing.build_paper_index(
                temporary, settings, provider or create_embedding_provider(settings),
                paper_id, revision,
            )
            pipeline._write_chunks_snapshot(pending, chunks)
            os.replace(pending, snapshot)
            # This transaction is the publication point. Prior revisions survive.
            storage.save_document(conn, updated, pages, chunks)
        finally:
            temporary.close()
            pending.unlink(missing_ok=True)
        return pipeline.IngestResult(
            paper=updated, paper_dir=settings.paper_dir(paper_id),
            page_count=len(pages), chunk_count=len(chunks),
            blank_page_numbers=tuple(p.page_number for p in pages if p.is_blank),
            duplicated=False,
        )
