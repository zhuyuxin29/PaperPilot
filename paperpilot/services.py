"""Shared preparation workflows for UI and API; preserve usable revisions on failure."""
from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager

from . import storage
from .ingestion import pipeline, parser, cleaning, chunker
from .llm import create_embedding_provider
from .rag import indexing
from .schemas import PaperState

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
