from __future__ import annotations

import os
import uuid
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import storage
from ..config import Settings
from ..llm import EmbeddingProvider

_PAPER_ID_RE = re.compile(r"^paper_\d{3,}$")


class IndexBuildError(RuntimeError):
    def __init__(self, message: str, code: str = "INDEX_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class IndexMeta:
    paper_id: str
    revision: int
    provider: str
    dimension: int
    chunk_count: int
    built_at: str
    path: Path


@dataclass(frozen=True)
class IndexData:
    chunk_ids: tuple[str, ...]
    vectors: np.ndarray
    provider: str
    dimension: int


def index_path(settings: Settings, paper_id: str, revision: int) -> Path:
    if not _PAPER_ID_RE.fullmatch(paper_id):
        raise ValueError(f"非法 paper_id：{paper_id!r}")
    return settings.index_dir / f"{paper_id}.r{int(revision)}.npz"


def build_paper_index(
    conn: sqlite3.Connection,
    settings: Settings,
    provider: EmbeddingProvider,
    paper_id: str,
    revision: int = 1,
) -> IndexMeta:
    paper = storage.get_paper(conn, paper_id)
    if paper is None:
        raise IndexBuildError(f"文献不存在：{paper_id}", "PAPER_NOT_FOUND")
    if paper.revision != revision:
        raise IndexBuildError(
            f"文献当前修订为 r{paper.revision}，不能按 r{revision} 建索引",
            "REVISION_MISMATCH",
        )

    chunks = storage.get_chunks(conn, paper_id, revision)
    if not chunks:
        raise IndexBuildError(
            f"文献没有可索引的片段：{paper_id} r{revision}", "NO_CHUNKS"
        )

    vectors = provider.embed([chunk.text for chunk in chunks])
    matrix = np.asarray(vectors, dtype=np.float32)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(chunks)
        or matrix.shape[1] != provider.dimension
    ):
        raise IndexBuildError(
            f"向量形状不符：期望 ({len(chunks)}, {provider.dimension})，"
            f"实际 {matrix.shape}",
            "BAD_VECTORS",
        )

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix = matrix / norms

    path = index_path(settings, paper_id, revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp.npz")
    np.savez(
        tmp,
        chunk_ids=np.array([chunk.chunk_id for chunk in chunks]),
        vectors=matrix,
        provider=np.array(provider.name),
        dimension=np.array(provider.dimension),
    )
    os.replace(tmp, path)

    if getattr(settings, "vector_backend", "numpy") == "chroma":
        try:
            from . import chroma_backend

            chroma_backend.build_chroma_index(settings, provider, chunks, matrix)
        except Exception as exc:
            raise IndexBuildError(
                f"Chroma 索引写入失败：{exc}", "CHROMA_BUILD_FAILED"
            ) from exc

    return IndexMeta(
        paper_id=paper_id,
        revision=revision,
        provider=provider.name,
        dimension=provider.dimension,
        chunk_count=len(chunks),
        built_at=datetime.now(timezone.utc).isoformat(),
        path=path,
    )


@lru_cache(maxsize=16)
def _read_index_file(
    path_text: str, mtime_ns: int, size: int
) -> tuple[tuple[str, ...], np.ndarray, str, int]:
    with np.load(path_text, allow_pickle=False) as data:
        chunk_ids = tuple(str(item) for item in data["chunk_ids"].tolist())
        vectors = np.array(data["vectors"], dtype=np.float32, copy=True)
        provider_name = str(data["provider"].item())
        dimension = int(data["dimension"].item())
    return chunk_ids, vectors, provider_name, dimension


def load_index(
    settings: Settings,
    paper_id: str,
    revision: int,
    *,
    provider: EmbeddingProvider | None = None,
) -> IndexData | None:
    path = index_path(settings, paper_id, revision)
    if not path.is_file():
        return None

    stat = path.stat()
    chunk_ids, vectors, provider_name, dimension = _read_index_file(
        str(path), stat.st_mtime_ns, stat.st_size
    )

    if provider is not None and (
        provider_name != provider.name or dimension != provider.dimension
    ):
        raise IndexBuildError(
            f"索引与当前 embedding 不匹配：索引为 {provider_name}/{dimension}，"
            f"当前为 {provider.name}/{provider.dimension}，需要重建索引",
            "INDEX_STALE",
        )
    return IndexData(
        chunk_ids=chunk_ids,
        vectors=vectors,
        provider=provider_name,
        dimension=dimension,
    )


def index_ready(
    settings: Settings,
    paper_id: str,
    revision: int,
    provider: EmbeddingProvider,
) -> bool:
    try:
        return (
            load_index(settings, paper_id, revision, provider=provider) is not None
        )
    except IndexBuildError:
        return False


def build_all_indexes(
    conn: sqlite3.Connection,
    settings: Settings,
    provider: EmbeddingProvider,
    paper_ids: Sequence[str] | None = None,
) -> list[IndexMeta]:
    selected = set(paper_ids) if paper_ids else None
    metas: list[IndexMeta] = []
    for paper in storage.list_papers(conn):
        if selected is not None and paper.id not in selected:
            continue
        try:
            metas.append(
                build_paper_index(conn, settings, provider, paper.id, paper.revision)
            )
        except IndexBuildError as exc:
            if exc.code == "NO_CHUNKS":
                continue
            raise
    return metas


def main() -> None:
    import sys

    from ..config import load_settings
    from ..llm import create_embedding_provider

    settings = load_settings()
    settings.ensure_dirs()
    provider = create_embedding_provider(settings)
    conn = storage.connect(settings.db_path)
    storage.ensure_schema(conn)
    try:
        metas = build_all_indexes(conn, settings, provider, sys.argv[1:] or None)
    finally:
        conn.close()

    if not metas:
        print("没有可建索引的文献（先上传 PDF）")
        return
    for meta in metas:
        print(
            f"{meta.paper_id} r{meta.revision}: {meta.chunk_count} 个片段 "
            f"-> {meta.path}"
        )


if __name__ == "__main__":
    main()
