"""Chroma 向量后端（可选，嵌入式持久化）。

只负责向量存取，元数据带 paper_id / revision / page_number；
检索时按论文 + 修订过滤，避免跨论文或旧修订泄漏。
"""

from __future__ import annotations

from typing import Sequence

from ..config import Settings
from ..llm import EmbeddingProvider
from ..schemas import Chunk

COLLECTION = "paper_chunks"


class ChromaBackendError(RuntimeError):
    def __init__(self, message: str, code: str = "CHROMA_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def _client(settings: Settings):
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - 依赖缺失时提示明确
        raise ChromaBackendError(
            "未安装 chromadb，请执行 pip install 'paperpilot[chroma]'",
            "CHROMA_NOT_INSTALLED",
        ) from exc
    from chromadb.config import Settings as ChromaSettings

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _collection(settings: Settings):
    client = _client(settings)
    return client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def build_chroma_index(
    settings: Settings,
    provider: EmbeddingProvider,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]] | None = None,
) -> None:
    if not chunks:
        return
    paper_id = chunks[0].paper_id
    revision = chunks[0].revision
    collection = _collection(settings)
    collection.delete(where={"$and": [{"paper_id": paper_id}, {"revision": revision}]})
    if vectors is None:
        vectors = provider.embed([chunk.text for chunk in chunks])
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=[list(map(float, vector)) for vector in vectors],
        metadatas=[
            {
                "paper_id": chunk.paper_id,
                "revision": chunk.revision,
                "page_number": chunk.page_number,
            }
            for chunk in chunks
        ],
    )


def query_chroma(
    settings: Settings,
    provider: EmbeddingProvider,
    query: str,
    paper_id: str,
    revision: int,
    *,
    top_k: int,
) -> list[str]:
    collection = _collection(settings)
    vector = provider.embed([query])[0]
    result = collection.query(
        query_embeddings=[list(map(float, vector))],
        n_results=top_k,
        where={"$and": [{"paper_id": paper_id}, {"revision": revision}]},
    )
    ids = result.get("ids") or []
    return list(ids[0]) if ids else []
