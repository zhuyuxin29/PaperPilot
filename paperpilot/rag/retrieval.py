from __future__ import annotations

import re
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .. import storage
from ..config import Settings
from ..llm import EmbeddingProvider
from ..schemas import Chunk, Paper
from . import indexing, keyword

_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9._-]*")
_CJK_CHAR = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")

DEFAULT_TOP_K = 8
DEFAULT_PER_ROUTE = 20
DEFAULT_RRF_K = 60
DEFAULT_MIN_SIMILARITY = 0.0


# 按论文+修订缓存 chunk_id 集合，避免 Agent 每轮检索都查库。
_CURRENT_IDS_CACHE: "OrderedDict[tuple[str, str, int, str], frozenset[str]]" = (
    OrderedDict()
)
_CURRENT_IDS_CACHE_SIZE = 64


def _current_chunk_ids(
    conn: sqlite3.Connection, settings: Settings, paper: Paper
) -> frozenset[str]:
    key = (
        str(settings.data_dir),
        paper.id,
        paper.revision,
        paper.sha256,
    )
    cached = _CURRENT_IDS_CACHE.get(key)
    if cached is not None:
        _CURRENT_IDS_CACHE.move_to_end(key)
        return cached
    ids = frozenset(
        storage.get_chunk_ids(conn, paper.id, paper.revision)
    )
    _CURRENT_IDS_CACHE[key] = ids
    while len(_CURRENT_IDS_CACHE) > _CURRENT_IDS_CACHE_SIZE:
        _CURRENT_IDS_CACHE.popitem(last=False)
    return ids


class RetrievalError(RuntimeError):
    def __init__(self, message: str, code: str = "RETRIEVAL_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    paper_id: str
    page_number: int
    snippet: str
    score: float
    rank: int
    routes: tuple[str, ...]


def query_terms(query: str) -> tuple[str, ...]:
    lowered = query.lower()
    terms = set(_ASCII_WORD.findall(lowered))
    cjk = _CJK_CHAR.findall(lowered)
    terms.update(cjk)
    terms.update("".join(pair) for pair in zip(cjk, cjk[1:]))
    return tuple(sorted(term for term in terms if term))


def search_chunks(
    conn: sqlite3.Connection,
    settings: Settings,
    query: str,
    paper_ids: Sequence[str],
    *,
    provider: EmbeddingProvider,
    mode: str = "hybrid",
    top_k: int = DEFAULT_TOP_K,
    per_route: int = DEFAULT_PER_ROUTE,
    rrf_k: int = DEFAULT_RRF_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> list[Candidate]:
    query = (query or "").strip()
    if not query:
        raise RetrievalError("查询不能为空", "EMPTY_QUERY")
    if mode not in ("hybrid", "vector", "keyword"):
        raise RetrievalError(f"未知检索模式：{mode!r}", "BAD_MODE")
    if top_k < 1 or top_k > 50:
        raise RetrievalError("top_k 必须在 1~50 之间", "BAD_TOP_K")
    if per_route < 1 or per_route > 100:
        raise RetrievalError("per_route 必须在 1~100 之间", "BAD_PER_ROUTE")

    scope = _resolve_scope(conn, paper_ids)
    if not scope:
        raise RetrievalError("所选论文不存在或不可用", "NO_SCOPE")


    current_ids: dict[str, set[str]] = {
        paper.id: set(_current_chunk_ids(conn, settings, paper))
        for paper in scope
    }
    if not any(current_ids.values()):
        return []

    routes: dict[str, list[str]] = {}
    if mode in ("hybrid", "vector"):
        routes["vector"] = _vector_route(
            settings,
            query,
            scope,
            current_ids,
            provider,
            per_route,
            min_similarity,
        )
    if mode in ("hybrid", "keyword"):
        routes["keyword"] = _keyword_route(
            conn, settings, query, [paper.id for paper in scope], per_route
        )

    fused = _fusion_scores(routes, rrf_k=rrf_k)
    if not fused:
        return []

    chunk_map = storage.get_chunks_by_ids(conn, tuple(fused))
    return _build_candidates(
        fused,
        chunk_map,
        top_k=top_k,
        boost_terms=query_terms(query),
        term_boost=getattr(settings, "retrieval_term_boost", 0.0),
    )


def _resolve_scope(
    conn: sqlite3.Connection, paper_ids: Sequence[str]
) -> list[Paper]:
    scope: list[Paper] = []
    seen: set[str] = set()
    for paper_id in paper_ids:
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        paper = storage.get_paper(conn, paper_id)
        if paper is not None:
            scope.append(paper)
    return scope


def _vector_route(
    settings: Settings,
    query: str,
    scope: list[Paper],
    current_ids: dict[str, set[str]],
    provider: EmbeddingProvider,
    per_route: int,
    min_similarity: float,
) -> list[str]:
    if getattr(settings, "vector_backend", "numpy") == "chroma":
        from . import chroma_backend

        try:
            ids: list[str] = []
            for paper in scope:
                current = current_ids[paper.id]
                for chunk_id in chroma_backend.query_chroma(
                    settings,
                    provider,
                    query,
                    paper.id,
                    paper.revision,
                    top_k=per_route,
                ):
                    if chunk_id in current and chunk_id not in ids:
                        ids.append(chunk_id)
            return ids[:per_route]
        except chroma_backend.ChromaBackendError as exc:
            raise RetrievalError(exc.message, exc.code) from exc

    query_vector = np.asarray(provider.embed([query])[0], dtype=np.float32)
    if query_vector.shape[0] != provider.dimension:
        raise RetrievalError(
            f"查询向量维度不符：{query_vector.shape[0]} ≠ {provider.dimension}",
            "EMBEDDING_DIMENSION",
        )

    scored: list[tuple[str, float]] = []
    for paper in scope:
        try:
            data = indexing.load_index(
                settings, paper.id, paper.revision, provider=provider
            )
        except indexing.IndexBuildError as exc:
            raise RetrievalError(exc.message, "INDEX_STALE") from exc
        if data is None:
            raise RetrievalError("论文向量索引尚未就绪，请重新上传以重试建索引", "INDEX_MISSING")
        current = current_ids[paper.id]
        if data.vectors.shape[1] != query_vector.shape[0]:
            raise RetrievalError(
                f"索引维度与查询向量不一致：{data.vectors.shape[1]} ≠ "
                f"{query_vector.shape[0]}",
                "INDEX_STALE",
            )
        similarities = data.vectors @ query_vector
        for chunk_id, similarity in zip(data.chunk_ids, similarities):
            if chunk_id in current and float(similarity) > min_similarity:
                scored.append((chunk_id, float(similarity)))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return [chunk_id for chunk_id, _ in scored[:per_route]]


def _keyword_route(
    conn: sqlite3.Connection,
    settings: Settings,
    query: str,
    paper_ids: list[str],
    per_route: int,
) -> list[str]:
    try:
        return keyword.keyword_search(
            conn, settings, query, paper_ids, top_k=per_route
        )
    except keyword.KeywordIndexError:
        return []


def _fusion_scores(
    routes: dict[str, list[str]],
    *,
    rrf_k: int,
) -> dict[str, dict[str, object]]:
    fused: dict[str, dict[str, object]] = {}
    for route, ranked_ids in routes.items():
        for position, chunk_id in enumerate(ranked_ids, start=1):
            entry = fused.setdefault(chunk_id, {"score": 0.0, "routes": []})
            entry["score"] = float(entry["score"]) + 1.0 / (rrf_k + position)
            entry["routes"].append(route)
    return fused


def _build_candidates(
    fused: dict[str, dict[str, object]],
    chunk_map: dict[str, Chunk],
    *,
    top_k: int,
    boost_terms: tuple[str, ...] = (),
    term_boost: float = 0.0,
) -> list[Candidate]:
    terms = [term for term in boost_terms if len(term) >= 2 or term.isdigit()]
    if term_boost > 0 and terms:
        max_bonus = min(term_boost, 0.05)
        for chunk_id, entry in fused.items():
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue
            text = chunk.text.lower()
            hits = sum(1 for term in terms if term in text)
            digit_hit = any(term.isdigit() and term in text for term in terms)
            bonus = min(
                max_bonus,
                max_bonus * 0.2 * hits + (max_bonus * 0.4 if digit_hit else 0.0),
            )
            entry["score"] = float(entry["score"]) + bonus

    ordered = sorted(
        fused.items(),
        key=lambda item: (-float(item[1]["score"]), item[0]),
    )[:top_k]

    candidates: list[Candidate] = []
    for rank, (chunk_id, entry) in enumerate(ordered, start=1):
        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            continue
        text = chunk.text
        snippet = text if len(text) <= 200 else text[:200] + "…"
        candidates.append(
            Candidate(
                chunk_id=chunk_id,
                paper_id=chunk.paper_id,
                page_number=chunk.page_number,
                snippet=snippet,
                score=float(entry["score"]),
                rank=rank,
                routes=tuple(sorted(str(route) for route in entry["routes"])),
            )
        )
    return candidates
