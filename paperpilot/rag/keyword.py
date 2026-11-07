"""关键词检索（模块 12.5）：jieba 分词 + 自实现 BM25。

为什么不用 FTS5：
- 当前语料规模小（几千片段），Python BM25 足够快，且便于单元测试；
- 不引入 FTS5 虚拟表和查询语法转义；
- 接口保持不变，未来可替换为 FTS5 实现。

索引按“论文 + 修订”存放：
``{data_dir}/index/paper_001.r1.jieba-v1.bm25.json``

规则：
- 入库和查询使用同一套 jieba 词典与自定义词，词典版本写入索引；
- IDF 在**本次检索的论文范围**内统计（跨论文比较不偏）；
- 返回 chunk_id 排名，最终与向量路 RRF 融合。
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jieba

from .. import storage
from ..config import Settings

TOKENIZER_VERSION = "jieba-v2"

_CJK_LINEBREAK = re.compile(
    r"(?<=[\u4e00-\u9fff])[ \t]*\n[ \t]*(?=[\u4e00-\u9fff])"
)
BM25_K1 = 1.2
BM25_B = 0.75

DOMAIN_WORDS = (
    "RAG",
    "GraphRAG",
    "Neo4j",
    "BM25",
    "FTS5",
    "LLM",
    "bge-m3",
    "检索增强生成",
    "知识图谱",
    "向量检索",
    "混合检索",
    "文献要点",
    "智能问答",
    "项目式教学",
)
for _word in DOMAIN_WORDS:
    jieba.add_word(_word)

_ALNUM_OR_CJK = re.compile(r"[0-9A-Za-z_\u3400-\u9fff]")


class KeywordIndexError(RuntimeError):
    def __init__(self, message: str, code: str = "KEYWORD_INDEX_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class KeywordIndex:
    paper_id: str
    revision: int
    chunk_ids: tuple[str, ...]
    lengths: tuple[int, ...]
    term_counts: tuple[dict[str, int], ...]
    df: dict[str, int]
    avgdl: float
    tokenizer_version: str


def tokenize(text: str) -> list[str]:
    """jieba 切词，过滤标点/空白，保留中文词与英文/数字词。"""
    text = _CJK_LINEBREAK.sub("", text)
    tokens: list[str] = []
    for raw in jieba.lcut(text):
        token = raw.strip().lower()
        if not token:
            continue
        if not _ALNUM_OR_CJK.search(token):
            continue
        if len(token) == 1 and not ("\u4e00" <= token <= "\u9fff"):
            continue
        tokens.append(token)
    return tokens


def index_path(settings: Settings, paper_id: str, revision: int) -> Path:
    return settings.index_dir / (
        f"{paper_id}.r{int(revision)}.{TOKENIZER_VERSION}.bm25.json"
    )


def _replace_with_retry(source: Path, target: Path, attempts: int = 5) -> None:
    """Windows 上目标文件可能被杀毒/索引服务短暂占用，重试替换。"""
    last_error: PermissionError | None = None
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error


def build_keyword_index(
    conn: sqlite3.Connection,
    settings: Settings,
    paper_id: str,
    revision: int | None = None,
) -> KeywordIndex:
    paper = storage.get_paper(conn, paper_id)
    if paper is None:
        raise KeywordIndexError(f"文献不存在：{paper_id}", "PAPER_NOT_FOUND")
    revision = revision or paper.revision
    chunks = storage.get_chunks(conn, paper_id, revision)
    if not chunks:
        raise KeywordIndexError(f"文献没有片段：{paper_id}", "NO_CHUNKS")

    chunk_ids: list[str] = []
    lengths: list[int] = []
    term_counts: list[dict[str, int]] = []
    df: Counter[str] = Counter()

    for chunk in chunks:
        tokens = tokenize(chunk.text)
        counts = Counter(tokens)
        chunk_ids.append(chunk.chunk_id)
        lengths.append(len(tokens))
        term_counts.append(dict(counts))
        for term in counts:
            df[term] += 1

    avgdl = sum(lengths) / len(lengths) if lengths else 0.0
    payload = {
        "paper_id": paper_id,
        "revision": revision,
        "tokenizer_version": TOKENIZER_VERSION,
        "chunk_ids": chunk_ids,
        "lengths": lengths,
        "term_counts": term_counts,
        "df": dict(df),
        "avgdl": avgdl,
    }
    path = index_path(settings, paper_id, revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    _replace_with_retry(tmp, path)

    return KeywordIndex(
        paper_id=paper_id,
        revision=revision,
        chunk_ids=tuple(chunk_ids),
        lengths=tuple(lengths),
        term_counts=tuple(term_counts),
        df=dict(df),
        avgdl=avgdl,
        tokenizer_version=TOKENIZER_VERSION,
    )


def load_keyword_index(
    settings: Settings, paper_id: str, revision: int
) -> KeywordIndex | None:
    path = index_path(settings, paper_id, revision)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("tokenizer_version") != TOKENIZER_VERSION:
        return None
    try:
        return KeywordIndex(
            paper_id=str(payload["paper_id"]),
            revision=int(payload["revision"]),
            chunk_ids=tuple(str(item) for item in payload["chunk_ids"]),
            lengths=tuple(int(item) for item in payload["lengths"]),
            term_counts=tuple(dict(item) for item in payload["term_counts"]),
            df={str(key): int(value) for key, value in payload["df"].items()},
            avgdl=float(payload["avgdl"]),
            tokenizer_version=str(payload["tokenizer_version"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def keyword_search(
    conn: sqlite3.Connection,
    settings: Settings,
    query: str,
    paper_ids: Sequence[str],
    *,
    top_k: int = 20,
) -> list[str]:
    """返回按 BM25 排序的 chunk_id；索引缺失时自动构建。"""
    terms = tokenize(query)
    if not terms or top_k <= 0:
        return []
    term_set = sorted(set(terms))

    indexes: list[KeywordIndex] = []
    for paper_id in dict.fromkeys(paper_ids):
        paper = storage.get_paper(conn, paper_id)
        if paper is None:
            continue
        index = load_keyword_index(settings, paper_id, paper.revision)
        if index is None:
            index = build_keyword_index(
                conn, settings, paper_id, paper.revision
            )
        indexes.append(index)
    if not indexes:
        return []

    total_docs = sum(len(index.chunk_ids) for index in indexes)
    total_length = sum(sum(index.lengths) for index in indexes)
    avgdl = total_length / total_docs if total_docs else 0.0
    df: Counter[str] = Counter()
    for index in indexes:
        for term, count in index.df.items():
            df[term] += count

    scores: list[tuple[str, float]] = []
    for index in indexes:
        for chunk_id, length, counts in zip(
            index.chunk_ids, index.lengths, index.term_counts
        ):
            score = 0.0
            for term in term_set:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                doc_freq = df.get(term, 0)
                idf = math.log(
                    1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5)
                )
                norm = BM25_K1 * (
                    1.0 - BM25_B + BM25_B * (length / avgdl if avgdl else 0.0)
                )
                score += idf * (freq * (BM25_K1 + 1.0)) / (freq + norm)
            if score > 0:
                scores.append((chunk_id, score))

    scores.sort(key=lambda item: (-item[1], item[0]))
    return [chunk_id for chunk_id, _ in scores[:top_k]]


def main() -> None:
    """python -m paperpilot.rag.keyword [paper_id ...]：构建关键词索引。"""
    import sys

    from ..config import load_settings

    settings = load_settings()
    settings.ensure_dirs()
    conn = storage.connect(settings.db_path)
    storage.ensure_schema(conn)
    try:
        paper_ids = sys.argv[1:] or [
            paper.id for paper in storage.list_papers(conn)
        ]
        for paper_id in paper_ids:
            index = build_keyword_index(conn, settings, paper_id)
            print(f"{paper_id} r{index.revision}: {len(index.chunk_ids)} chunks")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
