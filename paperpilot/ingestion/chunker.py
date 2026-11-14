from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

from ..schemas import Chunk, Page

CHUNKER_VERSION = "v2"
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = 600
    max_tokens: int = 900
    min_tokens: int = 1
    chunker_version: str = CHUNKER_VERSION

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("target_tokens 必须为正数")
        if self.max_tokens < self.target_tokens:
            raise ValueError("max_tokens 不能小于 target_tokens")
        if self.min_tokens <= 0:
            raise ValueError("min_tokens 必须为正数")
        if not self.chunker_version.strip():
            raise ValueError("chunker_version 不能为空")


DEFAULT_CONFIG = ChunkingConfig()


def estimate_tokens(text: str) -> int:
    cjk = len(_CJK_RE.findall(text))
    other = max(0, len(text) - cjk)
    return cjk + (other + 3) // 4


def chunk_pages(
    pages: Sequence[Page],
    config: ChunkingConfig = DEFAULT_CONFIG,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for page in pages:
        if page.is_blank or not page.text.strip():
            continue
        for start, end in _page_ranges(page, config):
            text = page.text[start:end]
            if not text.strip():
                continue
            chunk_id = _make_chunk_id(page, ordinal, start, end, text, config)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    paper_id=page.paper_id,
                    revision=page.revision,
                    page_index=page.page_index,
                    page_number=page.page_number,
                    ordinal=ordinal,
                    text=text,
                    start_offset=start,
                    end_offset=end,
                    token_estimate=estimate_tokens(text),
                )
            )
            ordinal += 1
    return chunks


def _page_ranges(page: Page, config: ChunkingConfig) -> list[tuple[int, int]]:
    blocks = list(_iter_blocks(page.text))
    if not blocks:
        return []

    ranges: list[tuple[int, int]] = []
    for start, end in blocks:
        block_tokens = estimate_tokens(page.text[start:end])
        if block_tokens > config.max_tokens:
            ranges.extend(
                _split_long_block(
                    page.text,
                    start,
                    end,
                    config.target_tokens,
                    config.max_tokens,
                )
            )
        else:
            ranges.append((start, end))

    merged: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end = 0
    current_tokens = 0
    for start, end in ranges:
        tokens = estimate_tokens(page.text[start:end])
        if current_start is None:
            current_start, current_end, current_tokens = start, end, tokens
            continue
        if current_tokens + tokens <= config.target_tokens:
            current_end = end
            current_tokens += tokens
        else:
            merged.append((current_start, current_end))
            current_start, current_end, current_tokens = start, end, tokens
    if current_start is not None:
        merged.append((current_start, current_end))
    return merged


def _iter_blocks(text: str) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    offset = 0
    block_start: int | None = None
    block_end = 0

    for line in text.splitlines(keepends=True):
        content = line.strip()
        if content:
            if block_start is None:
                block_start = offset + (len(line) - len(line.lstrip()))
            block_end = offset + len(line.rstrip())
        elif block_start is not None:
            blocks.append((block_start, block_end))
            block_start = None
        offset += len(line)

    if block_start is not None:
        blocks.append((block_start, block_end))
    return blocks


def _split_long_block(
    text: str,
    start: int,
    end: int,
    target_tokens: int,
    max_tokens: int,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end = 0
    current_tokens = 0

    for sentence_start, sentence_end in _sentence_spans(text, start, end):
        sentence_tokens = estimate_tokens(text[sentence_start:sentence_end])
        if sentence_tokens > max_tokens:
            if current_start is not None:
                ranges.append((current_start, current_end))
                current_start = None
            ranges.extend(
                _window_split(text, sentence_start, sentence_end, max_tokens)
            )
            continue

        if current_start is None:
            current_start, current_end, current_tokens = (
                sentence_start,
                sentence_end,
                sentence_tokens,
            )
        elif current_tokens + sentence_tokens <= target_tokens:
            current_end = sentence_end
            current_tokens += sentence_tokens
        else:
            ranges.append((current_start, current_end))
            current_start, current_end, current_tokens = (
                sentence_start,
                sentence_end,
                sentence_tokens,
            )

    if current_start is not None:
        ranges.append((current_start, current_end))
    return ranges


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    segment_start = start
    index = start
    while index < end:
        char = text[index]
        is_boundary = False
        if char in "。！？；\n":
            is_boundary = True
        elif char in ".!?;":
            next_char = text[index + 1] if index + 1 < end else " "
            is_boundary = next_char.isspace() or index + 1 >= end
        if is_boundary:
            spans.append((segment_start, index + 1))
            segment_start = index + 1
        index += 1
    if segment_start < end:
        spans.append((segment_start, end))

    cleaned: list[tuple[int, int]] = []
    for span_start, span_end in spans:
        while span_start < span_end and text[span_start].isspace():
            span_start += 1
        while span_end > span_start and text[span_end - 1].isspace():
            span_end -= 1
        if span_end > span_start:
            cleaned.append((span_start, span_end))
    return cleaned or [(start, end)]


def _window_split(
    text: str, start: int, end: int, max_tokens: int
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        lo, hi = cursor, end

        while lo < hi:
            mid = (lo + hi + 1) // 2
            if estimate_tokens(text[cursor:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        stop = lo
        if stop == cursor:
            stop = min(cursor + 1, end)
        ranges.append((cursor, stop))
        cursor = stop
    return ranges


def _make_chunk_id(
    page: Page,
    ordinal: int,
    start: int,
    end: int,
    text: str,
    config: ChunkingConfig,
) -> str:
    digest = hashlib.sha256(
        f"{page.page_index}:{start}:{end}:{text}".encode("utf-8")
    ).hexdigest()[:8]
    return (
        f"{page.paper_id}:r{page.revision}:p{page.page_number}"
        f":c{ordinal:04d}:{config.chunker_version}:{digest}"
    )
