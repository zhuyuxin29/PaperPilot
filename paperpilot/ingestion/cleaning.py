from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from ..schemas import Page

MIN_PAGES_FOR_DETECTION = 3
REPEAT_RATIO = 0.6
DEFAULT_EDGE_LINES = 2
MAX_CANDIDATE_LENGTH = 200

_DIGITS = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = (".", "。", "!", "！", "?", "？")

_DATE_HEADER = re.compile(
    r"^\s*(?:"
    r"\d(?:\s*\d){3}\s*年\s*\d{1,2}\s*月"
    r"(?:\s*[（(]\s*[上下]\s*[）)])?"
    r"(?:\s*第\s*\d(?:\s*\d)*\s*期)?"
    r"|\d{4}\s*[.．/\-]\s*\d{1,2}\s*(?:[上下])?"
    r"(?:\s*第?\s*\d(?:\s*\d)*\s*期)?"
    r")\s*"
)


@dataclass(frozen=True)
class _Line:
    start: int
    end: int
    text: str


def _split_lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        content = raw_line.rstrip("\r\n")
        lines.append(_Line(start=offset, end=offset + len(raw_line), text=content))
        offset += len(raw_line)
    return lines


def _normalize(text: str) -> str:
    value = _WHITESPACE.sub(" ", text.strip().lower())
    return _DIGITS.sub("#", value)


def _is_candidate(text: str, max_length: int) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > max_length:
        return False
    return not stripped.endswith(_SENTENCE_END)


def _edge_indexes(lines: Sequence[_Line], edge_lines: int) -> list[int]:
    non_empty = [index for index, line in enumerate(lines) if line.text.strip()]
    if not non_empty:
        return []
    edge = max(0, edge_lines)
    return sorted(set(non_empty[:edge] + non_empty[-edge:]))


def _remove_spans(text: str, spans: Sequence[tuple[int, int]]) -> str:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    result = text
    for start, end in reversed(merged):
        result = result[:start] + result[end:]
    return result


def strip_repeated_headers(
    pages: Sequence[Page],
    *,
    min_pages: int = MIN_PAGES_FOR_DETECTION,
    ratio: float = REPEAT_RATIO,
    edge_lines: int = DEFAULT_EDGE_LINES,
    max_candidate_length: int = MAX_CANDIDATE_LENGTH,
) -> tuple[Page, ...]:
    page_list = tuple(pages)
    if len(page_list) < min_pages:
        return page_list

    counts: Counter[str] = Counter()
    for page in page_list:
        lines = _split_lines(page.text)

        if sum(1 for line in lines if line.text.strip()) < 3:
            continue
        for index in _edge_indexes(lines, edge_lines):
            candidate = lines[index].text
            if not _is_candidate(candidate, max_candidate_length):
                continue
            normalized = _normalize(candidate)
            if normalized:
                counts[normalized] += 1


    threshold = max(2, math.ceil(len(page_list) * ratio))
    noise = {value for value, count in counts.items() if count >= threshold}

    cleaned: list[Page] = []
    for page in page_list:
        lines = _split_lines(page.text)
        edge = set(_edge_indexes(lines, edge_lines))
        spans: list[tuple[int, int]] = []
        for index, line in enumerate(lines):
            if index not in edge:
                continue

            match = _DATE_HEADER.match(line.text)
            if match and match.end() > match.start():
                remainder = line.text[match.end() :].strip()
                if remainder:
                    spans.append(
                        (
                            line.start + match.start(),
                            line.start + match.end(),
                        )
                    )
                else:
                    spans.append((line.start, line.end))

            if _is_candidate(line.text, max_candidate_length) and _normalize(
                line.text
            ) in noise:
                spans.append((line.start, line.end))

        if not spans:
            cleaned.append(page)
            continue

        text = _remove_spans(page.text, spans)
        is_blank = not text.strip()
        flags = list(page.quality_flags)
        if "header_stripped" not in flags:
            flags.append("header_stripped")
        if is_blank and "blank" not in flags:
            flags.append("blank")
        if not is_blank and "blank" in flags:
            flags.remove("blank")

        cleaned.append(
            Page(
                paper_id=page.paper_id,
                revision=page.revision,
                page_index=page.page_index,
                page_number=page.page_number,
                text=text,
                is_blank=is_blank,
                quality_flags=tuple(flags),
            )
        )
    return tuple(cleaned)
