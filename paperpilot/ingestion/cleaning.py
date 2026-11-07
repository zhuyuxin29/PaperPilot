"""页眉页脚清洗（模块 8.5）：在解析后、分块前去掉重复的页眉页脚。

为什么需要：
- 学术 PDF 每页都有 ``期刊名 Page 11 of 16`` 之类的页眉页脚；
- 它们会混进片段，污染关键词检索、占用上下文，也让引用第一眼看到的是页眉。

做法：
- 只看每页顶部/底部各 ``edge_lines`` 行（非空行）；
- 行文本做归一化（数字 → ``#``、压缩空白），统计哪些行在多数页面重复出现；
  “多数”= ≥60% 页面且至少 2 页（短文档也能生效）；
- 顶部/底部的**日期型页眉**（如 ``2026年7月（下）第14期``、``2026.7下``）
  即使没有重复也删除；
- 只删除“出现在页面边缘且命中重复模式/日期模式”的内容，避免误删正文；
- 清洗后的 ``Page.text`` 作为后续分块的原文，片段偏移仍满足
  ``chunk.text == page.text[start:end]``；
- 原始 PDF 不改动，页上记录 ``header_stripped`` 质量标记，便于追溯。

注意：
- 页数太少（< 3 页）时无法可靠判断“重复模式”，直接原样返回；
- 若某页删完只剩空白，按契约标记为空白页。
"""

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
# 常见的期刊页眉日期，例如“2026年7月（下）第14期”“2026.7下”
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
    """只把“像页眉页脚”的行当候选：短、非空、不以句末标点结尾。"""
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
    """删除多个区间；重叠区间先合并，避免重复删除导致偏移错乱。"""
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
    """删除跨页重复的页眉页脚，返回新的 Page 元组（不修改输入）。"""
    page_list = tuple(pages)
    if len(page_list) < min_pages:
        return page_list

    counts: Counter[str] = Counter()
    for page in page_list:
        lines = _split_lines(page.text)
        # 只有 1~2 行的页无法区分“页眉”和正文，不参与重复模式统计
        if sum(1 for line in lines if line.text.strip()) < 3:
            continue
        for index in _edge_indexes(lines, edge_lines):
            candidate = lines[index].text
            if not _is_candidate(candidate, max_candidate_length):
                continue
            normalized = _normalize(candidate)
            if normalized:
                counts[normalized] += 1

    # 3 页的短文档也允许“2 页重复”成立，避免整份文档不足 3 次出现而漏删
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
            # 1) 顶部/底部的日期型页眉（即使不重复也删）
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
            # 2) 跨页重复的页眉页脚行
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
