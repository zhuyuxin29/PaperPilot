"""模块 5（chunker）验收测试。"""

from __future__ import annotations

import unittest

from paperpilot.ingestion.chunker import (
    CHUNKER_VERSION,
    DEFAULT_CONFIG,
    ChunkingConfig,
    chunk_pages,
    estimate_tokens,
)
from paperpilot.schemas import Page


def page(text: str, index: int = 0, paper_id: str = "paper_001") -> Page:
    return Page(
        paper_id=paper_id,
        page_index=index,
        page_number=index + 1,
        text=text,
        is_blank=not text.strip(),
    )


class ChunkInvariantsTest(unittest.TestCase):
    def test_offsets_match_raw_text(self) -> None:
        pages = [
            page("第一段内容。\n\n第二段内容，稍微长一点点。", index=0),
            page("Second page text.\n\nAnother paragraph.", index=1),
        ]
        chunks = chunk_pages(pages)

        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            raw = pages[chunk.page_index].text
            self.assertEqual(raw[chunk.start_offset : chunk.end_offset], chunk.text)
            self.assertGreater(chunk.end_offset, chunk.start_offset)
            self.assertGreater(chunk.token_estimate or 0, 0)

    def test_ordinal_is_global_and_sequential(self) -> None:
        pages = [page("Page one text.", index=0), page("Page two text.", index=1)]
        chunks = chunk_pages(pages)
        self.assertEqual([c.ordinal for c in chunks], list(range(len(chunks))))
        self.assertEqual([c.page_number for c in chunks], [1, 2])

    def test_chunk_ids_are_unique_and_stable(self) -> None:
        pages = [page("Some text here.\n\nMore text.", index=0)]
        first = chunk_pages(pages)
        second = chunk_pages(pages)

        self.assertEqual([c.chunk_id for c in first], [c.chunk_id for c in second])
        self.assertEqual(len({c.chunk_id for c in first}), len(first))
        for chunk in first:
            self.assertTrue(chunk.chunk_id.startswith("paper_001:r1:"))
            self.assertIn(f":{CHUNKER_VERSION}:", chunk.chunk_id)

    def test_blank_pages_are_skipped(self) -> None:
        pages = [
            page("Real content.", index=0),
            page("", index=1),
        ]
        chunks = chunk_pages(pages)
        self.assertTrue(all(c.page_number == 1 for c in chunks))

    def test_empty_page_list(self) -> None:
        self.assertEqual(chunk_pages([]), [])

    def test_leading_whitespace_trimmed(self) -> None:
        raw = "\n  \nLeading spaces content."
        chunks = chunk_pages([page(raw)])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_offset, raw.index("Leading"))
        self.assertEqual(chunks[0].text, "Leading spaces content.")


class MergeAndSplitTest(unittest.TestCase):
    def test_short_paragraphs_merge(self) -> None:
        raw = "First paragraph.\n\nSecond paragraph."
        chunks = chunk_pages([page(raw)])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, raw)

    def test_long_block_splits_within_max_tokens(self) -> None:
        raw = "x" * 2500
        config = ChunkingConfig(target_tokens=200, max_tokens=200)
        chunks = chunk_pages([page(raw)], config=config)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk.token_estimate or 0, config.max_tokens)
            self.assertEqual(raw[chunk.start_offset : chunk.end_offset], chunk.text)
        self.assertEqual(chunks[0].start_offset, 0)
        self.assertEqual(chunks[-1].end_offset, len(raw))
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertEqual(earlier.end_offset, later.start_offset)

    def test_newlines_inside_paragraph_kept(self) -> None:
        raw = "Line one\nline two\n\nLine three"
        chunks = chunk_pages([page(raw)])
        self.assertEqual(len(chunks), 1)
        self.assertIn("Line one\nline two", chunks[0].text)

    def test_estimate_tokens(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("中文"), 2)
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertGreater(estimate_tokens("中文 mixed text"), 2)


class SentenceBoundaryTest(unittest.TestCase):
    def test_chinese_sentences_are_not_split(self) -> None:
        raw = "".join(f"第{index}句内容，用于测试句子边界。" for index in range(40))
        config = ChunkingConfig(target_tokens=20, max_tokens=100)
        chunks = chunk_pages([page(raw)], config=config)

        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertTrue(chunk.text.strip().endswith("。"), chunk.text)
            self.assertEqual(
                raw[chunk.start_offset : chunk.end_offset], chunk.text
            )

    def test_english_sentences_are_not_split(self) -> None:
        raw = "First sentence. Second sentence! Third sentence? Fourth one."
        config = ChunkingConfig(target_tokens=3, max_tokens=10)
        chunks = chunk_pages([page(raw)], config=config)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.text.strip()[-1] in ".!?", chunk.text)

    def test_single_long_sentence_falls_back_to_window(self) -> None:
        raw = "x" * 500
        config = ChunkingConfig(target_tokens=50, max_tokens=100)
        chunks = chunk_pages([page(raw)], config=config)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk.token_estimate or 0, config.max_tokens)
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertEqual(earlier.end_offset, later.start_offset)


class ConfigTest(unittest.TestCase):
    def test_default_config_values(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.target_tokens, 600)
        self.assertEqual(DEFAULT_CONFIG.max_tokens, 900)
        self.assertEqual(DEFAULT_CONFIG.chunker_version, "v2")

    def test_invalid_config(self) -> None:
        with self.assertRaises(ValueError):
            ChunkingConfig(target_tokens=0)
        with self.assertRaises(ValueError):
            ChunkingConfig(target_tokens=100, max_tokens=50)
        with self.assertRaises(ValueError):
            ChunkingConfig(min_tokens=0)
        with self.assertRaises(ValueError):
            ChunkingConfig(chunker_version="  ")


if __name__ == "__main__":
    unittest.main()
