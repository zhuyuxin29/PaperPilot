"""模块 8.5（cleaning）验收测试：跨页页眉页脚清洗。"""

from __future__ import annotations

import unittest

from paperpilot.ingestion import chunker, cleaning
from paperpilot.schemas import Page


def page(index: int, text: str, *, paper_id: str = "paper_001") -> Page:
    return Page(
        paper_id=paper_id,
        page_index=index,
        page_number=index + 1,
        text=text,
        is_blank=not text.strip(),
    )


class StripHeadersTest(unittest.TestCase):
    def test_repeated_header_and_footer_removed(self) -> None:
        pages = [
            page(
                index,
                f"JOURNAL OF TESTS\nBody text number {index}.\nPage {index + 1} of 4",
            )
            for index in range(4)
        ]

        cleaned = cleaning.strip_repeated_headers(pages)

        self.assertEqual(len(cleaned), 4)
        for index, item in enumerate(cleaned):
            self.assertNotIn("JOURNAL OF TESTS", item.text)
            self.assertNotIn("Page ", item.text)
            self.assertEqual(item.text.strip(), f"Body text number {index}.")
            self.assertIn("header_stripped", item.quality_flags)
            self.assertEqual(item.page_number, index + 1)

    def test_unique_lines_are_kept(self) -> None:
        pages = [
            page(index, f"Different opening line {index}\nUnique body {index}.")
            for index in range(4)
        ]

        cleaned = cleaning.strip_repeated_headers(pages)

        self.assertEqual([item.text for item in cleaned], [item.text for item in pages])

    def test_too_few_pages_are_untouched(self) -> None:
        pages = [
            page(0, "SAME HEADER\nBody zero"),
            page(1, "SAME HEADER\nBody one"),
        ]
        self.assertEqual(
            cleaning.strip_repeated_headers(pages), tuple(pages)
        )

    def test_majority_threshold(self) -> None:
        pages = [
            page(index, f"REPEATED HEADER\nBody {index}.\nMore {index}.")
            for index in range(5)
        ]
        pages[4] = page(4, "Different header\nBody 4.\nMore 4.")

        cleaned = cleaning.strip_repeated_headers(pages)

        for index, item in enumerate(cleaned[:4]):
            self.assertEqual(item.text.strip(), f"Body {index}.\nMore {index}.")
        self.assertEqual(cleaned[4].text, pages[4].text)

    def test_page_with_only_header_becomes_blank(self) -> None:
        pages = [
            page(index, f"RUNNING HEAD\nBody {index}.\nMore {index}.")
            for index in range(3)
        ]
        pages.append(page(3, "RUNNING HEAD\n"))

        cleaned = cleaning.strip_repeated_headers(pages)

        self.assertTrue(cleaned[3].is_blank)
        self.assertEqual(cleaned[3].text, "")
        self.assertIn("blank", cleaned[3].quality_flags)
        self.assertIn("header_stripped", cleaned[3].quality_flags)

    def test_existing_flags_are_preserved(self) -> None:
        pages = [
            Page(
                paper_id="paper_001",
                page_index=index,
                page_number=index + 1,
                text=f"RUNNING HEAD\nBody {index}.\nMore {index}.",
                quality_flags=("extract_error",),
            )
            for index in range(3)
        ]

        cleaned = cleaning.strip_repeated_headers(pages)

        self.assertIn("extract_error", cleaned[0].quality_flags)
        self.assertIn("header_stripped", cleaned[0].quality_flags)

    def test_repeated_header_on_two_of_three_pages(self) -> None:
        pages = [
            page(index, f"JOURNAL HEADER\nBody {index}.\nMore {index}.")
            for index in range(3)
        ]

        cleaned = cleaning.strip_repeated_headers(pages)

        for index, item in enumerate(cleaned):
            self.assertNotIn("JOURNAL HEADER", item.text)
            self.assertEqual(item.text.strip(), f"Body {index}.\nMore {index}.")

    def test_date_header_stripped_even_if_unique(self) -> None:
        pages = [
            page(0, "2026年7月（下）第14期\nBody zero.\nMore zero."),
            page(1, "Body one.\nMore one.\nExtra one."),
            page(2, "Body two.\nMore two.\nExtra two."),
        ]

        cleaned = cleaning.strip_repeated_headers(pages)

        self.assertNotIn("2026年7月", cleaned[0].text)
        self.assertIn("Body zero.", cleaned[0].text)

    def test_body_line_starting_with_year_is_kept(self) -> None:
        pages = [
            page(index, f"Body {index}.\n2026年的研究显示这个结论成立。\nMore {index}.")
            for index in range(3)
        ]

        cleaned = cleaning.strip_repeated_headers(pages)

        for index, item in enumerate(cleaned):
            self.assertIn("2026年的研究显示这个结论成立。", item.text)

    def test_chunks_after_cleaning_have_consistent_offsets(self) -> None:
        pages = [
            page(index, f"RUNNING HEAD\nBody text {index}.\nPage {index + 1} of 3")
            for index in range(3)
        ]

        cleaned = cleaning.strip_repeated_headers(pages)
        chunks = chunker.chunk_pages(cleaned)

        self.assertTrue(chunks)
        for chunk in chunks:
            raw = cleaned[chunk.page_index].text
            self.assertEqual(raw[chunk.start_offset : chunk.end_offset], chunk.text)
            self.assertNotIn("RUNNING HEAD", chunk.text)
            self.assertNotIn("Page ", chunk.text)


if __name__ == "__main__":
    unittest.main()
