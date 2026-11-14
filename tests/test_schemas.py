from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from paperpilot import schemas
from paperpilot.schemas import Chunk, Page, Paper, PaperState, page_number_for_index

SHA = "a" * 64


class PaperTest(unittest.TestCase):
    def test_valid_paper(self) -> None:
        paper = Paper(
            id="paper_001",
            title="示例论文",
            file_name="demo.pdf",
            sha256=SHA,
            page_count=3,
        )
        self.assertEqual(paper.revision, 1)
        self.assertIs(paper.state, PaperState.UPLOADED)
        self.assertEqual(paper.created_at.tzinfo, timezone.utc)
        self.assertIsInstance(paper.created_at, datetime)

    def test_sha256_normalized(self) -> None:
        paper = Paper(
            id="paper_001",
            title="示例论文",
            file_name="demo.pdf",
            sha256="A" * 64,
        )
        self.assertEqual(paper.sha256, "a" * 64)

    def test_invalid_sha256(self) -> None:
        with self.assertRaises(ValidationError):
            Paper(
                id="paper_001",
                title="示例论文",
                file_name="demo.pdf",
                sha256="not-a-hash",
            )

    def test_extra_field_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Paper(
                id="paper_001",
                title="示例论文",
                file_name="demo.pdf",
                sha256=SHA,
                unknown="x",
            )

    def test_frozen(self) -> None:
        paper = Paper(
            id="paper_001",
            title="示例论文",
            file_name="demo.pdf",
            sha256=SHA,
        )
        with self.assertRaises(ValidationError):
            paper.title = "改标题"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        paper = Paper(
            id="paper_001",
            title="示例论文",
            file_name="demo.pdf",
            sha256=SHA,
        )
        restored = Paper.model_validate(paper.model_dump())
        self.assertEqual(restored, paper)


class PageTest(unittest.TestCase):
    def test_page_number_must_match_index(self) -> None:
        with self.assertRaises(ValidationError):
            Page(paper_id="paper_001", page_index=0, page_number=2, text="x")

    def test_valid_page(self) -> None:
        page = Page(
            paper_id="paper_001",
            page_index=4,
            page_number=5,
            text="论文正文第 5 页。",
        )
        self.assertFalse(page.is_blank)

    def test_blank_page_requires_flag(self) -> None:
        with self.assertRaises(ValidationError):
            Page(paper_id="paper_001", page_index=0, page_number=1, text="")

        blank = Page(
            paper_id="paper_001",
            page_index=0,
            page_number=1,
            text="  \n ",
            is_blank=True,
        )
        self.assertTrue(blank.is_blank)

    def test_negative_index_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Page(paper_id="paper_001", page_index=-1, page_number=0)

    def test_page_number_helper(self) -> None:
        self.assertEqual(page_number_for_index(0), 1)
        self.assertEqual(page_number_for_index(11), 12)
        with self.assertRaises(ValueError):
            page_number_for_index(-1)


class ChunkTest(unittest.TestCase):
    def test_valid_chunk(self) -> None:
        chunk = Chunk(
            chunk_id="paper_001:r1:p1:c0:ab12cd34",
            paper_id="paper_001",
            page_index=0,
            page_number=1,
            ordinal=0,
            text="片段文本",
            start_offset=10,
            end_offset=14,
        )
        self.assertEqual(chunk.page_number, 1)

    def test_invalid_offsets(self) -> None:
        with self.assertRaises(ValidationError):
            Chunk(
                chunk_id="paper_001:r1:p1:c0:ab12cd34",
                paper_id="paper_001",
                page_index=0,
                page_number=1,
                ordinal=0,
                text="片段文本",
                start_offset=10,
                end_offset=10,
            )

    def test_page_number_mismatch(self) -> None:
        with self.assertRaises(ValidationError):
            Chunk(
                chunk_id="paper_001:r1:p1:c0:ab12cd34",
                paper_id="paper_001",
                page_index=0,
                page_number=3,
                ordinal=0,
                text="片段文本",
                start_offset=0,
                end_offset=4,
            )

    def test_blank_text_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Chunk(
                chunk_id="paper_001:r1:p1:c0:ab12cd34",
                paper_id="paper_001",
                page_index=0,
                page_number=1,
                ordinal=0,
                text="   ",
                start_offset=0,
                end_offset=3,
            )

    def test_bad_chunk_id_format(self) -> None:
        with self.assertRaises(ValidationError):
            Chunk(
                chunk_id="含空格 的 id",
                paper_id="paper_001",
                page_index=0,
                page_number=1,
                ordinal=0,
                text="片段文本",
                start_offset=0,
                end_offset=4,
            )


class EnumAndHelperTest(unittest.TestCase):
    def test_graph_mode_values(self) -> None:
        self.assertEqual(
            [mode.value for mode in schemas.GraphMode], ["off", "auto", "on"]
        )

    def test_paper_states(self) -> None:
        self.assertEqual(
            [state.value for state in schemas.PaperState],
            ["uploaded", "parsing", "ready", "partial", "failed"],
        )

    def test_is_valid_sha256(self) -> None:
        self.assertTrue(schemas.is_valid_sha256(SHA))
        self.assertTrue(schemas.is_valid_sha256(SHA.upper()))
        self.assertFalse(schemas.is_valid_sha256("abc"))
        self.assertFalse(schemas.is_valid_sha256("g" * 64))


if __name__ == "__main__":
    unittest.main()
