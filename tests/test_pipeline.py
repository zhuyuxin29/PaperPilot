"""模块 6（pipeline）验收测试：入库编排、去重、落盘、失败清理。"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from paperpilot import storage
from paperpilot.config import Settings
from paperpilot.ingestion import parser, pipeline
from paperpilot.schemas import Paper, PaperState
from pdf_factory import build_pdf

CLEAN_PDF = build_pdf(["First page content.", "Second page content."])
BLANK_PDF = build_pdf(["First page content.", None, "Third page content."])


class PipelineTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name) / "data"
        self.settings = Settings(data_dir=self.data_dir, graph_mode="off")
        self.settings.ensure_dirs()
        self.conn = storage.connect(self.settings.db_path)
        storage.ensure_schema(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def ingest(self, data: bytes, name: str = "demo.pdf", **kwargs):
        return pipeline.ingest_pdf(data, name, self.settings, self.conn, **kwargs)

    def paper_dirs(self) -> list[Path]:
        return [
            path
            for path in self.data_dir.iterdir()
            if path.is_dir() and path.name.startswith("paper_")
        ]

    def tmp_entries(self) -> list[Path]:
        return list(self.settings.tmp_dir.iterdir())


class IngestSuccessTest(PipelineTestBase):
    def test_clean_pdf_creates_dir_and_index(self) -> None:
        result = self.ingest(CLEAN_PDF)

        self.assertFalse(result.duplicated)
        self.assertEqual(result.paper.id, "paper_001")
        self.assertIs(result.paper.state, PaperState.READY)
        self.assertEqual(result.page_count, 2)
        self.assertEqual(result.blank_page_numbers, ())

        paper_dir = self.data_dir / "paper_001"
        self.assertEqual(result.paper_dir, paper_dir)
        self.assertEqual((paper_dir / "source.pdf").read_bytes(), CLEAN_PDF)

        snapshot = paper_dir / "chunks.jsonl"
        self.assertTrue(snapshot.is_file())
        lines = snapshot.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), result.chunk_count)
        self.assertGreater(result.chunk_count, 0)

        stored_chunks = storage.get_chunks(self.conn, "paper_001")
        self.assertEqual(len(stored_chunks), result.chunk_count)
        self.assertEqual(storage.get_paper(self.conn, "paper_001"), result.paper)
        self.assertEqual(len(storage.get_pages(self.conn, "paper_001")), 2)
        self.assertEqual(self.tmp_entries(), [])

    def test_blank_page_marks_partial(self) -> None:
        result = self.ingest(BLANK_PDF)

        self.assertIs(result.paper.state, PaperState.PARTIAL)
        self.assertEqual(result.blank_page_numbers, (2,))
        self.assertEqual(result.page_count, 3)

    def test_chunks_snapshot_matches_index(self) -> None:
        result = self.ingest(CLEAN_PDF)
        stored = storage.get_chunks(self.conn, "paper_001")
        lines = (
            (self.data_dir / "paper_001" / "chunks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )

        self.assertEqual(len(lines), len(stored))
        for line, chunk in zip(lines, stored):
            record = json.loads(line)
            self.assertEqual(record["chunk_id"], chunk.chunk_id)
            self.assertEqual(record["text"], chunk.text)
            self.assertEqual(record["start_offset"], chunk.start_offset)
            self.assertEqual(record["end_offset"], chunk.end_offset)
        self.assertEqual(result.chunk_count, len(stored))

    def test_title_from_file_name(self) -> None:
        result = self.ingest(CLEAN_PDF, name="My Paper.pdf")
        self.assertEqual(result.paper.file_name, "My Paper.pdf")
        self.assertEqual(result.paper.title, "My Paper")

    def test_second_file_gets_next_id(self) -> None:
        first = self.ingest(CLEAN_PDF)
        second = self.ingest(BLANK_PDF)

        self.assertEqual(first.paper.id, "paper_001")
        self.assertEqual(second.paper.id, "paper_002")


class DedupTest(PipelineTestBase):
    def test_same_file_ingested_twice(self) -> None:
        first = self.ingest(CLEAN_PDF, name="a.pdf")
        second = self.ingest(CLEAN_PDF, name="renamed.pdf")

        self.assertFalse(first.duplicated)
        self.assertTrue(second.duplicated)
        self.assertEqual(second.paper.id, first.paper.id)
        self.assertEqual(second.paper.file_name, "a.pdf")
        self.assertEqual(len(self.paper_dirs()), 1)
        self.assertEqual(len(storage.list_papers(self.conn)), 1)

    def test_failed_record_is_reused(self) -> None:
        sha = hashlib.sha256(CLEAN_PDF).hexdigest()
        storage.save_paper(
            self.conn,
            Paper(
                id="paper_005",
                title="旧的失败记录",
                file_name="demo.pdf",
                sha256=sha,
                state=PaperState.FAILED,
            ),
        )

        result = self.ingest(CLEAN_PDF)

        self.assertEqual(result.paper.id, "paper_005")
        self.assertFalse(result.duplicated)
        self.assertIs(result.paper.state, PaperState.READY)
        self.assertTrue((self.data_dir / "paper_005").is_dir())


class FailureTest(PipelineTestBase):
    def test_invalid_file(self) -> None:
        with self.assertRaises(pipeline.IngestValidationError) as ctx:
            self.ingest(b"this is not a pdf")
        self.assertEqual(ctx.exception.code, "NOT_PDF")

        self.assertEqual(self.paper_dirs(), [])
        self.assertEqual(storage.list_papers(self.conn), [])
        self.assertEqual(self.tmp_entries(), [])

    def test_file_too_large(self) -> None:
        with self.assertRaises(pipeline.IngestValidationError) as ctx:
            self.ingest(CLEAN_PDF, max_bytes=10)
        self.assertEqual(ctx.exception.code, "FILE_TOO_LARGE")

    def test_encrypted_pdf_leaves_no_trace(self) -> None:
        encrypted = build_pdf(["Secret page"], encrypt_password="secret")

        with self.assertRaises(parser.PdfEncryptedError):
            self.ingest(encrypted)

        self.assertEqual(self.paper_dirs(), [])
        self.assertEqual(storage.list_papers(self.conn), [])
        self.assertEqual(self.tmp_entries(), [])


if __name__ == "__main__":
    unittest.main()
