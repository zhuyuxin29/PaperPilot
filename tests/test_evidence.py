"""模块 8（evidence）验收测试：证据登记、原文回读、引用校验。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from doc_factory import make_document
from paperpilot import storage
from paperpilot.config import Settings
from paperpilot.evidence.passages import read_passages
from paperpilot.evidence.registry import EvidenceRegistry
from paperpilot.evidence.validation import (
    EvidenceError,
    assert_evidence_in_scope,
    is_valid_citation,
)
from paperpilot.schemas import Chunk, Page, Paper


class EvidenceTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = Settings(
            data_dir=Path(self._tmp.name) / "data", graph_mode="off"
        )
        self.settings.ensure_dirs()
        self.conn = storage.connect(self.settings.db_path)
        storage.ensure_schema(self.conn)
        self.paper, self.pages, self.chunks = make_document()
        storage.save_document(self.conn, self.paper, self.pages, self.chunks)
        self.registry = EvidenceRegistry()

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def chunk_id(self, index: int = 0) -> str:
        return self.chunks[index].chunk_id


class RegistryTest(EvidenceTestBase):
    def test_same_chunk_registered_once(self) -> None:
        first = self.registry.register(self.chunks[0])
        second = self.registry.register(self.chunks[0])
        self.assertEqual(first.evidence_id, second.evidence_id)
        self.assertEqual(first.evidence_id, "ev_001")
        self.assertEqual(len(self.registry), 1)

    def test_ids_are_sequential(self) -> None:
        self.registry.register(self.chunks[0])
        self.registry.register(self.chunks[1])
        self.assertEqual(
            [record.evidence_id for record in self.registry.all()],
            ["ev_001", "ev_002"],
        )

    def test_get_and_clear(self) -> None:
        record = self.registry.register(self.chunks[0])
        self.assertEqual(self.registry.get(record.evidence_id), record)
        self.assertIsNone(self.registry.get("ev_999"))
        self.registry.clear()
        self.assertEqual(len(self.registry), 0)


class PassagesTest(EvidenceTestBase):
    def test_read_single_passage(self) -> None:
        passages = read_passages(self.conn, self.registry, [self.chunk_id(0)])

        self.assertEqual(len(passages), 1)
        passage = passages[0]
        self.assertEqual(passage.evidence.chunk_id, self.chunk_id(0))
        self.assertEqual(passage.evidence.quote, self.chunks[0].text)
        self.assertEqual(passage.evidence.page_number, 1)
        self.assertEqual(len(self.registry), 1)

    def test_batch_read_dedupes(self) -> None:
        passages = read_passages(
            self.conn,
            self.registry,
            [self.chunk_id(0), self.chunk_id(1), self.chunk_id(0)],
        )
        self.assertEqual(len(passages), 2)
        self.assertEqual(len(self.registry), 2)

    def test_empty_batch(self) -> None:
        with self.assertRaises(EvidenceError) as ctx:
            read_passages(self.conn, self.registry, [])
        self.assertEqual(ctx.exception.code, "EMPTY_BATCH")

    def test_batch_too_large(self) -> None:
        paper = Paper(
            id="paper_999",
            title="many chunks",
            file_name="many.pdf",
            sha256="c" * 64,
            page_count=1,
        )
        text = "0123456789" * 12
        chunks = [
            Chunk(
                chunk_id=f"paper_999:r1:p1:c{index:04d}:v1:{index:08x}",
                paper_id="paper_999",
                page_index=0,
                page_number=1,
                ordinal=index,
                text=text[index * 10 : index * 10 + 10],
                start_offset=index * 10,
                end_offset=index * 10 + 10,
            )
            for index in range(6)
        ]
        pages = [Page(paper_id="paper_999", page_index=0, page_number=1, text=text)]
        storage.save_document(self.conn, paper, pages, chunks)

        with self.assertRaises(EvidenceError) as ctx:
            read_passages(
                self.conn, self.registry, [chunk.chunk_id for chunk in chunks]
            )
        self.assertEqual(ctx.exception.code, "BATCH_TOO_LARGE")

    def test_missing_chunk(self) -> None:
        with self.assertRaises(EvidenceError) as ctx:
            read_passages(self.conn, self.registry, ["missing:chunk"])
        self.assertEqual(ctx.exception.code, "CHUNK_NOT_FOUND")

    def test_out_of_scope(self) -> None:
        with self.assertRaises(EvidenceError) as ctx:
            read_passages(
                self.conn,
                self.registry,
                [self.chunk_id(0)],
                allowed_paper_ids={"paper_999"},
            )
        self.assertEqual(ctx.exception.code, "OUT_OF_SCOPE")

    def test_revision_mismatch(self) -> None:
        storage.save_paper(
            self.conn,
            Paper(
                id="paper_001",
                title="new revision",
                file_name="paper_001.pdf",
                sha256="d" * 64,
                revision=2,
                page_count=2,
            ),
        )
        with self.assertRaises(EvidenceError) as ctx:
            read_passages(self.conn, self.registry, [self.chunk_id(0)])
        self.assertEqual(ctx.exception.code, "REVISION_MISMATCH")

    def test_neighbor_context(self) -> None:
        paper = Paper(
            id="paper_777",
            title="neighbor",
            file_name="neighbor.pdf",
            sha256="e" * 64,
            page_count=1,
        )
        text = "alpha beta gamma"
        chunks = [
            Chunk(
                chunk_id="paper_777:r1:p1:c0000:v1:00000000",
                paper_id="paper_777",
                page_index=0,
                page_number=1,
                ordinal=0,
                text="alpha",
                start_offset=0,
                end_offset=5,
            ),
            Chunk(
                chunk_id="paper_777:r1:p1:c0001:v1:00000001",
                paper_id="paper_777",
                page_index=0,
                page_number=1,
                ordinal=1,
                text=" beta",
                start_offset=5,
                end_offset=10,
            ),
        ]
        pages = [Page(paper_id="paper_777", page_index=0, page_number=1, text=text)]
        storage.save_document(self.conn, paper, pages, chunks)

        passages = read_passages(
            self.conn,
            self.registry,
            [chunks[0].chunk_id],
            neighbor=1,
        )
        self.assertEqual(len(passages), 1)
        self.assertEqual([item.chunk_id for item in passages[0].neighbors], [chunks[1].chunk_id])
        # 相邻片段只是上下文，不登记为证据
        self.assertEqual(len(self.registry), 1)

    def test_bad_neighbor(self) -> None:
        with self.assertRaises(EvidenceError) as ctx:
            read_passages(
                self.conn, self.registry, [self.chunk_id(0)], neighbor=3
            )
        self.assertEqual(ctx.exception.code, "BAD_NEIGHBOR")


class ValidationTest(EvidenceTestBase):
    def test_valid_citation(self) -> None:
        passages = read_passages(self.conn, self.registry, [self.chunk_id(0)])
        evidence_id = passages[0].evidence.evidence_id
        self.assertTrue(
            is_valid_citation(self.registry, evidence_id, {"paper_001"})
        )
        records = assert_evidence_in_scope(
            self.registry, [evidence_id], {"paper_001"}
        )
        self.assertEqual(records[0].evidence_id, evidence_id)

    def test_unknown_evidence(self) -> None:
        self.assertFalse(is_valid_citation(self.registry, "ev_999"))
        with self.assertRaises(EvidenceError) as ctx:
            assert_evidence_in_scope(self.registry, ["ev_999"])
        self.assertEqual(ctx.exception.code, "UNKNOWN_EVIDENCE")

    def test_out_of_scope_citation(self) -> None:
        passages = read_passages(self.conn, self.registry, [self.chunk_id(0)])
        evidence_id = passages[0].evidence.evidence_id
        self.assertFalse(
            is_valid_citation(self.registry, evidence_id, {"paper_999"})
        )
        with self.assertRaises(EvidenceError) as ctx:
            assert_evidence_in_scope(
                self.registry, [evidence_id], {"paper_999"}
            )
        self.assertEqual(ctx.exception.code, "OUT_OF_SCOPE")


if __name__ == "__main__":
    unittest.main()
