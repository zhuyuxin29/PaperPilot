"""模块 3（storage）验收测试。"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from paperpilot import storage
from paperpilot.schemas import Chunk, Page, Paper, PaperState

UTC = timezone.utc
SHA = "a" * 64


def make_paper(paper_id: str = "paper_001", created_at: datetime | None = None) -> Paper:
    return Paper(
        id=paper_id,
        title="示例论文",
        file_name="demo.pdf",
        sha256=SHA,
        page_count=3,
        created_at=created_at or datetime(2026, 1, 1, tzinfo=UTC),
    )


def make_pages(paper_id: str = "paper_001") -> list[Page]:
    return [
        Page(paper_id=paper_id, page_index=0, page_number=1, text="第一页正文内容。"),
        Page(paper_id=paper_id, page_index=1, page_number=2, text="", is_blank=True),
        Page(paper_id=paper_id, page_index=2, page_number=3, text="第三页内容。"),
    ]


def make_chunks(paper_id: str = "paper_001") -> list[Chunk]:
    first = "第一页正文内容。"
    third = "第三页内容。"
    return [
        Chunk(
            chunk_id=f"{paper_id}:r1:p1:c0:abcd1234",
            paper_id=paper_id,
            page_index=0,
            page_number=1,
            ordinal=0,
            text=first,
            start_offset=0,
            end_offset=len(first),
        ),
        Chunk(
            chunk_id=f"{paper_id}:r1:p3:c1:abcd1234",
            paper_id=paper_id,
            page_index=2,
            page_number=3,
            ordinal=1,
            text=third,
            start_offset=0,
            end_offset=len(third),
        ),
    ]


class StorageTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "app.sqlite3"
        self.conn = storage.connect(self.db_path)
        storage.ensure_schema(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()


class SchemaTest(StorageTestBase):
    def test_schema_idempotent(self) -> None:
        storage.ensure_schema(self.conn)
        storage.ensure_schema(self.conn)
        self.assertEqual(storage.current_schema_version(self.conn), storage.SCHEMA_VERSION)

    def test_newer_schema_is_rejected(self) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                (str(storage.SCHEMA_VERSION + 1),),
            )
        with self.assertRaises(storage.SchemaVersionError):
            storage.ensure_schema(self.conn)

    def test_connect_to_directory_raises_storage_error(self) -> None:
        bad_path = Path(self._tmp.name) / "db_is_a_directory.sqlite3"
        bad_path.mkdir()
        with self.assertRaises(storage.StorageError) as ctx:
            storage.connect(bad_path)
        self.assertIn(str(bad_path), str(ctx.exception))


class DocumentTest(StorageTestBase):
    def test_roundtrip(self) -> None:
        paper = make_paper()
        pages = make_pages()
        chunks = make_chunks()
        storage.save_document(self.conn, paper, pages, chunks)

        self.assertEqual(storage.get_paper(self.conn, "paper_001"), paper)
        self.assertEqual(storage.get_pages(self.conn, "paper_001"), pages)
        self.assertEqual(storage.get_chunks(self.conn, "paper_001"), chunks)
        self.assertEqual(storage.get_chunk(self.conn, chunks[0].chunk_id), chunks[0])

    def test_save_is_idempotent(self) -> None:
        paper = make_paper()
        pages = make_pages()
        chunks = make_chunks()
        storage.save_document(self.conn, paper, pages, chunks)
        storage.save_document(self.conn, paper, pages, chunks)

        self.assertEqual(len(storage.get_pages(self.conn, "paper_001")), 3)
        self.assertEqual(len(storage.get_chunks(self.conn, "paper_001")), 2)
        self.assertEqual(len(storage.list_papers(self.conn)), 1)

    def test_atomic_rollback(self) -> None:
        paper = make_paper()
        pages = make_pages()
        chunks = make_chunks()
        # 第二段与第一段 ordinal 冲突（UNIQUE(paper_id, revision, ordinal)）
        chunks[1] = chunks[1].model_copy(update={"ordinal": 0})

        with self.assertRaises(sqlite3.IntegrityError):
            storage.save_document(self.conn, paper, pages, chunks)

        self.assertIsNone(storage.get_paper(self.conn, "paper_001"))
        self.assertEqual(storage.get_pages(self.conn, "paper_001"), [])

    def test_foreign_key_enforced(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            storage.save_pages(
                self.conn,
                [Page(paper_id="missing", page_index=0, page_number=1, text="x")],
            )

    def test_persistence_across_reopen(self) -> None:
        storage.save_document(self.conn, make_paper(), make_pages(), make_chunks())
        self.conn.close()

        self.conn = storage.connect(self.db_path)
        storage.ensure_schema(self.conn)
        self.assertEqual(
            storage.get_paper(self.conn, "paper_001").title, "示例论文"
        )
        self.assertEqual(len(storage.get_pages(self.conn, "paper_001")), 3)


class QueriesTest(StorageTestBase):
    def test_find_paper_by_sha256(self) -> None:
        storage.save_document(self.conn, make_paper())
        found = storage.find_paper_by_sha256(self.conn, SHA.upper())
        self.assertIsNotNone(found)
        self.assertEqual(found.id, "paper_001")
        self.assertIsNone(storage.find_paper_by_sha256(self.conn, "b" * 64))

    def test_list_papers_order(self) -> None:
        later = make_paper("paper_002", datetime(2026, 2, 1, tzinfo=UTC))
        earlier = make_paper("paper_001", datetime(2026, 1, 1, tzinfo=UTC))
        storage.save_paper(self.conn, later)
        storage.save_paper(self.conn, earlier)

        self.assertEqual(
            [paper.id for paper in storage.list_papers(self.conn)],
            ["paper_002", "paper_001"],
        )

    def test_update_paper_state(self) -> None:
        storage.save_paper(self.conn, make_paper())
        storage.update_paper_state(self.conn, "paper_001", PaperState.READY)
        self.assertIs(storage.get_paper(self.conn, "paper_001").state, PaperState.READY)

        with self.assertRaises(KeyError):
            storage.update_paper_state(self.conn, "missing", PaperState.READY)

    def test_delete_paper_cascades(self) -> None:
        storage.save_document(self.conn, make_paper(), make_pages(), make_chunks())
        self.assertTrue(storage.delete_paper(self.conn, "paper_001"))
        self.assertIsNone(storage.get_paper(self.conn, "paper_001"))
        self.assertEqual(storage.get_pages(self.conn, "paper_001"), [])
        self.assertEqual(storage.get_chunks(self.conn, "paper_001"), [])

    def test_revision_isolation(self) -> None:
        storage.save_document(self.conn, make_paper(), make_pages(), make_chunks())
        self.assertEqual(storage.get_pages(self.conn, "paper_001", revision=2), [])
        self.assertEqual(storage.get_chunks(self.conn, "paper_001", revision=2), [])


if __name__ == "__main__":
    unittest.main()
