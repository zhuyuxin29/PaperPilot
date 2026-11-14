from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from doc_factory import make_document
from paperpilot import storage
from paperpilot.config import Settings
from paperpilot.llm import EmbeddingError, HashEmbedding
from paperpilot.rag import indexing


class DummyProvider:
    def __init__(self, name: str, dimension: int) -> None:
        self.name = name
        self.dimension = dimension

    def embed(self, texts):
        return [[0.01 * (index + 1)] * self.dimension for index, _ in enumerate(texts)]


class FailingProvider:
    name = "failing"
    dimension = 8

    def embed(self, texts):
        raise EmbeddingError("boom", "EMBEDDING_NETWORK")


class IndexingTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = Settings(
            data_dir=Path(self._tmp.name) / "data", graph_mode="off"
        )
        self.settings.ensure_dirs()
        self.conn = storage.connect(self.settings.db_path)
        storage.ensure_schema(self.conn)
        self.provider = HashEmbedding(dimension=8)

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def save_document(
        self,
        paper_id: str = "paper_001",
        texts: tuple[str, ...] = ("alpha beta", "gamma delta"),
        revision: int = 1,
    ):
        paper, pages, chunks = make_document(
            paper_id=paper_id, texts=texts, revision=revision
        )
        storage.save_document(self.conn, paper, pages, chunks)
        return chunks


class BuildAndLoadTest(IndexingTestBase):
    def test_roundtrip(self) -> None:
        chunks = self.save_document()
        meta = indexing.build_paper_index(
            self.conn, self.settings, self.provider, "paper_001"
        )
        self.assertTrue(meta.path.is_file())
        self.assertEqual(meta.provider, "hash")
        self.assertEqual(meta.dimension, 8)
        self.assertEqual(meta.chunk_count, len(chunks))

        data = indexing.load_index(
            self.settings, "paper_001", 1, provider=self.provider
        )
        self.assertIsNotNone(data)
        self.assertEqual(list(data.chunk_ids), [chunk.chunk_id for chunk in chunks])
        self.assertEqual(data.vectors.shape, (len(chunks), 8))
        norms = np.linalg.norm(data.vectors, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_build_all_skips_zero_chunk_paper(self) -> None:
        self.save_document("paper_001")
        paper, pages, chunks = make_document(
            paper_id="paper_002", texts=(), sha256="b" * 64
        )
        storage.save_document(self.conn, paper, pages, chunks)
        metas = indexing.build_all_indexes(
            self.conn, self.settings, self.provider
        )
        self.assertEqual([meta.paper_id for meta in metas], ["paper_001"])

    def test_rebuild_is_idempotent(self) -> None:
        self.save_document()
        first = indexing.build_paper_index(
            self.conn, self.settings, self.provider, "paper_001"
        )
        second = indexing.build_paper_index(
            self.conn, self.settings, self.provider, "paper_001"
        )
        self.assertEqual(first.chunk_count, second.chunk_count)
        self.assertTrue(second.path.is_file())

        self.assertEqual(
            [p.name for p in self.settings.index_dir.iterdir() if ".tmp" in p.name],
            [],
        )

    def test_index_ready(self) -> None:
        self.save_document()
        self.assertFalse(
            indexing.index_ready(self.settings, "paper_001", 1, self.provider)
        )
        indexing.build_paper_index(
            self.conn, self.settings, self.provider, "paper_001"
        )
        self.assertTrue(
            indexing.index_ready(self.settings, "paper_001", 1, self.provider)
        )

    def test_stale_provider_is_rejected(self) -> None:
        self.save_document()
        indexing.build_paper_index(
            self.conn, self.settings, self.provider, "paper_001"
        )
        other = DummyProvider(name="other", dimension=8)
        with self.assertRaises(indexing.IndexBuildError) as ctx:
            indexing.load_index(self.settings, "paper_001", 1, provider=other)
        self.assertEqual(ctx.exception.code, "INDEX_STALE")


class BuildFailuresTest(IndexingTestBase):
    def test_missing_paper(self) -> None:
        with self.assertRaises(indexing.IndexBuildError) as ctx:
            indexing.build_paper_index(
                self.conn, self.settings, self.provider, "paper_404"
            )
        self.assertEqual(ctx.exception.code, "PAPER_NOT_FOUND")

    def test_no_chunks(self) -> None:
        paper, _, _ = make_document()
        storage.save_paper(self.conn, paper)
        with self.assertRaises(indexing.IndexBuildError) as ctx:
            indexing.build_paper_index(
                self.conn, self.settings, self.provider, "paper_001"
            )
        self.assertEqual(ctx.exception.code, "NO_CHUNKS")

    def test_revision_mismatch(self) -> None:
        self.save_document(revision=1)
        with self.assertRaises(indexing.IndexBuildError) as ctx:
            indexing.build_paper_index(
                self.conn, self.settings, self.provider, "paper_001", revision=2
            )
        self.assertEqual(ctx.exception.code, "REVISION_MISMATCH")

    def test_embedding_failure_leaves_no_index(self) -> None:
        self.save_document()
        with self.assertRaises(EmbeddingError):
            indexing.build_paper_index(
                self.conn, self.settings, FailingProvider(), "paper_001"
            )
        self.assertEqual(list(self.settings.index_dir.iterdir()), [])


class BuildAllTest(IndexingTestBase):
    def test_build_all_and_filter(self) -> None:
        self.save_document("paper_001")
        self.save_document("paper_002", texts=("one two",), revision=1)

        metas = indexing.build_all_indexes(
            self.conn, self.settings, self.provider
        )
        self.assertEqual(
            sorted(meta.paper_id for meta in metas), ["paper_001", "paper_002"]
        )

        only = indexing.build_all_indexes(
            self.conn, self.settings, self.provider, ["paper_002"]
        )
        self.assertEqual([meta.paper_id for meta in only], ["paper_002"])


if __name__ == "__main__":
    unittest.main()
