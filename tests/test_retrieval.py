"""模块 8（rag/retrieval）验收测试：范围隔离、融合、稳定性。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from doc_factory import make_document
from paperpilot import storage
from paperpilot.rag import indexing, retrieval


class DummyProvider:
    def __init__(self, name: str, dimension: int) -> None:
        self.name = name
        self.dimension = dimension

    def embed(self, texts):
        return [[0.0] * self.dimension for _ in texts]


class RetrievalTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        from paperpilot.config import Settings
        from paperpilot.llm import HashEmbedding

        self.settings = Settings(
            data_dir=Path(self._tmp.name) / "data", graph_mode="off"
        )
        self.settings.ensure_dirs()
        self.conn = storage.connect(self.settings.db_path)
        storage.ensure_schema(self.conn)
        self.provider = HashEmbedding(dimension=16)

        self._save("paper_001", ("alpha beta", "gamma delta"), "a" * 64)
        self._save("paper_002", ("epsilon zeta", "eta theta"), "b" * 64)
        for paper_id in ("paper_001", "paper_002"):
            indexing.build_paper_index(
                self.conn, self.settings, self.provider, paper_id
            )

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def _save(self, paper_id: str, texts: tuple[str, ...], sha: str) -> None:
        paper, pages, chunks = make_document(
            paper_id=paper_id, texts=texts, sha256=sha
        )
        storage.save_document(self.conn, paper, pages, chunks)

    def search(self, query, paper_ids, **kwargs):
        return retrieval.search_chunks(
            self.conn,
            self.settings,
            query,
            paper_ids,
            provider=self.provider,
            **kwargs,
        )


class ScopeTest(RetrievalTestBase):
    def test_scope_isolation(self) -> None:
        candidates = self.search("alpha beta", ["paper_001"])
        self.assertTrue(candidates)
        self.assertTrue(all(c.paper_id == "paper_001" for c in candidates))
        self.assertEqual(
            {c.paper_id for c in candidates}, {"paper_001"}
        )

    def test_other_paper_terms_do_not_leak(self) -> None:
        # 查询 B 论文的术语，但范围只有 A：应返回空
        self.assertEqual(self.search("epsilon zeta", ["paper_001"]), [])

    def test_keyword_route_finds_term(self) -> None:
        candidates = self.search("epsilon", ["paper_002"], mode="keyword")
        self.assertTrue(candidates)
        self.assertIn("epsilon", candidates[0].snippet)
        self.assertEqual(candidates[0].routes, ("keyword",))

    def test_vector_route_exact_match(self) -> None:
        candidates = self.search("alpha beta", ["paper_001"], mode="vector")
        self.assertTrue(candidates)
        self.assertIn("alpha", candidates[0].snippet)
        self.assertIn("vector", candidates[0].routes)

    def test_hybrid_has_both_routes(self) -> None:
        candidates = self.search("alpha beta", ["paper_001"], mode="hybrid")
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].routes, ("keyword", "vector"))

    def test_no_scope(self) -> None:
        with self.assertRaises(retrieval.RetrievalError) as ctx:
            self.search("alpha", ["paper_404"])
        self.assertEqual(ctx.exception.code, "NO_SCOPE")

    def test_deleted_paper_is_not_retrievable(self) -> None:
        storage.delete_paper(self.conn, "paper_001")
        with self.assertRaises(retrieval.RetrievalError) as ctx:
            self.search("alpha", ["paper_001"])
        self.assertEqual(ctx.exception.code, "NO_SCOPE")


class StabilityTest(RetrievalTestBase):
    def test_deterministic_order(self) -> None:
        first = [c.chunk_id for c in self.search("alpha", ["paper_001"])]
        second = [c.chunk_id for c in self.search("alpha", ["paper_001"])]
        self.assertEqual(first, second)

    def test_top_k(self) -> None:
        candidates = self.search("alpha", ["paper_001"], top_k=1)
        self.assertEqual(len(candidates), 1)

    def test_rank_starts_at_one(self) -> None:
        candidates = self.search("alpha", ["paper_001"])
        self.assertEqual([c.rank for c in candidates], list(range(1, len(candidates) + 1)))


class ErrorTest(RetrievalTestBase):
    def test_empty_query(self) -> None:
        with self.assertRaises(retrieval.RetrievalError) as ctx:
            self.search("   ", ["paper_001"])
        self.assertEqual(ctx.exception.code, "EMPTY_QUERY")

    def test_bad_top_k(self) -> None:
        with self.assertRaises(retrieval.RetrievalError) as ctx:
            self.search("alpha", ["paper_001"], top_k=0)
        self.assertEqual(ctx.exception.code, "BAD_TOP_K")

    def test_bad_mode(self) -> None:
        with self.assertRaises(retrieval.RetrievalError) as ctx:
            self.search("alpha", ["paper_001"], mode="fancy")
        self.assertEqual(ctx.exception.code, "BAD_MODE")

    def test_stale_index(self) -> None:
        other = DummyProvider(name="other", dimension=16)
        with self.assertRaises(retrieval.RetrievalError) as ctx:
            retrieval.search_chunks(
                self.conn,
                self.settings,
                "alpha",
                ["paper_001"],
                provider=other,
            )
        self.assertEqual(ctx.exception.code, "INDEX_STALE")


if __name__ == "__main__":
    unittest.main()
