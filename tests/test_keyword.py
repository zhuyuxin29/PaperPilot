"""模块 12.5（keyword）验收测试：jieba 分词 + BM25。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from doc_factory import make_document
from paperpilot import storage
from paperpilot.config import Settings
from paperpilot.llm import HashEmbedding
from paperpilot.rag import indexing, keyword, retrieval


class KeywordTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = Settings(
            data_dir=Path(self._tmp.name) / "data", graph_mode="off"
        )
        self.settings.ensure_dirs()
        self.conn = storage.connect(self.settings.db_path)
        storage.ensure_schema(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def save(self, paper_id: str = "paper_001", texts=("alpha beta gamma",)):
        paper, pages, chunks = make_document(
            paper_id=paper_id,
            texts=texts,
            sha256=(paper_id[-1] * 64)[:64],
        )
        storage.save_document(self.conn, paper, pages, chunks)
        return chunks


class TokenizeTest(unittest.TestCase):
    def test_chinese_and_english(self) -> None:
        tokens = keyword.tokenize("RAG 检索增强生成 可以提升问答系统。")
        self.assertIn("rag", tokens)
        self.assertIn("检索增强生成", tokens)
        self.assertIn("问答", tokens)
        self.assertNotIn("。", tokens)

    def test_empty_text(self) -> None:
        self.assertEqual(keyword.tokenize("   ！？ "), [])


class Bm25Test(KeywordTestBase):
    def test_rare_term_ranks_first(self) -> None:
        chunks = self.save(
            texts=("alpha beta gamma", "alpha", "delta epsilon")
        )
        ids = keyword.keyword_search(
            self.conn, self.settings, "epsilon", ["paper_001"]
        )
        self.assertEqual(ids[0], chunks[2].chunk_id)

    def test_shorter_chunk_wins_with_same_frequency(self) -> None:
        chunks = self.save(texts=("alpha beta gamma", "alpha"))
        ids = keyword.keyword_search(
            self.conn, self.settings, "alpha", ["paper_001"]
        )
        self.assertEqual(ids[0], chunks[1].chunk_id)

    def test_chinese_query(self) -> None:
        chunks = self.save(
            texts=(
                "检索增强生成可以提升问答系统的准确性。",
                "本文讨论教学改革与课程评价。",
            )
        )
        ids = keyword.keyword_search(
            self.conn, self.settings, "检索增强生成", ["paper_001"]
        )
        self.assertEqual(ids[0], chunks[0].chunk_id)

    def test_lazy_index_build(self) -> None:
        self.save()
        path = keyword.index_path(self.settings, "paper_001", 1)
        self.assertFalse(path.is_file())
        keyword.keyword_search(self.conn, self.settings, "alpha", ["paper_001"])
        self.assertTrue(path.is_file())

    def test_scope_isolation(self) -> None:
        first = self.save(texts=("alpha epsilon",))
        second = self.save(
            paper_id="paper_002", texts=("epsilon zeta",)
        )
        ids = keyword.keyword_search(
            self.conn, self.settings, "epsilon", ["paper_001"]
        )
        self.assertIn(first[0].chunk_id, ids)
        self.assertNotIn(second[0].chunk_id, ids)

    def test_stale_tokenizer_version_rebuilds(self) -> None:
        self.save()
        path = keyword.index_path(self.settings, "paper_001", 1)
        path.write_text(
            json.dumps({"tokenizer_version": "old-version"}), encoding="utf-8"
        )
        self.assertIsNone(
            keyword.load_keyword_index(self.settings, "paper_001", 1)
        )
        ids = keyword.keyword_search(
            self.conn, self.settings, "alpha", ["paper_001"]
        )
        self.assertTrue(ids)

    def test_empty_query(self) -> None:
        self.save()
        self.assertEqual(
            keyword.keyword_search(self.conn, self.settings, "！！", ["paper_001"]),
            [],
        )


class RetrievalIntegrationTest(KeywordTestBase):
    def test_hybrid_uses_bm25_route(self) -> None:
        chunks = self.save(texts=("alpha beta gamma", "delta epsilon"))
        provider = HashEmbedding(dimension=16)
        indexing.build_paper_index(
            self.conn, self.settings, provider, "paper_001"
        )
        candidates = retrieval.search_chunks(
            self.conn,
            self.settings,
            "epsilon",
            ["paper_001"],
            provider=provider,
            mode="hybrid",
        )
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].chunk_id, chunks[1].chunk_id)
        self.assertIn("keyword", candidates[0].routes)


if __name__ == "__main__":
    unittest.main()
