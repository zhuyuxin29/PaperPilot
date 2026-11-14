from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

from doc_factory import make_document
from paperpilot.config import Settings
from paperpilot.llm import HashEmbedding
from paperpilot.rag import chroma_backend


class FakeCollection:
    def __init__(self) -> None:
        self.upserted = None
        self.deleted = None
        self.query_kwargs = None

    def delete(self, where):
        self.deleted = where

    def upsert(self, ids, embeddings, metadatas):
        self.upserted = (ids, embeddings, metadatas)

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {"ids": [["chunk_a", "chunk_b"]]}


class FakeClient:
    def __init__(self) -> None:
        self.collection = FakeCollection()

    def get_or_create_collection(self, name, metadata=None):
        return self.collection


class ChromaBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = Settings(
            data_dir=Path(self._tmp.name) / "data",
            vector_backend="chroma",
        )
        self.settings.ensure_dirs()
        self.provider = HashEmbedding(dimension=8)
        self.client = FakeClient()
        self._module = types.ModuleType("chromadb")
        self._module.PersistentClient = lambda **kwargs: self.client
        self._config = types.ModuleType("chromadb.config")

        class FakeSettings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        self._config.Settings = FakeSettings
        self._module.config = self._config
        self._saved = sys.modules.get("chromadb")
        sys.modules["chromadb"] = self._module
        sys.modules["chromadb.config"] = self._config

    def tearDown(self) -> None:
        if self._saved is None:
            sys.modules.pop("chromadb", None)
        else:
            sys.modules["chromadb"] = self._saved
        sys.modules.pop("chromadb.config", None)
        self._tmp.cleanup()

    def test_build_writes_metadata_and_filters(self) -> None:
        paper, pages, chunks = make_document()
        chroma_backend.build_chroma_index(self.settings, self.provider, chunks)
        ids, embeddings, metadatas = self.client.collection.upserted
        self.assertEqual(ids, [chunk.chunk_id for chunk in chunks])
        self.assertEqual(len(embeddings), len(chunks))
        self.assertEqual(metadatas[0]["paper_id"], paper.id)
        self.assertEqual(metadatas[0]["revision"], paper.revision)
        self.assertIn("paper_id", self.client.collection.deleted["$and"][0])

    def test_query_returns_chunk_ids_with_scope(self) -> None:
        ids = chroma_backend.query_chroma(
            self.settings,
            self.provider,
            "alpha",
            "paper_001",
            1,
            top_k=2,
        )
        self.assertEqual(ids, ["chunk_a", "chunk_b"])
        where = self.client.collection.query_kwargs["where"]["$and"]
        self.assertEqual(where[0], {"paper_id": "paper_001"})
        self.assertEqual(where[1], {"revision": 1})


if __name__ == "__main__":
    unittest.main()
