from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from doc_factory import make_document
from paperpilot import storage
from paperpilot.config import Settings
from paperpilot.evidence.registry import EvidenceRegistry
from paperpilot.llm import HashEmbedding, ToolCall
from paperpilot.rag import indexing
from paperpilot.tools import build_registry
from paperpilot.tools.registry import ToolContext


class ToolsTestBase(unittest.TestCase):
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
        self.provider = HashEmbedding(dimension=16)
        indexing.build_paper_index(
            self.conn, self.settings, self.provider, self.paper.id
        )

        self.registry = build_registry(graph_enabled=True)
        self.context = ToolContext(
            conn=self.conn,
            settings=self.settings,
            provider=self.provider,
            evidence=EvidenceRegistry(),
            allowed_paper_ids=(self.paper.id,),
        )

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()


class RegistryTest(ToolsTestBase):
    def test_registered_tools(self) -> None:
        self.assertEqual(
            self.registry.names(),
            ("get_key_points", "query_graph", "read_passages", "search_chunks"),
        )
        schema_names = {
            item["function"]["name"] for item in self.registry.openai_tools()
        }
        self.assertEqual(
            schema_names,
            {"search_chunks", "read_passages", "get_key_points", "query_graph"},
        )

    def test_unknown_tool(self) -> None:
        result = self.registry.execute(
            ToolCall("call_1", "delete_paper", {}), self.context
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error["code"], "UNKNOWN_TOOL")
        self.assertIsNone(result.data)

    def test_bad_arguments(self) -> None:
        result = self.registry.execute(
            ToolCall("call_1", "search_chunks", {"top_k": 3}), self.context
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error["code"], "BAD_ARGUMENTS")
        self.assertTrue(result.error["retryable"])

    def test_result_envelope_fields(self) -> None:
        result = self.registry.execute(
            ToolCall("call_1", "search_chunks", {"query": "alpha"}),
            self.context,
        )
        self.assertEqual(result.tool_call_id, "call_1")
        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.error)
        self.assertIsInstance(result.duration_ms, int)
        payload = result.as_tool_message()
        self.assertEqual(payload["tool_call_id"], "call_1")
        self.assertEqual(payload["status"], "ok")


class SearchToolTest(ToolsTestBase):
    def test_search_returns_candidates_without_evidence(self) -> None:
        result = self.registry.execute(
            ToolCall("call_1", "search_chunks", {"query": "alpha beta", "top_k": 2}),
            self.context,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.evidence_ids, ())
        candidates = result.data["candidates"]
        self.assertTrue(candidates)
        self.assertIn("chunk_id", candidates[0])
        self.assertIn("page_number", candidates[0])
        self.assertTrue(all(c["paper_id"] == "paper_001" for c in candidates))

    def test_search_empty_result_is_ok(self) -> None:
        result = self.registry.execute(
            ToolCall("call_1", "search_chunks", {"query": "not-in-this-paper"}),
            self.context,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["candidates"], [])


class ReadToolTest(ToolsTestBase):
    def test_read_registers_evidence(self) -> None:
        chunk_id = self.chunks[0].chunk_id
        result = self.registry.execute(
            ToolCall("call_1", "read_passages", {"chunk_ids": [chunk_id]}),
            self.context,
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.evidence_ids, ("ev_001",))
        passage = result.data["passages"][0]
        self.assertEqual(passage["chunk_id"], chunk_id)
        self.assertEqual(passage["evidence_id"], "ev_001")
        self.assertEqual(passage["page_number"], 1)

    def test_read_out_of_scope(self) -> None:
        other = make_document(paper_id="paper_002", sha256="b" * 64)
        storage.save_document(self.conn, other[0], other[1], other[2])
        result = self.registry.execute(
            ToolCall(
                "call_1", "read_passages", {"chunk_ids": [other[2][0].chunk_id]}
            ),
            self.context,
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error["code"], "OUT_OF_SCOPE")

    def test_read_missing_chunk(self) -> None:
        result = self.registry.execute(
            ToolCall("call_1", "read_passages", {"chunk_ids": ["missing"]}),
            self.context,
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error["code"], "CHUNK_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
