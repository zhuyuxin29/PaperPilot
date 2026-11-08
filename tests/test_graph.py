"""模块 11（graph）验收测试：抽取、构建、工具（使用 Fake，不连真实 Neo4j）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from doc_factory import make_document
from paperpilot import storage
from paperpilot.config import Settings
from paperpilot.evidence.registry import EvidenceRegistry
from paperpilot.graph import extractor
from paperpilot.graph.builder import build_paper_graph, entity_id
from paperpilot.graph.client import GraphError, WORKSPACE_ID
from paperpilot.llm import ChatMessage, ChatResponse, HashEmbedding, ToolCall
from paperpilot.tools import build_registry
from paperpilot.tools.registry import ToolContext


class FakeChat:
    name = "fake"
    model = "deepseek-chat"

    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)

    def chat(self, messages, **kwargs):
        return ChatResponse(
            message=ChatMessage(role="assistant", content=self._contents.pop(0))
        )


class FakeClient:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.claims: list[dict] = []
        self.schema_called = False
        self.rows = rows or []
        self.query_calls: list[dict] = []
        self.cleared: tuple[str, str] | None = None

    def ensure_schema(self) -> None:
        self.schema_called = True

    def clear_paper_graph(self, workspace_id: str, paper_id: str) -> None:
        self.cleared = (workspace_id, paper_id)

    def write_claim(self, claim: dict) -> None:
        self.claims.append(claim)

    def query_template(self, query_type, **kwargs):
        self.query_calls.append({"query_type": query_type, **kwargs})
        return list(self.rows)

    def close(self) -> None:
        pass


def claims_json(claims: list[dict]) -> str:
    return json.dumps({"claims": claims}, ensure_ascii=False)


class GraphTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = Settings(
            data_dir=Path(self._tmp.name) / "data", graph_mode="auto"
        )
        self.settings.ensure_dirs()
        self.conn = storage.connect(self.settings.db_path)
        storage.ensure_schema(self.conn)

        self.paper, self.pages, self.chunks = make_document(
            texts=(
                "We use MethodA and apply it to ExperimentB.",
                "This chunk has no matching sentence.",
            )
        )
        storage.save_document(self.conn, self.paper, self.pages, self.chunks)

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def tool_context(self, client) -> ToolContext:
        return ToolContext(
            conn=self.conn,
            settings=self.settings,
            provider=HashEmbedding(dimension=8),
            evidence=EvidenceRegistry(),
            allowed_paper_ids=(self.paper.id,),
            graph=client,
        )


class ExtractorTest(GraphTestBase):
    def test_extracts_and_checks_alignment(self) -> None:
        payload = claims_json(
            [
                {
                    "chunk_id": self.chunks[0].chunk_id,
                    "relation_type": "APPLIED_IN",
                    "subject": {"type": "Method", "name": "MethodA"},
                    "object": {"type": "Experiment", "name": "ExperimentB"},
                    "evidence_text": "use MethodA and apply it to ExperimentB",
                },
                {
                    "chunk_id": self.chunks[1].chunk_id,
                    "relation_type": "PRODUCES",
                    "subject": {"type": "Experiment", "name": "ExperimentX"},
                    "object": {"type": "Result", "name": "ResultY"},
                    "evidence_text": "这段原文并不存在",
                },
            ]
        )
        claims = extractor.extract_claims(FakeChat([payload]), self.chunks)

        self.assertEqual(len(claims), 2)
        self.assertTrue(claims[0].evidence_aligned)
        self.assertFalse(claims[1].evidence_aligned)

    def test_invalid_relation_is_skipped(self) -> None:
        payload = claims_json(
            [
                {
                    "chunk_id": self.chunks[0].chunk_id,
                    "relation_type": "CAUSES_EVERYTHING",
                    "subject": {"type": "Method", "name": "A"},
                    "object": {"type": "Result", "name": "B"},
                    "evidence_text": "We use MethodA",
                }
            ]
        )
        self.assertEqual(
            extractor.extract_claims(FakeChat([payload]), self.chunks), []
        )

    def test_bad_json(self) -> None:
        with self.assertRaises(GraphError) as ctx:
            extractor.extract_claims(FakeChat(["not json"]), self.chunks)
        self.assertEqual(ctx.exception.code, "GRAPH_BAD_EXTRACTION")


class BuilderTest(GraphTestBase):
    def _claims_payload(self) -> str:
        return claims_json(
            [
                {
                    "chunk_id": self.chunks[0].chunk_id,
                    "relation_type": "APPLIED_IN",
                    "subject": {"type": "Method", "name": "MethodA"},
                    "object": {"type": "Experiment", "name": "ExperimentB"},
                    "evidence_text": "use MethodA and apply it to ExperimentB",
                },
                {
                    "chunk_id": self.chunks[1].chunk_id,
                    "relation_type": "PRODUCES",
                    "subject": {"type": "Experiment", "name": "ExperimentX"},
                    "object": {"type": "Result", "name": "ResultY"},
                    "evidence_text": "不存在",
                },
            ]
        )

    def test_build_records_state_and_claims(self) -> None:
        client = FakeClient()
        result = build_paper_graph(
            self.conn,
            self.settings,
            FakeChat([self._claims_payload()]),
            client,
            self.paper.id,
        )

        self.assertTrue(client.schema_called)
        self.assertEqual(client.cleared, (WORKSPACE_ID, self.paper.id))
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.claims_total, 2)
        self.assertEqual(result.claims_validated, 1)
        self.assertEqual(result.claims_pending, 1)
        self.assertEqual(len(client.claims), 2)
        statuses = {claim["status"] for claim in client.claims}
        self.assertEqual(statuses, {"validated", "pending_review"})

        state = storage.get_graph_state(self.conn, self.paper.id)
        self.assertIsNotNone(state)
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["claim_count"], 2)

    def test_claim_and_entity_ids_are_stable(self) -> None:
        first = entity_id(WORKSPACE_ID, "Method", "MethodA")
        second = entity_id(WORKSPACE_ID, "Method", "  methoda ")
        self.assertEqual(first, second)

    def test_rebuild_reuses_extraction_cache(self) -> None:
        # 第一次构建写入抽取缓存；第二次用没有内容的 FakeChat 重建也不再调模型。
        build_paper_graph(
            self.conn,
            self.settings,
            FakeChat([self._claims_payload()]),
            FakeClient(),
            self.paper.id,
        )
        client = FakeClient()
        result = build_paper_graph(
            self.conn,
            self.settings,
            FakeChat([]),
            client,
            self.paper.id,
        )
        self.assertEqual(result.claims_total, 2)
        self.assertEqual(len(client.claims), 2)


class QueryGraphToolTest(GraphTestBase):
    def test_tool_description_targets_relation_questions(self) -> None:
        from paperpilot.tools.query_graph import SPEC

        self.assertIn("关系型", SPEC.description)
        self.assertIn("search_chunks", SPEC.description)

    def test_query_returns_claims(self) -> None:
        rows = [
            {
                "claim_id": "clm_1",
                "relation_type": "APPLIED_IN",
                "subject": "MethodA",
                "object": "ExperimentB",
                "chunk_id": self.chunks[0].chunk_id,
                "page_number": 1,
                "paper_id": self.paper.id,
            }
        ]
        client = FakeClient(rows=rows)
        storage.save_graph_state(self.conn, self.paper.id, self.paper.revision, "ready")
        registry = build_registry(graph_enabled=True)
        result = registry.execute(
            ToolCall(
                "call_1",
                "query_graph",
                {"query_type": "method_experiments"},
            ),
            self.tool_context(client),
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.data["claims"]), 1)
        self.assertEqual(result.evidence_ids, ())
        self.assertEqual(client.query_calls[0]["paper_ids"], [self.paper.id])

    def test_graph_unavailable(self) -> None:
        registry = build_registry(graph_enabled=True)
        result = registry.execute(
            ToolCall(
                "call_1", "query_graph", {"query_type": "method_experiments"}
            ),
            self.tool_context(None),
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error["code"], "GRAPH_UNAVAILABLE")

    def test_graph_disabled(self) -> None:
        self.settings.graph_mode = "off"
        registry = build_registry(graph_enabled=True)
        result = registry.execute(
            ToolCall(
                "call_1", "query_graph", {"query_type": "method_experiments"}
            ),
            self.tool_context(FakeClient()),
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error["code"], "GRAPH_DISABLED")

    def test_bad_query_type(self) -> None:
        registry = build_registry(graph_enabled=True)
        result = registry.execute(
            ToolCall("call_1", "query_graph", {"query_type": "anything"}),
            self.tool_context(FakeClient()),
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error["code"], "BAD_ARGUMENTS")


if __name__ == "__main__":
    unittest.main()
