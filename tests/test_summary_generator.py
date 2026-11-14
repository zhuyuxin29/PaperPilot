from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from doc_factory import make_document
from paperpilot import storage
from paperpilot.config import Settings
from paperpilot.evidence.registry import EvidenceRegistry
from paperpilot.llm import ChatMessage, ChatResponse, HashEmbedding, ToolCall
from paperpilot.schemas import SectionStatus
from paperpilot.summary.generator import (
    SUMMARY_KEYS,
    SummaryError,
    current_summary_model,
    generate_summary,
    load_summary,
    summary_path,
    _select_chunks,
)
from paperpilot.summary.validation import SummaryValidationError
from paperpilot.tools import build_registry
from paperpilot.tools.registry import ToolContext


class FakeChat:
    name = "fake"
    model = "deepseek-chat"

    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return ChatResponse(
            message=ChatMessage(role="assistant", content=self._contents.pop(0))
        )


def summary_json(
    evidence_ids: list[str] | None = None,
    limitations_status: str = "insufficient_evidence",
) -> str:
    evidence_ids = evidence_ids or ["ev_001"]
    sections = {
        key: {
            "status": "insufficient_evidence",
            "items": [],
            "note": "材料未覆盖该类目",
        }
        for key in SUMMARY_KEYS
    }
    sections["background"] = {
        "status": "supported",
        "items": [
            {
                "text": "研究背景说明。",
                "statement_kind": "reported_by_paper",
                "evidence_ids": evidence_ids,
            }
        ],
    }
    sections["limitations"]["status"] = limitations_status
    if limitations_status == "not_found":
        sections["limitations"]["note"] = "论文未讨论局限性"
    return json.dumps({"sections": sections}, ensure_ascii=False)


class SelectionTest(unittest.TestCase):
    def test_stratified_selection_keeps_head_and_tail(self) -> None:
        _, _, chunks = make_document(
            texts=tuple(f"chunk-{index} " + "x" * 30 for index in range(60))
        )
        selected = _select_chunks(chunks, max_chars=120)

        self.assertIn(chunks[0], selected)
        self.assertIn(chunks[-1], selected)
        self.assertLessEqual(
            sum(len(chunk.text) for chunk in selected), 120
        )
        self.assertEqual(
            [chunk.ordinal for chunk in selected],
            sorted(chunk.ordinal for chunk in selected),
        )

    def test_small_document_keeps_all_chunks(self) -> None:
        _, _, chunks = make_document()
        self.assertEqual(_select_chunks(chunks, max_chars=100_000), chunks)


class SummaryTestBase(unittest.TestCase):
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

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def tool_context(self) -> ToolContext:
        return ToolContext(
            conn=self.conn,
            settings=self.settings,
            provider=HashEmbedding(dimension=8),
            evidence=EvidenceRegistry(),
            allowed_paper_ids=(self.paper.id,),
        )


class GenerateTest(SummaryTestBase):
    def test_generate_and_reuse_cache(self) -> None:
        chat = FakeChat([summary_json()])
        first = generate_summary(
            self.conn, self.settings, chat, self.paper.id
        )

        self.assertFalse(first.reused)
        self.assertTrue(first.path.is_file())
        self.assertIs(
            first.summary.sections["background"].status, SectionStatus.SUPPORTED
        )
        self.assertEqual(first.summary.coverage.status, "complete")
        self.assertEqual(first.summary.coverage.processed_ratio, 1.0)
        self.assertEqual(
            first.summary.evidence_map["ev_001"], self.chunks[0].chunk_id
        )
        self.assertGreaterEqual(len(first.summary.evidence_map), 1)
        self.assertEqual(chat.calls, 1)

        second = generate_summary(self.conn, self.settings, chat, self.paper.id)
        self.assertTrue(second.reused)
        self.assertEqual(chat.calls, 1)

    def test_not_found_downgraded_when_coverage_partial(self) -> None:
        chat = FakeChat([summary_json(limitations_status="not_found")])
        result = generate_summary(
            self.conn, self.settings, chat, self.paper.id, max_chars=5
        )

        self.assertEqual(result.summary.coverage.status, "partial")
        self.assertLess(result.summary.coverage.processed_ratio, 1.0)
        self.assertIs(
            result.summary.sections["limitations"].status,
            SectionStatus.INSUFFICIENT_EVIDENCE,
        )

    def test_retry_on_invalid_json(self) -> None:
        chat = FakeChat(["这不是 JSON", summary_json()])
        result = generate_summary(
            self.conn, self.settings, chat, self.paper.id
        )
        self.assertFalse(result.reused)
        self.assertEqual(chat.calls, 2)

    def test_invalid_evidence_is_rejected(self) -> None:
        chat = FakeChat([summary_json(evidence_ids=["ev_999"])])
        with self.assertRaises(SummaryValidationError):
            generate_summary(self.conn, self.settings, chat, self.paper.id)

    def test_stale_cache_is_detected(self) -> None:
        generate_summary(
            self.conn, self.settings, FakeChat([summary_json()]), self.paper.id
        )
        path = summary_path(
            self.settings,
            self.paper.id,
            self.paper.revision,
            current_summary_model(self.settings),
        )
        with self.assertRaises(SummaryError) as ctx:
            load_summary(
                self.settings,
                path,
                paper_id=self.paper.id,
                revision=self.paper.revision,
                model_id="another-model",
            )
        self.assertEqual(ctx.exception.code, "SUMMARY_STALE")


class FakeGraph:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = rows or []
        self.calls = 0

    def query_template(self, query_type, **kwargs):
        self.calls += 1
        return list(self.rows)


class SummaryGraphTest(SummaryTestBase):
    def _rows(self, page: int = 1) -> list[dict]:
        return [
            {
                "claim_id": "clm_1",
                "relation_type": "APPLIED_IN",
                "subject": "方法A",
                "object": "实验B",
                "chunk_id": self.chunks[0].chunk_id,
                "page_number": page,
                "paper_id": self.paper.id,
            }
        ]

    def test_graph_context_is_used_and_recorded(self) -> None:
        graph = FakeGraph(self._rows())
        result = generate_summary(
            self.conn,
            self.settings,
            FakeChat([summary_json()]),
            self.paper.id,
            graph=graph,
        )
        self.assertTrue(result.summary.graph_digest)
        self.assertEqual(graph.calls, 3)

    def test_graph_change_invalidates_cache(self) -> None:
        generate_summary(
            self.conn,
            self.settings,
            FakeChat([summary_json()]),
            self.paper.id,
            graph=FakeGraph(self._rows(1)),
        )
        chat = FakeChat([summary_json()])
        result = generate_summary(
            self.conn,
            self.settings,
            chat,
            self.paper.id,
            graph=FakeGraph(self._rows(2)),
        )
        self.assertFalse(result.reused)
        self.assertEqual(chat.calls, 1)

    def test_graph_failure_degrades(self) -> None:
        class BrokenGraph:
            def query_template(self, *args, **kwargs):
                raise RuntimeError("graph down")

        result = generate_summary(
            self.conn,
            self.settings,
            FakeChat([summary_json()]),
            self.paper.id,
            graph=BrokenGraph(),
        )
        self.assertEqual(result.summary.graph_digest, "")


class GetKeyPointsToolTest(SummaryTestBase):
    def test_not_generated(self) -> None:
        registry = build_registry()
        result = registry.execute(
            ToolCall("call_1", "get_key_points", {"paper_id": self.paper.id}),
            self.tool_context(),
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["status"], "not_generated")

    def test_valid_summary_is_returned(self) -> None:
        generate_summary(
            self.conn, self.settings, FakeChat([summary_json()]), self.paper.id
        )
        registry = build_registry()
        result = registry.execute(
            ToolCall("call_1", "get_key_points", {"paper_id": self.paper.id}),
            self.tool_context(),
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["status"], "valid")
        self.assertIn("sections", result.data["summary"])


if __name__ == "__main__":
    unittest.main()
