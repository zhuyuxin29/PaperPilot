"""模块 14（API）验收测试：健康检查、文献、要点、问答与错误处理。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app, get_settings
from paperpilot import storage
from paperpilot.llm import (
    ChatMessage,
    ChatResponse,
    HashEmbedding,
    ToolCall,
)
from paperpilot.rag import indexing
from pdf_factory import build_pdf

CLEAN_PDF = build_pdf(["First page content.", "Second page content."])


def summary_json(evidence_id: str = "ev_001") -> str:
    sections = {
        key: {
            "status": "insufficient_evidence",
            "items": [],
            "note": "材料未覆盖",
        }
        for key in (
            "background",
            "research_question",
            "methods",
            "experiments",
            "conclusions",
            "contributions",
            "limitations",
        )
    }
    sections["background"] = {
        "status": "supported",
        "items": [
            {
                "text": "研究背景说明。",
                "statement_kind": "reported_by_paper",
                "evidence_ids": [evidence_id],
            }
        ],
    }
    return json.dumps({"sections": sections}, ensure_ascii=False)


class FakeChat:
    name = "fake"
    model = "fake-model"

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return ChatResponse(
            message=ChatMessage(
                role="assistant", content=self.contents.pop(0)
            )
        )


class ScriptedChat:
    name = "scripted"
    model = "fake-model"

    def __init__(self, responses) -> None:
        self.responses = list(responses)

    def chat(self, messages, **kwargs):
        return self.responses.pop(0)


def tool_response(call: ToolCall) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role="assistant", content=None, tool_calls=(call,))
    )


def text_response(text: str) -> ChatResponse:
    from agent_factory import final_json
    return ChatResponse(message=ChatMessage(role="assistant", content=final_json(text)))


class ApiTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = str(Path(self._tmp.name) / "data")
        self._env_backup = {
            "PAPERPILOT_DATA_DIR": os.environ.get("PAPERPILOT_DATA_DIR"),
            "GRAPH_MODE": os.environ.get("GRAPH_MODE"),
            "VECTOR_BACKEND": os.environ.get("VECTOR_BACKEND"),
        }
        os.environ["PAPERPILOT_DATA_DIR"] = self.data_dir
        os.environ["GRAPH_MODE"] = "off"
        os.environ["VECTOR_BACKEND"] = "numpy"
        get_settings.cache_clear()
        from unittest.mock import patch
        self._embedding_patch = patch("paperpilot.services.create_embedding_provider", return_value=HashEmbedding(dimension=16))
        self._embedding_patch.start()
        self.addCleanup(self._embedding_patch.stop)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()
        self._tmp.cleanup()

    def upload(self) -> dict:
        response = self.client.post(
            "/papers",
            files={"file": ("a.pdf", CLEAN_PDF, "application/pdf")},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()


class PaperApiTest(ApiTestBase):
    def test_health_and_empty_list(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(self.client.get("/papers").json(), [])

    def test_upload_list_pdf_and_delete(self) -> None:
        paper = self.upload()
        self.assertEqual(paper["id"], "paper_001")
        self.assertEqual(paper["page_count"], 2)
        self.assertGreater(paper["chunk_count"], 0)

        listed = self.client.get("/papers").json()
        self.assertEqual(len(listed), 1)

        pdf = self.client.get(f"/papers/{paper['id']}/pdf")
        self.assertEqual(pdf.status_code, 200)
        self.assertEqual(pdf.headers["content-type"], "application/pdf")

        deleted = self.client.delete(f"/papers/{paper['id']}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get("/papers").json(), [])

    def test_unknown_paper_returns_404(self) -> None:
        response = self.client.get("/papers/paper_999/pdf")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["code"], "PAPER_NOT_FOUND")

    def test_empty_upload_rejected(self) -> None:
        response = self.client.post(
            "/papers", files={"file": ("empty.pdf", b"", "application/pdf")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "EMPTY_FILE")


class SummaryApiTest(ApiTestBase):
    def test_summary_not_generated_then_generated(self) -> None:
        paper = self.upload()
        missing = self.client.get(f"/papers/{paper['id']}/summary")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(
            missing.json()["detail"]["code"], "SUMMARY_NOT_GENERATED"
        )

        import api.main as api_main

        original = api_main.create_chat_provider
        api_main.create_chat_provider = lambda settings: FakeChat(
            [summary_json()]
        )
        try:
            response = self.client.post(f"/papers/{paper['id']}/summary")
        finally:
            api_main.create_chat_provider = original

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(
            payload["sections"]["background"]["status"], "supported"
        )
        fetched = self.client.get(f"/papers/{paper['id']}/summary")
        self.assertEqual(fetched.status_code, 200)


class ChatApiTest(ApiTestBase):
    def _prepare_index(self, paper_id: str) -> None:
        settings = get_settings()
        conn = storage.connect(settings.db_path)
        storage.ensure_schema(conn)
        try:
            indexing.build_paper_index(
                conn, settings, HashEmbedding(dimension=16), paper_id
            )
        finally:
            conn.close()

    def test_chat_returns_citations_and_trace(self) -> None:
        paper = self.upload()
        self._prepare_index(paper["id"])
        settings = get_settings()
        conn = storage.connect(settings.db_path)
        chunk_id = storage.get_chunks(conn, paper["id"])[0].chunk_id
        conn.close()

        import api.main as api_main

        original_chat = api_main.create_chat_provider
        original_embedding = api_main.create_embedding_provider
        api_main.create_embedding_provider = lambda settings: HashEmbedding(
            dimension=16
        )
        api_main.create_chat_provider = lambda settings: ScriptedChat(
            [
                tool_response(
                    ToolCall("c1", "search_chunks", {"query": "alpha beta"})
                ),
                tool_response(
                    ToolCall("c2", "read_passages", {"chunk_ids": [chunk_id]})
                ),
                text_response("论文的研究方法是 alpha beta。[ev_001]"),
            ]
        )
        try:
            response = self.client.post(
                f"/papers/{paper['id']}/chat",
                json={"question": "论文的研究方法是什么？"},
            )
        finally:
            api_main.create_chat_provider = original_chat
            api_main.create_embedding_provider = original_embedding

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "COMPLETED")
        self.assertEqual(len(payload["citations"]), 1)
        self.assertEqual(payload["invalid_citations"], [])
        self.assertEqual(
            [entry["tool"] for entry in payload["trace"]],
            ["search_chunks", "read_passages"],
        )

    def test_chat_rejects_empty_question(self) -> None:
        paper = self.upload()
        response = self.client.post(
            f"/papers/{paper['id']}/chat", json={"question": ""}
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
