from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from doc_factory import make_document
from paperpilot import storage
from paperpilot.agent.policy import Budget
from paperpilot.agent.runner import run_agent
from paperpilot.config import Settings
from paperpilot.llm import ChatMessage, ChatResponse, HashEmbedding, ToolCall
from paperpilot.rag import indexing
from paperpilot.tools import build_registry


def text_response(text: str) -> ChatResponse:
    from agent_factory import final_json
    return ChatResponse(message=ChatMessage(role="assistant", content=final_json(text)))


def tool_response(*calls: ToolCall) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(role="assistant", content=None, tool_calls=tuple(calls))
    )


class ScriptedChat:
    name = "scripted"

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages, *, tools=None, timeout=None):
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("scripted chat exhausted")
        return self._responses.pop(0)


class AgentTestBase(unittest.TestCase):
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

        self.tools = build_registry()
        self.paper_ids = [self.paper.id]

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def run_agent(self, responses: list[ChatResponse], **kwargs):
        chat = ScriptedChat(responses)
        result = run_agent(
            question="论文的研究方法是什么？",
            paper_ids=self.paper_ids,
            conn=self.conn,
            settings=self.settings,
            embedding=self.provider,
            chat=chat,
            tools=self.tools,
            **kwargs,
        )
        return result, chat


class HappyPathTest(AgentTestBase):
    def test_search_read_then_answer(self) -> None:
        chunk_id = self.chunks[0].chunk_id

        result, chat = self.run_agent(
            [
                tool_response(
                    ToolCall("call_1", "search_chunks", {"query": "alpha beta", "top_k": 2})
                ),
                tool_response(
                    ToolCall("call_2", "read_passages", {"chunk_ids": [chunk_id]})
                ),
                text_response("论文的研究方法是 alpha beta。[ev_001]"),
            ]
        )

        self.assertEqual(result.status, "COMPLETED")
        self.assertIsNone(result.stop_reason)
        self.assertIn("[ev_001]", result.answer)
        self.assertEqual([c.evidence_id for c in result.citations], ["ev_001"])
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(len(result.trace), 2)
        self.assertEqual(
            [entry["tool"] for entry in result.trace], ["search_chunks", "read_passages"]
        )

        last_messages = chat.calls[-1]
        self.assertTrue(
            any(
                message.role == "tool" and message.tool_call_id == "call_2"
                for message in last_messages
            )
        )

    def test_answer_without_citation_is_allowed(self) -> None:
        result, _ = self.run_agent([text_response("当前文献中未找到相关证据。")])
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.citations, ())


class BudgetTest(AgentTestBase):
    def test_step_budget_stops(self) -> None:
        result, _ = self.run_agent(
            [
                tool_response(
                    ToolCall("call_1", "search_chunks", {"query": "alpha"})
                )
            ],
            budget=Budget(max_steps=1),
        )
        self.assertEqual(result.status, "STOPPED")
        self.assertEqual(result.stop_reason, "BUDGET_STEPS")
        self.assertEqual(result.steps, 1)

    def test_tool_call_budget_stops(self) -> None:
        result, _ = self.run_agent(
            [
                tool_response(
                    ToolCall("call_1", "search_chunks", {"query": "alpha"}),
                    ToolCall("call_2", "search_chunks", {"query": "beta"}),
                )
            ],
            budget=Budget(max_tool_calls=1),
        )
        self.assertEqual(result.status, "STOPPED")
        self.assertEqual(result.stop_reason, "BUDGET_TOOL_CALLS")
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(len(result.trace), 1)

    def test_repeated_call_stops(self) -> None:
        same_call = ToolCall("call_1", "search_chunks", {"query": "alpha"})
        result, _ = self.run_agent(
            [tool_response(same_call), tool_response(same_call)]
        )
        self.assertEqual(result.status, "STOPPED")
        self.assertEqual(result.stop_reason, "REPEATED_CALL")
        self.assertEqual(len(result.trace), 2)


class PromptRoutingTest(unittest.TestCase):
    def test_prompt_routes_relation_questions_to_graph(self) -> None:
        from paperpilot.agent.runner import SYSTEM_PROMPT

        self.assertIn("关系型", SYSTEM_PROMPT)
        self.assertIn("query_graph", SYSTEM_PROMPT)
        self.assertIn("search_chunks", SYSTEM_PROMPT)


class ValidationTest(AgentTestBase):
    def test_invalid_citation_is_repaired(self) -> None:
        result, _ = self.run_agent(
            [
                text_response("结论是……[ev_999]"),
                text_response("当前文献中未找到可支持该结论的证据。"),
            ]
        )
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.repair_attempts, 1)
        self.assertEqual(result.citations, ())
        self.assertTrue(
            any(
                entry["tool"] == "citation_repair"
                and entry["status"] == "ok"
                for entry in result.trace
            )
        )

    def test_invalid_citation_rejected_after_repair(self) -> None:
        result, _ = self.run_agent(
            [
                text_response("结论是……[ev_999]"),
                text_response("仍然引用[ev_888]"),
            ]
        )
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.stop_reason, "INVALID_CITATION")
        self.assertEqual(result.invalid_citations, ("ev_888",))
        self.assertEqual(result.repair_attempts, 1)
        self.assertEqual(result.citations, ())

    def test_repair_can_fix_citation_with_registered_evidence(self) -> None:
        chunk_id = self.chunks[0].chunk_id
        result, _ = self.run_agent(
            [
                tool_response(
                    ToolCall("call_1", "read_passages", {"chunk_ids": [chunk_id]})
                ),
                text_response("论文的研究方法是 alpha beta。[ev_999]"),
                text_response("论文的研究方法是 alpha beta。[ev_001]"),
            ]
        )
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual([c.evidence_id for c in result.citations], ["ev_001"])
        self.assertEqual(result.repair_attempts, 1)

    def test_unknown_tool_is_reported_to_model(self) -> None:
        result, _ = self.run_agent(
            [
                tool_response(ToolCall("call_1", "delete_database", {})),
                text_response("无法执行该操作。"),
            ]
        )
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.trace[0]["status"], "error")
        self.assertEqual(result.trace[0]["error"]["code"], "UNKNOWN_TOOL")
        self.assertEqual(result.tool_calls, 1)

    def test_empty_answer_is_failed(self) -> None:
        result, _ = self.run_agent([text_response("   ")])
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.stop_reason, "EMPTY_ANSWER")

    def test_cancel_event_stops_before_first_step(self) -> None:
        import threading

        cancel = threading.Event()
        cancel.set()
        result, _ = self.run_agent(
            [text_response("不应该被调用")], cancel_event=cancel
        )
        self.assertEqual(result.status, "STOPPED")
        self.assertEqual(result.stop_reason, "CANCELLED")
        self.assertEqual(result.steps, 0)


if __name__ == "__main__":
    unittest.main()
