"""Regression cases for the 2026-11-06 review; no paid APIs or live Neo4j."""
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_agent as agent_cases
import test_summary_generator as summary_cases
from pdf_factory import build_pdf
from paperpilot import services, storage
from paperpilot.agent.runner import run_agent
from paperpilot.agent.policy import Budget
from paperpilot.config import Settings
from paperpilot.llm import ChatMessage, ChatResponse, HashEmbedding, ToolCall
from paperpilot.rag import indexing, retrieval
from paperpilot.schemas import Page, PaperState, SectionStatus, SummaryCoverage
from paperpilot.tools import build_registry


def raw_response(content):
    return ChatResponse(ChatMessage(role="assistant", content=content))


class AgentBoundaries(agent_cases.AgentTestBase):
    def test_uncited_fact_cannot_be_published(self):
        result, _ = self.run_agent([
            raw_response("The paper proves 99.9% accuracy."),
            raw_response("The paper proves 99.9% accuracy."),
        ])
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.answer, "")

    def test_removing_citation_is_not_a_repair(self):
        result, _ = self.run_agent([
            agent_cases.text_response("Accuracy is 99%.[ev_999]"),
            agent_cases.text_response("Accuracy is 99%."),
        ])
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.answer, "")

    def test_invalid_citation_draft_is_not_published(self):
        result, _ = self.run_agent([
            agent_cases.text_response("Wrong[ev_999]"),
            agent_cases.text_response("Still wrong[ev_888]"),
        ])
        self.assertEqual(result.status, "FAILED")
        self.assertFalse(result.answer)

    def test_refusal_cannot_smuggle_fact_text(self):
        payload = json.dumps({"status": "insufficient_evidence", "claims": [], "text": "Accuracy is 99%"})
        result, _ = self.run_agent([raw_response(payload), raw_response(payload)])
        self.assertEqual(result.status, "FAILED")

    def test_cancel_during_model_discards_response(self):
        event = threading.Event()
        chat = Mock()
        def respond(*args, **kwargs):
            event.set()
            return raw_response('{"status":"insufficient_evidence","claims":[]}')
        chat.chat.side_effect = respond
        result = run_agent(question="q", paper_ids=self.paper_ids, conn=self.conn,
                           settings=self.settings, embedding=self.provider,
                           chat=chat, tools=self.tools, cancel_event=event)
        self.assertEqual(result.stop_reason, "CANCELLED")
        self.assertFalse(result.answer)

    def test_deadline_checked_after_model(self):
        clock = [0.0]
        chat = Mock()
        def respond(*args, **kwargs):
            self.assertLessEqual(kwargs["timeout"], 1)
            clock[0] = 2.0
            return raw_response('{"status":"insufficient_evidence","claims":[]}')
        chat.chat.side_effect = respond
        with patch("paperpilot.agent.runner.time.monotonic", side_effect=lambda: clock[0]):
            result = run_agent(question="q", paper_ids=self.paper_ids, conn=self.conn,
                               settings=self.settings, embedding=self.provider,
                               chat=chat, tools=self.tools, budget=Budget(max_seconds=1))
        self.assertEqual(result.stop_reason, "TIMEOUT")
        self.assertFalse(result.answer)

    def test_repeated_read_without_new_evidence_stops(self):
        call = ToolCall("read", "read_passages", {"chunk_ids": [self.chunks[0].chunk_id]})
        result, _ = self.run_agent([agent_cases.tool_response(call), agent_cases.tool_response(call)])
        self.assertEqual(result.stop_reason, "REPEATED_CALL")

    def test_finished_run_is_persisted(self):
        self.run_agent([raw_response('{"status":"insufficient_evidence","claims":[]}')])
        row = self.conn.execute("SELECT status,result_json FROM agent_runs").fetchone()
        self.assertEqual(row["status"], "COMPLETED")
        self.assertEqual(json.loads(row["result_json"])["answer_kind"], "insufficient_evidence")

    def test_finished_run_restores_chat_after_session_loss(self):
        from paperpilot.history import load_history
        result, _ = self.run_agent([raw_response('{"status":"insufficient_evidence","claims":[]}')])
        history = load_history(self.settings, self.paper.id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["run_id"], result.run_id)
        self.assertEqual(history[0]["answer"], result.answer)
        self.assertEqual(history[0]["revision"], 1)

    def test_dead_owner_marked_interrupted_but_live_owner_preserved(self):
        from paperpilot.agent import runs
        identifier = runs.start(self.conn, "q", self.paper_ids)
        with patch("paperpilot.agent.runs._owner_alive", return_value=True):
            runs.recover_orphans(self.conn)
        self.assertEqual(runs.recent(self.conn)[0]["status"], "RUNNING")
        with patch("paperpilot.agent.runs._owner_alive", return_value=False):
            runs.recover_orphans(self.conn)
        self.assertEqual(runs.recent(self.conn)[0]["status"], "INTERRUPTED")

    def test_failure_persists_last_completed_tool_trace(self):
        from paperpilot.llm import ChatError
        chat = agent_cases.ScriptedChat([agent_cases.tool_response(ToolCall("read", "read_passages", {"chunk_ids": [self.chunks[0].chunk_id]}))])
        first = chat.chat
        calls = [0]
        def respond(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise ChatError("simulated unavailable", "CHAT_NETWORK")
            return first(*args, **kwargs)
        chat.chat = respond
        with self.assertRaises(ChatError):
            run_agent(question="q", paper_ids=self.paper_ids, conn=self.conn, settings=self.settings,
                      embedding=self.provider, chat=chat, tools=self.tools)
        row = self.conn.execute("SELECT status,trace_json FROM agent_runs").fetchone()
        self.assertEqual(row["status"], "FAILED")
        self.assertEqual(json.loads(row["trace_json"])[0]["tool"], "read_passages")

    def test_missing_vector_index_is_explicit(self):
        indexing.index_path(self.settings, self.paper.id, self.paper.revision).unlink()
        with self.assertRaises(retrieval.RetrievalError) as ctx:
            retrieval.search_chunks(self.conn, self.settings, "alpha", self.paper_ids, provider=self.provider)
        self.assertEqual(ctx.exception.code, "INDEX_MISSING")


class PreparationBoundaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(data_dir=Path(self.tmp.name), graph_mode="off")
        self.settings.ensure_dirs()
        self.conn = storage.connect(self.settings.db_path)
        self.addCleanup(self.conn.close)
        storage.ensure_schema(self.conn)
        self.provider = HashEmbedding(dimension=16)
        self.pdf = build_pdf(["First page alpha beta.", "Second page gamma delta."])

    def prepare(self):
        return services.prepare_paper(self.pdf, "sample.pdf", self.settings, self.conn, provider=self.provider)

    def test_upload_builds_queryable_vector_index(self):
        paper = self.prepare().paper
        self.assertTrue(indexing.index_ready(self.settings, paper.id, 1, self.provider))
        hits = retrieval.search_chunks(self.conn, self.settings, "alpha beta", [paper.id], provider=self.provider)
        self.assertTrue(any("vector" in hit.routes for hit in hits))

    def test_index_failure_keeps_document_for_retry(self):
        bad = Mock()
        bad.embed.side_effect = RuntimeError("temporary embedding failure")
        with self.assertRaises(RuntimeError):
            services.prepare_paper(self.pdf, "sample.pdf", self.settings, self.conn, provider=bad)
        paper = storage.list_papers(self.conn)[0]
        self.assertEqual(paper.state, PaperState.PARTIAL)
        self.assertTrue((self.settings.paper_dir(paper.id) / "source.pdf").is_file())
        retried = self.prepare()
        self.assertTrue(retried.duplicated)
        self.assertEqual(retried.paper.state, PaperState.READY)

    def test_reprocess_parse_failure_keeps_previous_revision(self):
        paper = self.prepare().paper
        old = storage.get_chunks(self.conn, paper.id)
        with patch("paperpilot.services.parser.parse_pdf", side_effect=RuntimeError("parse")):
            with self.assertRaises(RuntimeError):
                services.reprocess_paper(self.settings, self.conn, paper.id, provider=self.provider)
        self.assertEqual(storage.get_paper(self.conn, paper.id), paper)
        self.assertEqual(storage.get_chunks(self.conn, paper.id), old)
        self.assertEqual((self.settings.paper_dir(paper.id) / "source.pdf").read_bytes(), self.pdf)

    def test_reprocess_index_failure_keeps_previous_revision(self):
        paper = self.prepare().paper
        with patch("paperpilot.services.indexing.build_paper_index", side_effect=RuntimeError("embedding")):
            with self.assertRaises(RuntimeError):
                services.reprocess_paper(self.settings, self.conn, paper.id, provider=self.provider)
        self.assertEqual(storage.get_paper(self.conn, paper.id).revision, 1)
        self.assertTrue(indexing.index_ready(self.settings, paper.id, 1, self.provider))

    def test_reprocess_success_preserves_id_and_old_chunks(self):
        paper = self.prepare().paper
        result = services.reprocess_paper(self.settings, self.conn, paper.id, provider=self.provider)
        self.assertEqual(result.paper.id, paper.id)
        self.assertEqual(result.paper.revision, 2)
        self.assertTrue(storage.get_chunks(self.conn, paper.id, 1))
        self.assertTrue(storage.get_chunks(self.conn, paper.id, 2))
        self.assertTrue(indexing.index_ready(self.settings, paper.id, 2, self.provider))

    def test_database_publish_failure_keeps_old_revision(self):
        paper = self.prepare().paper
        original = storage.save_document
        def save(conn, *args):
            if conn is self.conn:
                raise sqlite3.OperationalError("simulated disk failure")
            return original(conn, *args)
        with patch("paperpilot.services.storage.save_document", side_effect=save):
            with self.assertRaises(sqlite3.OperationalError):
                services.reprocess_paper(self.settings, self.conn, paper.id, provider=self.provider)
        self.assertEqual(storage.get_paper(self.conn, paper.id).revision, 1)
        self.assertTrue(indexing.index_ready(self.settings, paper.id, 1, self.provider))


class SummaryBoundaries(summary_cases.SummaryTestBase):
    def test_unparsed_page_prevents_full_coverage(self):
        blank = Page(paper_id=self.paper.id, page_index=2, page_number=3, text="", is_blank=True, quality_flags=("extract_error",))
        storage.save_pages(self.conn, [blank])
        result = summary_cases.generate_summary(self.conn, self.settings,
            summary_cases.FakeChat([summary_cases.summary_json(limitations_status="not_found")]), self.paper.id)
        self.assertEqual(result.summary.coverage.status, "partial")
        self.assertEqual(result.summary.coverage.unparsed_pages, (3,))
        self.assertEqual(result.summary.sections["limitations"].status, SectionStatus.INSUFFICIENT_EVIDENCE)

    def test_budget_omissions_are_listed(self):
        result = summary_cases.generate_summary(self.conn, self.settings,
            summary_cases.FakeChat([summary_cases.summary_json()]), self.paper.id, max_chars=1)
        self.assertEqual(result.summary.coverage.unprocessed_units, (self.chunks[1].chunk_id,))

    def test_graph_summary_readable_without_old_evidence_ids(self):
        graph = summary_cases.FakeGraph([{"subject":"a", "object":"b", "relation_type":"APPLIED_IN", "page_number":1}])
        summary_cases.generate_summary(self.conn, self.settings,
            summary_cases.FakeChat([summary_cases.summary_json()]), self.paper.id, graph=graph)
        result = build_registry().execute(ToolCall("get", "get_key_points", {"paper_id":self.paper.id}), self.tool_context())
        self.assertEqual(result.data["status"], "valid")
        item = result.data["summary"]["sections"]["background"]["items"][0]
        self.assertNotIn("evidence_ids", item)
        self.assertEqual(item["source_chunk_ids"], [self.chunks[0].chunk_id])


class CapabilityBoundaries(unittest.TestCase):
    def test_graph_absent_from_disabled_registry(self):
        self.assertNotIn("query_graph", build_registry().names())
        self.assertIn("query_graph", build_registry(graph_enabled=True).names())

    def test_graph_on_does_not_silently_downgrade(self):
        from api.main import _graph_client
        from fastapi import HTTPException
        from paperpilot.graph.client import GraphError
        settings = Settings(data_dir=Path("unused"), graph_mode="on")
        with patch("api.main.Neo4jClient.connect", side_effect=GraphError("unavailable")):
            with self.assertRaises(HTTPException) as ctx:
                _graph_client(settings)
            self.assertEqual(ctx.exception.status_code, 503)
            settings.graph_mode = "auto"
            self.assertIsNone(_graph_client(settings))

    def test_duplicate_pending_does_not_start_new_thread(self):
        import app
        pending = {"question":"old"}
        fake_st = SimpleNamespace(session_state={"pending":pending}, warning=Mock())
        with patch.object(app, "st", fake_st), patch.object(app.threading, "Thread") as thread:
            app._start_pending(Settings(data_dir=Path("unused")), "paper_001", "new")
        thread.assert_not_called()
        self.assertIs(fake_st.session_state["pending"], pending)

    def test_graph_eval_passes_actual_question(self):
        from evaluation.runner import _graph_chunk_ids
        graph = Mock()
        graph.query_template.return_value = []
        _graph_chunk_ids(graph, ["paper_001"], "Which experiment used MethodA?")
        self.assertTrue(all(call.kwargs["question"] == "Which experiment used MethodA?" for call in graph.query_template.call_args_list))


if __name__ == "__main__":
    unittest.main()
