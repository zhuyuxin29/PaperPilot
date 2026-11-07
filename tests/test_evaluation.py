"""模块 12（evaluation）验收测试：指标、数据集、A/B 与 Agent 评测。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from doc_factory import make_document
from evaluation.dataset import DatasetError, load_dataset
from evaluation.metrics import (
    evidence_group_coverage,
    hit_at_k,
    looks_like_refusal,
    recall_at_k,
)
from evaluation.report import render_markdown
from evaluation.runner import run_agent_eval, run_retrieval_eval
from paperpilot import storage
from paperpilot.config import Settings
from paperpilot.llm import ChatMessage, ChatResponse, HashEmbedding
from paperpilot.rag import indexing


class MetricsTest(unittest.TestCase):
    def test_recall_hit_coverage(self) -> None:
        gold = ["a", "b"]
        retrieved = ["x", "a", "y"]
        self.assertEqual(recall_at_k(gold, retrieved, 3), 0.5)
        self.assertEqual(hit_at_k(gold, retrieved, 3), 1)
        self.assertIsNone(recall_at_k([], retrieved, 3))

        groups = [["a"], ["b", "z"]]
        self.assertEqual(evidence_group_coverage(groups, ["a", "x"], 2), 0.5)

    def test_refusal(self) -> None:
        self.assertTrue(looks_like_refusal("当前文献中未找到相关证据。"))
        self.assertFalse(looks_like_refusal("论文提出了 MethodX。"))


class DatasetTest(unittest.TestCase):
    def test_load_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "paper_ids": ["paper_001"],
                        "question": "问题",
                        "question_type": "fact",
                        "answerable": True,
                        "gold_chunk_ids": ["chunk_a"],
                        "gold_evidence_groups": [["chunk_a"]],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            questions = load_dataset(path)
            self.assertEqual(len(questions), 1)
            self.assertEqual(questions[0].gold_chunk_ids, ("chunk_a",))

    def test_invalid_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text('{"question_id": "q1"}\n', encoding="utf-8")
            with self.assertRaises(DatasetError):
                load_dataset(path)


class FakeGraph:
    def __init__(self, chunk_ids: list[str]) -> None:
        self.chunk_ids = chunk_ids

    def query_template(self, query_type, **kwargs):
        return [
            {
                "claim_id": f"clm_{index}",
                "relation_type": "APPLIED_IN",
                "subject": "a",
                "object": "b",
                "chunk_id": chunk_id,
                "page_number": 1,
                "paper_id": "paper_001",
            }
            for index, chunk_id in enumerate(self.chunk_ids)
        ]


class FakeChat:
    name = "fake"

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def chat(self, messages, **kwargs):
        return ChatResponse(
            message=ChatMessage(role="assistant", content=__import__("agent_factory").final_json(self.answer))
        )


class EvalTestBase(unittest.TestCase):
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

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()


class RetrievalEvalTest(EvalTestBase):
    def _questions(self, question: str):
        from evaluation.dataset import Question

        return [
            Question(
                question_id="q1",
                paper_ids=(self.paper.id,),
                question=question,
                question_type="fact",
                answerable=True,
                gold_chunk_ids=(self.chunks[0].chunk_id,),
                gold_evidence_groups=((self.chunks[0].chunk_id,),),
            )
        ]

    def test_config_a_recall(self) -> None:
        result = run_retrieval_eval(
            self.conn,
            self.settings,
            self.provider,
            self._questions("alpha beta"),
        )
        self.assertEqual(result["config"], "A")
        self.assertEqual(result["rows"][0]["recall_at_k"], 1.0)

    def test_config_b_graph_adds_candidate(self) -> None:
        graph = FakeGraph([self.chunks[0].chunk_id])
        result = run_retrieval_eval(
            self.conn,
            self.settings,
            self.provider,
            self._questions("zzz 没有匹配"),
            use_graph=True,
            graph_client=graph,
        )
        self.assertEqual(result["config"], "B")
        self.assertTrue(result["rows"][0]["graph_used"])
        self.assertEqual(result["rows"][0]["recall_at_k"], 1.0)

    def test_report_contains_metrics(self) -> None:
        result = run_retrieval_eval(
            self.conn,
            self.settings,
            self.provider,
            self._questions("alpha beta"),
        )
        report = render_markdown(result)
        self.assertIn("Recall@K", report)
        self.assertIn("q1", report)


class AgentEvalTest(EvalTestBase):
    def test_refusal_on_unanswerable(self) -> None:
        from evaluation.dataset import Question

        questions = [
            Question(
                question_id="q_unanswerable",
                paper_ids=(self.paper.id,),
                question="论文用了什么 GPU 集群？",
                question_type="unanswerable",
                answerable=False,
                gold_chunk_ids=(),
                gold_evidence_groups=(),
            )
        ]
        result = run_agent_eval(
            self.conn,
            self.settings,
            self.provider,
            FakeChat("当前文献中未找到相关证据。"),
            questions,
        )
        self.assertEqual(result["config"], "C")
        self.assertEqual(result["summary"]["refusal_rate_unanswerable"], 1.0)
        self.assertEqual(result["rows"][0]["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
