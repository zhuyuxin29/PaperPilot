"""多轮对话改写测试。"""

from __future__ import annotations

import unittest

from paperpilot.conversation import rewrite_followup
from paperpilot.llm import ChatMessage, ChatResponse


class FakeChat:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return ChatResponse(
            message=ChatMessage(role="assistant", content=self.content)
        )


class RewriteTest(unittest.TestCase):
    def test_rewrites_with_history(self) -> None:
        chat = FakeChat("这篇论文的实验设计是什么？")
        result = rewrite_followup(
            chat,
            "那它的实验呢？",
            [
                {
                    "question": "论文提出了什么方法？",
                    "answer": "论文提出了方法 A。",
                }
            ],
        )
        self.assertEqual(result, "这篇论文的实验设计是什么？")
        self.assertEqual(chat.calls, 1)

    def test_without_history_returns_question(self) -> None:
        chat = FakeChat("不应被调用")
        self.assertEqual(rewrite_followup(chat, "什么是 RAG？", []), "什么是 RAG？")
        self.assertEqual(chat.calls, 0)

    def test_failure_falls_back(self) -> None:
        class Broken:
            def chat(self, messages, **kwargs):
                raise RuntimeError("network")

        self.assertEqual(
            rewrite_followup(Broken(), "那它呢？", [{"question": "q"}]),
            "那它呢？",
        )


if __name__ == "__main__":
    unittest.main()
