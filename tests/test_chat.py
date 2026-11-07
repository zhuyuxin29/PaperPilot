"""模块 9（llm/chat）验收测试：对话模型解析与工厂。"""

from __future__ import annotations

import unittest
from pathlib import Path

import requests

from paperpilot.config import Settings
from paperpilot.llm import (
    ChatError,
    ChatMessage,
    OpenAICompatibleChat,
    ToolCall,
    create_chat_provider,
)


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def assistant_message_payload(content=None, tool_calls=None):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            }
        ],
        "model": "deepseek-chat",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class ParseTest(unittest.TestCase):
    def _provider(self, post):
        return OpenAICompatibleChat(
            api_key="test-key",
            model="deepseek-chat",
            base_url="https://api.example.com/v1",
            post=post,
        )

    def test_content_and_usage(self) -> None:
        calls = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            calls.update(url=url, payload=json, headers=headers)
            return FakeResponse(payload=assistant_message_payload("你好"))

        response = self._provider(fake_post).chat(
            [ChatMessage(role="user", content="hi")]
        )

        self.assertEqual(response.message.content, "你好")
        self.assertEqual(response.message.tool_calls, ())
        self.assertEqual(response.usage["total_tokens"], 15)
        self.assertEqual(calls["url"], "https://api.example.com/v1/chat/completions")
        self.assertEqual(calls["payload"]["messages"][0]["content"], "hi")

    def test_tool_calls_parsed(self) -> None:
        def fake_post(*args, **kwargs):
            return FakeResponse(
                payload=assistant_message_payload(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_01",
                            "type": "function",
                            "function": {
                                "name": "search_chunks",
                                "arguments": '{"query": "alpha", "top_k": 3}',
                            },
                        }
                    ],
                )
            )

        response = self._provider(fake_post).chat(
            [ChatMessage(role="user", content="hi")]
        )

        self.assertEqual(len(response.message.tool_calls), 1)
        call = response.message.tool_calls[0]
        self.assertEqual(call.id, "call_01")
        self.assertEqual(call.name, "search_chunks")
        self.assertEqual(call.arguments, {"query": "alpha", "top_k": 3})

    def test_serializes_tool_result_messages(self) -> None:
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured.update(json)
            return FakeResponse(payload=assistant_message_payload("ok"))

        messages = [
            ChatMessage(role="assistant", tool_calls=(ToolCall("c1", "search_chunks", {"query": "x"}),)),
            ChatMessage(role="tool", tool_call_id="c1", name="search_chunks", content="{}"),
        ]
        self._provider(fake_post).chat(messages)

        serialized = captured["messages"]
        self.assertEqual(serialized[0]["tool_calls"][0]["id"], "c1")
        self.assertEqual(serialized[1]["tool_call_id"], "c1")

    def test_http_error(self) -> None:
        def fake_post(*args, **kwargs):
            return FakeResponse(status_code=401, text="unauthorized")

        with self.assertRaises(ChatError) as ctx:
            self._provider(fake_post).chat([ChatMessage(role="user", content="hi")])
        self.assertEqual(ctx.exception.code, "CHAT_HTTP")

    def test_network_error(self) -> None:
        def fake_post(*args, **kwargs):
            raise requests.ConnectionError("boom")

        with self.assertRaises(ChatError) as ctx:
            self._provider(fake_post).chat([ChatMessage(role="user", content="hi")])
        self.assertEqual(ctx.exception.code, "CHAT_NETWORK")

    def test_bad_json(self) -> None:
        def fake_post(*args, **kwargs):
            return FakeResponse(payload=None)

        with self.assertRaises(ChatError) as ctx:
            self._provider(fake_post).chat([ChatMessage(role="user", content="hi")])
        self.assertEqual(ctx.exception.code, "CHAT_BAD_RESPONSE")


class FactoryTest(unittest.TestCase):
    def test_deepseek_defaults(self) -> None:
        settings = Settings(
            data_dir=Path("."),
            chat_api_key="sk-test",
            chat_provider="",
            chat_model="",
            chat_base_url="",
        )
        provider = create_chat_provider(settings)
        self.assertEqual(provider.name, "openai-compatible:deepseek-chat")
        self.assertEqual(provider.base_url, "https://api.deepseek.com/v1")

    def test_missing_key(self) -> None:
        settings = Settings(data_dir=Path("."), chat_api_key="")
        with self.assertRaises(ChatError) as ctx:
            create_chat_provider(settings)
        self.assertEqual(ctx.exception.code, "CHAT_NOT_CONFIGURED")


if __name__ == "__main__":
    unittest.main()
