"""模块 8（llm / embedding）验收测试：不访问网络。"""

from __future__ import annotations

import unittest

import requests

from paperpilot.config import Settings
from paperpilot.llm import (
    EmbeddingError,
    HashEmbedding,
    SiliconFlowEmbedding,
    create_embedding_provider,
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


class HashEmbeddingTest(unittest.TestCase):
    def test_deterministic_and_normalized(self) -> None:
        provider = HashEmbedding(dimension=32)
        first = provider.embed(["hello world"])[0]
        second = provider.embed(["hello world"])[0]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertAlmostEqual(sum(value * value for value in first), 1.0, places=6)

    def test_different_texts_differ(self) -> None:
        provider = HashEmbedding(dimension=32)
        a, b = provider.embed(["alpha", "beta"])
        self.assertNotEqual(a, b)

    def test_invalid_dimension(self) -> None:
        with self.assertRaises(ValueError):
            HashEmbedding(dimension=0)


class SiliconFlowEmbeddingTest(unittest.TestCase):
    def _provider(self, post):
        return SiliconFlowEmbedding(
            api_key="test-key",
            model="BAAI/bge-m3",
            base_url="https://api.example.com/v1",
            dimension=4,
            post=post,
        )

    def test_request_shape_and_parse(self) -> None:
        calls: dict = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            calls.update(url=url, payload=json, headers=headers, timeout=timeout)
            return FakeResponse(
                payload={
                    "data": [
                        {"index": index, "embedding": [0.1, 0.2, 0.3, 0.4]}
                        for index in range(len(json["input"]))
                    ]
                }
            )

        vectors = self._provider(fake_post).embed(["a", "b"])

        self.assertEqual(len(vectors), 2)
        self.assertTrue(all(len(vector) == 4 for vector in vectors))
        self.assertEqual(calls["url"], "https://api.example.com/v1/embeddings")
        self.assertEqual(calls["payload"]["model"], "BAAI/bge-m3")
        self.assertEqual(calls["payload"]["input"], ["a", "b"])
        self.assertTrue(calls["headers"]["Authorization"].startswith("Bearer "))

    def test_http_error(self) -> None:
        def fake_post(*args, **kwargs):
            return FakeResponse(status_code=401, text="unauthorized")

        with self.assertRaises(EmbeddingError) as ctx:
            self._provider(fake_post).embed(["x"])
        self.assertEqual(ctx.exception.code, "EMBEDDING_HTTP")

    def test_bad_count(self) -> None:
        def fake_post(*args, **kwargs):
            return FakeResponse(payload={"data": []})

        with self.assertRaises(EmbeddingError) as ctx:
            self._provider(fake_post).embed(["x"])
        self.assertEqual(ctx.exception.code, "EMBEDDING_BAD_RESPONSE")

    def test_dimension_mismatch(self) -> None:
        def fake_post(*args, **kwargs):
            return FakeResponse(payload={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

        with self.assertRaises(EmbeddingError) as ctx:
            self._provider(fake_post).embed(["x"])
        self.assertEqual(ctx.exception.code, "EMBEDDING_DIMENSION")

    def test_network_error(self) -> None:
        def fake_post(*args, **kwargs):
            raise requests.ConnectionError("boom")

        with self.assertRaises(EmbeddingError) as ctx:
            self._provider(fake_post).embed(["x"])
        self.assertEqual(ctx.exception.code, "EMBEDDING_NETWORK")

    def test_retries_transient_status(self) -> None:
        calls = {"count": 0}

        def fake_post(url, json=None, headers=None, timeout=None):
            calls["count"] += 1
            if calls["count"] == 1:
                return FakeResponse(status_code=503, text="busy")
            return FakeResponse(
                payload={
                    "data": [
                        {"index": index, "embedding": [0.1, 0.2, 0.3, 0.4]}
                        for index in range(len(json["input"]))
                    ]
                }
            )

        vectors = self._provider(fake_post).embed(["a"])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(calls["count"], 2)

    def test_empty_input(self) -> None:
        self.assertEqual(self._provider(lambda *a, **k: None).embed([]), [])


class FactoryTest(unittest.TestCase):
    def test_hash_provider(self) -> None:
        settings = Settings(data_dir=__import__("pathlib").Path("."), embedding_provider="hash", embedding_dimension=8)
        provider = create_embedding_provider(settings)
        self.assertIsInstance(provider, HashEmbedding)
        self.assertEqual(provider.dimension, 8)

    def test_siliconflow_from_settings(self) -> None:
        settings = Settings(
            data_dir=__import__("pathlib").Path("."),
            embedding_provider="siliconflow",
            embedding_api_key="k",
            embedding_model="BAAI/bge-m3",
            embedding_dimension=4,
        )
        provider = create_embedding_provider(settings)
        self.assertEqual(provider.name, "siliconflow:BAAI/bge-m3")
        self.assertEqual(provider.dimension, 4)

    def test_missing_key(self) -> None:
        settings = Settings(data_dir=__import__("pathlib").Path("."), embedding_provider="siliconflow")
        with self.assertRaises(EmbeddingError) as ctx:
            create_embedding_provider(settings)
        self.assertEqual(ctx.exception.code, "EMBEDDING_NOT_CONFIGURED")

    def test_unknown_provider(self) -> None:
        settings = Settings(data_dir=__import__("pathlib").Path("."), embedding_provider="nope")
        with self.assertRaises(EmbeddingError) as ctx:
            create_embedding_provider(settings)
        self.assertEqual(ctx.exception.code, "EMBEDDING_UNKNOWN_PROVIDER")


if __name__ == "__main__":
    unittest.main()
