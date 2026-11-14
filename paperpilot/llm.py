from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import requests

from .config import Settings

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DIMENSION = 1024
_BATCH_SIZE = 32


_SESSION = requests.Session()
# 只重试限流和暂时性 5xx，其余 4xx 直接返回给上层。
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 0.15


def _sleep_backoff(attempt: int, response: object | None = None) -> None:
    delay = _BACKOFF_BASE * (2 ** attempt)
    headers = getattr(response, "headers", None) or {}
    retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
    if retry_after is not None:
        try:
            delay = max(delay, float(retry_after))
        except (TypeError, ValueError):
            pass
    time.sleep(min(delay, 2.0))


def _post_with_retry(
    post: Callable[..., object],
    url: str,
    *,
    json: dict,
    headers: dict,
    timeout: float,
) -> object:
    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = post(url, json=json, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 >= _MAX_ATTEMPTS:
                raise
            _sleep_backoff(attempt)
            continue
        status = int(getattr(response, "status_code", 0))
        if status in _RETRY_STATUS and attempt + 1 < _MAX_ATTEMPTS:
            _sleep_backoff(attempt, response)
            continue
        return response
    if last_error is not None:
        raise last_error
    raise RuntimeError("HTTP 请求重试失败")

_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9._-]*")
_CJK_CHAR = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


class EmbeddingError(RuntimeError):
    def __init__(self, message: str, code: str = "EMBEDDING_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class EmbeddingProvider(Protocol):
    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = _ASCII_WORD.findall(lowered)
    cjk = _CJK_CHAR.findall(lowered)
    tokens.extend(cjk)
    tokens.extend("".join(pair) for pair in zip(cjk, cjk[1:]))
    return tokens or [lowered]


class HashEmbedding:
    name = "hash"

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension 必须为正数")
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(item * item for item in vector))
        if norm == 0.0:
            return [1.0] + [0.0] * (self.dimension - 1)
        return [item / norm for item in vector]


class SiliconFlowEmbedding:
    name = "siliconflow"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        dimension: int = DEFAULT_DIMENSION,
        timeout: float = 60.0,
        batch_size: int = _BATCH_SIZE,
        post: Callable[..., object] | None = None,
    ) -> None:
        if not api_key:
            raise EmbeddingError(
                "未配置 EMBEDDING_API_KEY，无法调用硅基流动接口",
                "EMBEDDING_NOT_CONFIGURED",
            )
        if dimension <= 0:
            raise ValueError("dimension 必须为正数")
        self._api_key = api_key
        self.model = model

        self.name = f"siliconflow:{model}"
        self.base_url = base_url.rstrip("/")
        self.dimension = dimension
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self._post = post or _SESSION.post

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": batch}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = _post_with_retry(
                self._post,
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise EmbeddingError(f"Embedding 请求失败：{exc}", "EMBEDDING_NETWORK") from exc

        status = int(getattr(response, "status_code", 0))
        if status != 200:
            body = str(getattr(response, "text", ""))[:300]
            raise EmbeddingError(
                f"Embedding 接口返回 HTTP {status}：{body}", "EMBEDDING_HTTP"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise EmbeddingError(
                "Embedding 返回不是合法 JSON", "EMBEDDING_BAD_RESPONSE"
            ) from exc

        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != len(batch):
            received = len(items) if isinstance(items, list) else "未知"
            raise EmbeddingError(
                f"Embedding 返回条数与请求不一致：请求 {len(batch)}，返回 {received}",
                "EMBEDDING_BAD_RESPONSE",
            )
        items = sorted(items, key=lambda item: item.get("index", 0))

        vectors: list[list[float]] = []
        for item in items:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list) or not vector:
                raise EmbeddingError(
                    "Embedding 返回缺少向量", "EMBEDDING_BAD_RESPONSE"
                )
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    f"Embedding 维度不符：期望 {self.dimension}，实际 {len(vector)}",
                    "EMBEDDING_DIMENSION",
                )
            vectors.append([float(value) for value in vector])
        return vectors


def create_embedding_provider(settings: Settings, *, post=None) -> EmbeddingProvider:
    provider = (settings.embedding_provider or "").strip().lower()
    if provider == "hash":
        return HashEmbedding(settings.embedding_dimension or 256)
    if provider in ("", "siliconflow", "openai", "openai-compatible"):
        if not settings.embedding_api_key:
            raise EmbeddingError(
                "未配置 EMBEDDING_API_KEY；请在 .env 中填写硅基流动的 API Key",
                "EMBEDDING_NOT_CONFIGURED",
            )
        kwargs = dict(
            api_key=settings.embedding_api_key,
            model=settings.embedding_model or DEFAULT_MODEL,
            base_url=settings.embedding_base_url or DEFAULT_BASE_URL,
            dimension=settings.embedding_dimension or DEFAULT_DIMENSION,
        )
        if post is not None:
            kwargs["post"] = post
        return SiliconFlowEmbedding(**kwargs)
    raise EmbeddingError(
        f"未知的 EMBEDDING_PROVIDER：{provider!r}", "EMBEDDING_UNKNOWN_PROVIDER"
    )


class ChatError(RuntimeError):
    def __init__(self, message: str, code: str = "CHAT_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class ChatResponse:
    message: ChatMessage
    usage: dict[str, int] | None = None
    model: str | None = None


class ChatProvider(Protocol):
    name: str

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict] | None = None,
        timeout: float | None = None,
    ) -> ChatResponse:
        ...


class OpenAICompatibleChat:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 60.0,
        post: Callable[..., object] | None = None,
    ) -> None:
        if not api_key:
            raise ChatError("未配置对话模型 API Key", "CHAT_NOT_CONFIGURED")
        if not model or not base_url:
            raise ChatError(
                "请配置 CHAT_MODEL 与 CHAT_BASE_URL", "CHAT_NOT_CONFIGURED"
            )
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = f"openai-compatible:{model}"
        self._post = post or _SESSION.post

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict] | None = None,
        timeout: float | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._serialize(message) for message in messages],
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = _post_with_retry(
                self._post,
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout or self.timeout,
            )
        except requests.RequestException as exc:
            raise ChatError(f"对话模型请求失败：{exc}", "CHAT_NETWORK") from exc

        status = int(getattr(response, "status_code", 0))
        if status != 200:
            body = str(getattr(response, "text", ""))[:300]
            raise ChatError(f"对话模型返回 HTTP {status}：{body}", "CHAT_HTTP")
        try:
            data = response.json()
        except ValueError as exc:
            raise ChatError(
                "对话模型返回不是合法 JSON", "CHAT_BAD_RESPONSE"
            ) from exc

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ChatError("对话模型返回缺少 choices", "CHAT_BAD_RESPONSE")
        raw_message = (
            choices[0].get("message") if isinstance(choices[0], dict) else None
        )
        if not isinstance(raw_message, dict):
            raise ChatError("对话模型返回缺少 message", "CHAT_BAD_RESPONSE")

        tool_calls: list[ToolCall] = []
        for index, item in enumerate(raw_message.get("tool_calls") or []):
            function = item.get("function") if isinstance(item, dict) else None
            if not isinstance(function, dict):
                raise ChatError("工具调用缺少 function", "CHAT_BAD_RESPONSE")
            raw_arguments = function.get("arguments") or "{}"
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    raise ChatError(
                        f"工具参数不是合法 JSON：{raw_arguments[:100]}",
                        "CHAT_BAD_RESPONSE",
                    ) from exc
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                raise ChatError("工具参数类型不支持", "CHAT_BAD_RESPONSE")
            tool_calls.append(
                ToolCall(
                    id=str(item.get("id") or f"call_{index}"),
                    name=str(function.get("name") or ""),
                    arguments=dict(arguments),
                )
            )

        content = raw_message.get("content")
        if content is None and not tool_calls:
            raise ChatError("对话模型既没有内容也没有工具调用", "CHAT_BAD_RESPONSE")

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content=content,
                tool_calls=tuple(tool_calls),
            ),
            usage=usage,
            model=data.get("model"),
        )

    @staticmethod
    def _serialize(message: ChatMessage) -> dict[str, Any]:
        item: dict[str, Any] = {"role": message.role}
        if message.content is not None:
            item["content"] = message.content
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        if message.name:
            item["name"] = message.name
        return item


def create_chat_provider(settings: Settings, *, post=None) -> ChatProvider:
    provider = (settings.chat_provider or "").strip().lower()
    if not settings.chat_api_key:
        raise ChatError(
            "未配置 CHAT_API_KEY（或 DEEPSEEK_API_KEY）", "CHAT_NOT_CONFIGURED"
        )
    if provider in ("", "deepseek", "siliconflow", "openai", "openai-compatible"):
        default_base = (
            "https://api.deepseek.com/v1"
            if provider in ("", "deepseek")
            else "https://api.siliconflow.cn/v1"
        )
        default_model = "deepseek-chat" if provider in ("", "deepseek") else ""
        kwargs = dict(
            api_key=settings.chat_api_key,
            model=settings.chat_model or default_model,
            base_url=settings.chat_base_url or default_base,
            timeout=float(settings.timeout_seconds or 60),
        )
        if post is not None:
            kwargs["post"] = post
        return OpenAICompatibleChat(**kwargs)
    raise ChatError(f"未知的 CHAT_PROVIDER：{provider!r}", "CHAT_UNKNOWN_PROVIDER")
