"""图谱抽取（模块 11）：从片段中抽取受约束的关系声明。

关系类型固定四种；每条声明必须带原文证据短句，程序会检查它是否真的
出现在该片段里（对不上的标为待复核，不发布为有效关系）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..llm import ChatMessage, ChatProvider
from ..schemas import Chunk
from .client import EXTRACTOR_VERSION, GraphError

# 每种关系允许的主语/宾语类型
EXPECTED_TYPES: dict[str, tuple[set[str], set[str]]] = {
    "USES_METHOD": ({"Paper"}, {"Method"}),
    "APPLIED_IN": ({"Method"}, {"Experiment"}),
    "PRODUCES": ({"Experiment"}, {"Result"}),
    "SUPPORTS_CONCLUSION": ({"Result", "Experiment"}, {"Conclusion"}),
}

SYSTEM_PROMPT = """你是学术论文信息抽取助手。请从给定片段中抽取受约束的关系声明。

只输出一个 JSON 对象，不要解释，不要代码块：
{"claims": [
  {"chunk_id": "原样返回片段编号",
   "relation_type": "USES_METHOD|APPLIED_IN|PRODUCES|SUPPORTS_CONCLUSION",
   "subject": {"type": "Paper|Method|Experiment|Result|Conclusion", "name": "实体名"},
   "object": {"type": "Paper|Method|Experiment|Result|Conclusion", "name": "实体名"},
   "evidence_text": "片段中出现的原文短句（不超过 200 字，不能改写）",
   "qualifiers": {"可选": "实验条件等"}}
]}

关系与类型限制：
- USES_METHOD：Paper → Method（论文采用/提出/使用某方法；综述中“本文综述了某方法”也可视为讨论对象，但要忠于原文表述）
- APPLIED_IN：Method → Experiment（某方法用于某项实验/数据集/评测）
- PRODUCES：Experiment → Result（实验产生某结果/指标）
- SUPPORTS_CONCLUSION：Result/Experiment → Conclusion（结果支持某结论）

示例（仅示意格式，不要照抄）：
{"claims": [
  {"chunk_id": "c1", "relation_type": "APPLIED_IN",
   "subject": {"type": "Method", "name": "BM25 检索"},
   "object": {"type": "Experiment", "name": "NaturalQuestions 评测"},
   "evidence_text": "we evaluate BM25 retrieval on NaturalQuestions"},
  {"chunk_id": "c1", "relation_type": "PRODUCES",
   "subject": {"type": "Experiment", "name": "NaturalQuestions 评测"},
   "object": {"type": "Result", "name": "F1 提升 2.1"},
   "evidence_text": "the F1 score improves by 2.1"}
]}

要求：
1. 只抽取片段中明确表达的关系，不要根据常识补充；
2. evidence_text 必须是片段原文的子串，不能改写、不能翻译；
3. 没有可抽取的关系就返回 {"claims": []}；
4. 实体名保持原文语言，简洁具体。"""


@dataclass(frozen=True)
class ExtractedClaim:
    chunk_id: str
    relation_type: str
    subject_type: str
    subject_name: str
    object_type: str
    object_name: str
    evidence_text: str
    qualifiers: dict
    evidence_aligned: bool


def extract_claims(
    chat: ChatProvider,
    chunks: Sequence[Chunk],
    *,
    batch_size: int = 5,
    cache_dir: Path | None = None,
) -> list[ExtractedClaim]:
    """按批次抽取关系声明；模型输出不合法时抛 ``GraphError``。

    传入 ``cache_dir`` 时按（模型 + 抽取版本 + 批次原文）缓存模型输出，
    图谱重建且片段未变化时不再重复调用模型。
    """
    claims: list[ExtractedClaim] = []
    step = max(1, batch_size)
    for start in range(0, len(chunks), step):
        batch = list(chunks[start : start + step])
        payload = _extract_batch(chat, batch, cache_dir)
        for raw in payload.get("claims") or []:
            claim = _build_claim(raw, batch)
            if claim is not None:
                claims.append(claim)
    return claims


def _cache_path(cache_dir: Path, chat: ChatProvider, batch: Sequence[Chunk]) -> Path:
    digest = hashlib.sha256()
    digest.update(EXTRACTOR_VERSION.encode("utf-8"))
    digest.update(b"|")
    digest.update(
        str(getattr(chat, "model", "") or getattr(chat, "name", "")).encode("utf-8")
    )
    digest.update(b"|")
    for chunk in batch:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\x1e")
    return cache_dir / f"{digest.hexdigest()}.json"


def _extract_batch(
    chat: ChatProvider, batch: Sequence[Chunk], cache_dir: Path | None
) -> dict:
    path = _cache_path(cache_dir, chat, batch) if cache_dir is not None else None
    if path is not None:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, dict):
            return cached

    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=_materials(batch)),
    ]
    response = chat.chat(messages)
    payload = _parse_json(response.message.content or "")
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, path)
        except OSError:
            pass  # 缓存失败不影响抽取结果
    return payload


def _materials(batch: Sequence[Chunk]) -> str:
    parts = ["请从以下片段中抽取关系（chunk_id 必须原样返回）："]
    for chunk in batch:
        parts.append(f"chunk_id: {chunk.chunk_id}\n{chunk.text}")
    return "\n\n".join(parts)


def _build_claim(raw, batch: Sequence[Chunk]) -> ExtractedClaim | None:
    if not isinstance(raw, dict):
        return None
    chunk_id = str(raw.get("chunk_id") or "").strip()
    chunk = next((item for item in batch if item.chunk_id == chunk_id), None)
    if chunk is None:
        return None

    relation = str(raw.get("relation_type") or "").strip().upper()
    expected = EXPECTED_TYPES.get(relation)
    if expected is None:
        return None

    subject = raw.get("subject") if isinstance(raw.get("subject"), dict) else {}
    obj = raw.get("object") if isinstance(raw.get("object"), dict) else {}
    subject_type = str(subject.get("type") or "").strip()
    object_type = str(obj.get("type") or "").strip()
    if subject_type not in expected[0] or object_type not in expected[1]:
        return None

    subject_name = str(subject.get("name") or "").strip()
    object_name = str(obj.get("name") or "").strip()
    if not (1 <= len(subject_name) <= 200 and 1 <= len(object_name) <= 200):
        return None

    evidence_text = str(raw.get("evidence_text") or "").strip()
    if not evidence_text:
        return None

    qualifiers = raw.get("qualifiers")
    if not isinstance(qualifiers, dict):
        qualifiers = {}

    return ExtractedClaim(
        chunk_id=chunk_id,
        relation_type=relation,
        subject_type=subject_type,
        subject_name=subject_name,
        object_type=object_type,
        object_name=object_name,
        evidence_text=evidence_text[:200],
        qualifiers=qualifiers,
        evidence_aligned=_normalize(evidence_text) in _normalize(chunk.text),
    )


def _normalize(text: str) -> str:
    """对齐比较忽略空白：PDF 提取常在中文词之间插入空格（“产 教 融 合”）。"""
    return re.sub(r"\s+", "", text).lower()


def _parse_json(content: str) -> dict:
    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise GraphError("模型输出中没有 JSON 对象", "GRAPH_BAD_EXTRACTION")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise GraphError(
            f"抽取结果不是合法 JSON：{exc}", "GRAPH_BAD_EXTRACTION"
        ) from exc
    if not isinstance(data, dict):
        raise GraphError("抽取结果顶层必须是对象", "GRAPH_BAD_EXTRACTION")
    return data
