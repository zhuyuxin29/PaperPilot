"""search_chunks 工具：在当前论文范围内检索候选片段。

输出只是**候选**，不是证据；模型要引用必须再调用 ``read_passages``。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..rag import retrieval
from .registry import ToolContext, ToolOutcome, ToolSpec


class SearchChunksArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"


def handle(args: SearchChunksArgs, ctx: ToolContext) -> ToolOutcome:
    candidates = retrieval.search_chunks(
        ctx.conn,
        ctx.settings,
        args.query,
        list(ctx.allowed_paper_ids),
        provider=ctx.provider,
        mode=args.mode,
        top_k=args.top_k,
    )
    return ToolOutcome(
        data={
            "recommended_chunk_ids": [
                candidate.chunk_id for candidate in candidates[:2]
            ],
            "candidates": [
                {
                    "chunk_id": candidate.chunk_id,
                    "paper_id": candidate.paper_id,
                    "page_number": candidate.page_number,
                    "snippet": candidate.snippet,
                    "rank": candidate.rank,
                    "routes": list(candidate.routes),
                    "score": round(candidate.score, 6),
                }
                for candidate in candidates
            ]
        }
    )


SPEC = ToolSpec(
    name="search_chunks",
    description=(
        "在当前选中的论文范围内检索候选片段（向量+关键词混合）。"
        "返回的只是候选，不能直接引用；需要引用时请再用 read_passages 回读原文。"
        "返回中的 recommended_chunk_ids 是融合排名前 2，请优先读取它们再判断。"
        "query 用自然语言描述要查找的内容，可用英文关键词或中文。"
    ),
    args_model=SearchChunksArgs,
    handler=handle,
)
