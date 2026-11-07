"""read_passages 工具：回读原文并登记本次运行的证据。

这是唯一能产生 ``evidence_id`` 的工具；模型回答里的引用只能来自这里。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..evidence.passages import read_passages as read_evidence_passages
from .registry import ToolContext, ToolOutcome, ToolSpec


class ReadPassagesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_ids: list[str] = Field(min_length=1, max_length=5)
    neighbor: int = Field(default=1, ge=0, le=2)


def handle(args: ReadPassagesArgs, ctx: ToolContext) -> ToolOutcome:
    passages = read_evidence_passages(
        ctx.conn,
        ctx.evidence,
        args.chunk_ids,
        allowed_paper_ids=set(ctx.allowed_paper_ids),
        neighbor=args.neighbor,
    )
    return ToolOutcome(
        data={
            "passages": [
                {
                    "evidence_id": passage.evidence.evidence_id,
                    "chunk_id": passage.chunk.chunk_id,
                    "paper_id": passage.chunk.paper_id,
                    "page_number": passage.evidence.page_number,
                    "text": passage.chunk.text,
                    "neighbors": [
                        {
                            "chunk_id": neighbor.chunk_id,
                            "page_number": neighbor.page_number,
                            "text": neighbor.text,
                        }
                        for neighbor in passage.neighbors
                    ],
                }
                for passage in passages
            ]
        },
        evidence_ids=tuple(
            passage.evidence.evidence_id for passage in passages
        ),
    )


SPEC = ToolSpec(
    name="read_passages",
    description=(
        "按 chunk_id 回读原文并登记证据，返回原文、页码和 evidence_id。"
        "每次 1~5 个片段；回答中只能引用这里返回的 evidence_id（格式 ev_001）。"
        "如果需要上下文，可以把 neighbor 设为 1 或 2。"
    ),
    args_model=ReadPassagesArgs,
    handler=handle,
)
