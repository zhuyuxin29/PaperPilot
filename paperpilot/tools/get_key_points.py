"""get_key_points 工具：读取已生成的七类要点。

只读缓存，不触发生成；缓存缺失/过期分别返回 ``not_generated`` / ``stale``。
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from .. import storage
from ..summary.generator import (
    SummaryError,
    current_summary_model,
    load_summary,
    summary_path,
)
from .registry import ToolContext, ToolOutcome, ToolSpec, ToolError


class GetKeyPointsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str = Field(min_length=1, max_length=64)


def handle(args: GetKeyPointsArgs, ctx: ToolContext) -> ToolOutcome:
    if args.paper_id not in ctx.allowed_paper_ids:
        raise ToolError(
            f"论文不在当前对话范围：{args.paper_id}", "OUT_OF_SCOPE"
        )
    paper = storage.get_paper(ctx.conn, args.paper_id)
    if paper is None:
        raise ToolError(f"文献不存在：{args.paper_id}", "PAPER_NOT_FOUND")

    model_id = current_summary_model(ctx.settings)
    path = summary_path(ctx.settings, paper.id, paper.revision, model_id)
    if not path.is_file():
        return ToolOutcome(
            data={
                "status": "not_generated",
                "summary": None,
                "reason": "尚未生成七类要点，可以在界面点击“生成要点”",
            }
        )

    try:
        summary = load_summary(
            ctx.settings,
            path,
            paper_id=paper.id,
            revision=paper.revision,
            model_id=model_id,
        )
    except SummaryError as exc:
        return ToolOutcome(
            data={"status": "stale", "summary": None, "reason": exc.message}
        )

    payload = json.loads(summary.model_dump_json())
    mapping = payload.pop("evidence_map", {})
    for section in payload["sections"].values():
        for item in section["items"]:
            ids = item.pop("evidence_ids")
            if any(identifier not in mapping for identifier in ids):
                return ToolOutcome(data={"status": "stale", "summary": None,
                                         "reason": "缓存缺少来源锚点，请重新生成"})
            item["source_chunk_ids"] = [mapping[identifier] for identifier in ids]
    return ToolOutcome(
        data={
            "status": "valid",
            "summary": payload,
            "reason": None,
        }
    )


SPEC = ToolSpec(
    name="get_key_points",
    description=(
        "读取已生成的七类要点（背景/问题/方法/实验/结论/贡献/局限）。"
        "返回缓存内容与状态；要点里的引用是生成时登记的，不能直接当作本次运行的证据，"
        "需要引用原文时请再用 read_passages 回读。"
    ),
    args_model=GetKeyPointsArgs,
    handler=handle,
)
