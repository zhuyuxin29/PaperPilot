from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..graph.client import WORKSPACE_ID, GraphError
from .. import storage
from .registry import ToolContext, ToolOutcome, ToolSpec, ToolError


class QueryGraphArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_type: Literal[
        "method_experiments", "experiment_results", "conclusion_support"
    ]
    entity_name: str | None = Field(default=None, max_length=200)
    limit: int = Field(default=10, ge=1, le=20)


def handle(args: QueryGraphArgs, ctx: ToolContext) -> ToolOutcome:
    if ctx.settings.graph_mode == "off":
        raise ToolError("图谱已关闭（GRAPH_MODE=off）", "GRAPH_DISABLED")
    if ctx.graph is None:
        raise ToolError("图谱暂不可用（未连接或未构建）", "GRAPH_UNAVAILABLE")

    for paper_id in ctx.allowed_paper_ids:
        paper = storage.get_paper(ctx.conn, paper_id)
        state = storage.get_graph_state(ctx.conn, paper_id)
        if (paper is None or state is None or state["revision"] != paper.revision
                or state["status"] != "ready"):
            raise ToolError("当前论文修订的图谱尚未就绪，请先构建图谱", "GRAPH_NOT_READY")
    try:
        rows = ctx.graph.query_template(
            args.query_type,
            workspace_id=WORKSPACE_ID,
            paper_ids=list(ctx.allowed_paper_ids),
            entity_name=args.entity_name,
            limit=args.limit,
        )
    except GraphError as exc:
        raise ToolError(exc.message, exc.code) from exc

    return ToolOutcome(
        data={
            "query_type": args.query_type,
            "claims": rows,
            "graph_mode": ctx.settings.graph_mode,
        }
    )


SPEC = ToolSpec(
    name="query_graph",
    description=(
        "查询论文内部的实体关系（方法→实验、实验→结果、结果/实验→结论）。"
        "关系型问题通常适合使用它；普通事实/术语类问题用 search_chunks 即可，由你判断。"
        "本工具返回关系声明和来源 chunk_id，不返回原文；引用前请再用 read_passages 回读。"
        "query_type 只能是 method_experiments / experiment_results / conclusion_support。"
    ),
    args_model=QueryGraphArgs,
    handler=handle,
)
