from __future__ import annotations

import argparse
import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from paperpilot import storage
from paperpilot.agent.policy import Budget
from paperpilot.config import load_settings
from paperpilot.graph.client import GraphError, Neo4jClient
from paperpilot.llm import create_chat_provider, create_embedding_provider

from .dataset import DatasetError, load_dataset
from .report import render_markdown
from .runner import run_agent_eval, run_retrieval_eval

CONFIG_DESCRIPTIONS = {
    "A": "固定流程 RAG（无图谱）",
    "B": "固定流程 RAG + 图谱候选",
    "C": "Agent（无图谱工具）",
    "D": "Agent + 图谱工具",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PaperPilot 评测")
    parser.add_argument(
        "--dataset",
        required=True,
        help="JSONL 数据集路径",
    )
    parser.add_argument(
        "--config",
        choices=sorted(CONFIG_DESCRIPTIONS),
        default="A",
        help="评测配置：A/B 检索，C/D Agent",
    )
    parser.add_argument("--top-k", type=int, default=8, help="检索 Top-K")
    parser.add_argument("--max-steps", type=int, default=8, help="Agent 最大轮次")
    parser.add_argument(
        "--max-tool-calls", type=int, default=12, help="Agent 最大工具调用"
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="单题超时秒数")
    parser.add_argument("--out", default=None, help="结果 JSON 输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    settings.ensure_dirs()

    try:
        questions = load_dataset(args.dataset)
    except DatasetError as exc:
        print(f"数据集错误：{exc}", file=sys.stderr)
        return 2

    conn = storage.connect(settings.db_path)
    storage.ensure_schema(conn)
    provider = create_embedding_provider(settings)

    graph_client = None
    if args.config in ("B", "D"):
        try:
            graph_client = Neo4jClient.connect(settings)
        except GraphError as exc:
            conn.close()
            print(f"图谱不可用：{exc}", file=sys.stderr)
            return 2

    try:
        if args.config in ("A", "B"):
            result = run_retrieval_eval(
                conn,
                settings,
                provider,
                questions,
                use_graph=args.config == "B",
                graph_client=graph_client,
                top_k=args.top_k,
            )
        else:
            chat = create_chat_provider(settings)
            budget = Budget(
                max_steps=args.max_steps,
                max_tool_calls=args.max_tool_calls,
                max_seconds=args.timeout,
            )
            result = run_agent_eval(
                conn,
                settings,
                provider,
                chat,
                questions,
                use_graph=args.config == "D",
                graph_client=graph_client,
                budget=budget,
            )
    finally:
        if graph_client is not None:
            graph_client.close()
        conn.close()

    now = datetime.now(timezone.utc)
    result["config_description"] = CONFIG_DESCRIPTIONS[args.config]
    result["dataset"] = str(Path(args.dataset))
    result["generated_at"] = now.isoformat()
    result["manifest"] = {
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "graph_mode": settings.graph_mode,
        "max_steps": args.max_steps, "max_tool_calls": args.max_tool_calls,
        "timeout": args.timeout, "evaluation_version": "2",
        "agent_output_schema": "claims-v1",
        "summary_cache_enabled": False,
    }

    out_path = (
        Path(args.out)
        if args.out
        else Path("evaluation/results")
        / f"{args.config}-{now.strftime('%Y%m%d-%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = out_path.with_suffix(".md")
    report_path.write_text(render_markdown(result), encoding="utf-8")

    print(f"配置 {args.config}：{CONFIG_DESCRIPTIONS[args.config]}")
    for key, value in result["summary"].items():
        if key != "config":
            print(f"  {key}: {value}")
    print(f"结果：{out_path}")
    print(f"报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
