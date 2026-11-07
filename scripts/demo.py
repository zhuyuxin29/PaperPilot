"""PaperPilot 一条命令演示：入库 → 索引 → 要点 → 问答 →（可选）图谱。

用法::

    python scripts/demo.py
    python scripts/demo.py --pdf path/to/paper.pdf
    python scripts/demo.py --question "这篇论文的主要方法是什么？"
    python scripts/demo.py --with-graph
    python scripts/demo.py --force-summary

需要先在 .env 中配置 CHAT_API_KEY / EMBEDDING_API_KEY（或对应服务商 Key）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperpilot import storage  # noqa: E402
from paperpilot.agent.runner import run_agent  # noqa: E402
from paperpilot.config import load_settings  # noqa: E402
from paperpilot.graph.builder import build_paper_graph  # noqa: E402
from paperpilot.graph.client import WORKSPACE_ID, GraphError, Neo4jClient  # noqa: E402
from paperpilot.ingestion import pipeline  # noqa: E402
from paperpilot.llm import create_chat_provider, create_embedding_provider  # noqa: E402
from paperpilot.rag import indexing, keyword  # noqa: E402
from paperpilot.summary.generator import (  # noqa: E402
    SUMMARY_LABELS,
    generate_summary,
)
from paperpilot.tools import build_registry  # noqa: E402


def _section(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def main() -> int:
    parser = argparse.ArgumentParser(description="PaperPilot 演示")
    parser.add_argument("--pdf", type=Path, help="可选：先入库这个 PDF")
    parser.add_argument("--paper-id", help="使用指定论文（默认最新一篇）")
    parser.add_argument(
        "--question",
        default="这篇论文的主要研究方法是什么？",
        help="演示提问",
    )
    parser.add_argument(
        "--with-graph", action="store_true", help="构建并使用 Neo4j 图谱"
    )
    parser.add_argument(
        "--force-summary", action="store_true", help="强制重新生成七类要点"
    )
    args = parser.parse_args()

    settings = load_settings()
    settings.ensure_dirs()
    conn = storage.connect(settings.db_path)
    storage.ensure_schema(conn)

    graph_client = None
    try:
        if args.pdf is not None:
            _section("入库 PDF")
            data = args.pdf.read_bytes()
            result = pipeline.ingest_pdf(data, args.pdf.name, settings, conn)
            print(
                f"入库完成：{result.paper.id} · {result.page_count} 页 · "
                f"{result.chunk_count} 个片段"
            )

        papers = storage.list_papers(conn)
        if not papers:
            print("文献库为空：请先用 --pdf 指定一篇 PDF。")
            return 2
        paper = (
            storage.get_paper(conn, args.paper_id)
            if args.paper_id
            else papers[0]
        )
        if paper is None:
            print(f"找不到论文：{args.paper_id}")
            return 2
        print(f"\n使用论文：{paper.id} · {paper.title}")

        _section("建立索引")
        vector_path = settings.index_dir / f"{paper.id}.r{paper.revision}.npz"
        provider = create_embedding_provider(settings)
        if not vector_path.is_file():
            meta = indexing.build_paper_index(
                conn, settings, provider, paper.id, paper.revision
            )
            print(f"向量索引：{meta.chunk_count} 个片段")
        else:
            print(f"向量索引已存在：{vector_path.name}")
        if not keyword.index_path(settings, paper.id, paper.revision).is_file():
            index = keyword.build_keyword_index(
                conn, settings, paper.id, paper.revision
            )
            print(f"关键词索引：{len(index.chunk_ids)} 个片段")
        else:
            print("关键词索引已存在")

        _section("七类阅读要点")
        if args.with_graph:
            try:
                graph_client = Neo4jClient.connect(settings)
            except GraphError as exc:
                print(f"图谱不可用，跳过：{exc.message}")
        chat = create_chat_provider(settings)
        summary_result = generate_summary(
            conn,
            settings,
            chat,
            paper.id,
            force=args.force_summary,
            graph=graph_client,
        )
        summary = summary_result.summary
        print("缓存复用" if summary_result.reused else "已重新生成")
        for key, section in summary.sections.items():
            print(f"  {SUMMARY_LABELS[key]}：{section.status.value}")
        print(
            f"  覆盖：{summary.coverage.status} · "
            f"{summary.coverage.processed_units}/{summary.coverage.total_units} 片段"
        )

        if args.with_graph and graph_client is not None:
            _section("构建图谱")
            build = build_paper_graph(
                conn, settings, chat, graph_client, paper.id
            )
            print(
                f"状态：{build.status} · 有效关系 {build.claims_validated} 条 · "
                f"待复核 {build.claims_pending} 条"
            )

        _section("文献问答")
        question = args.question
        print(f"问题：{question}")
        answer = run_agent(
            question=question,
            paper_ids=(paper.id,),
            conn=conn,
            settings=settings,
            embedding=provider,
            chat=chat,
            tools=build_registry(graph_enabled=graph_client is not None),
            graph=graph_client,
        )
        print(f"状态：{answer.status}（{answer.stop_reason or '正常结束'}）")
        print(f"回答：{(answer.answer or '').strip()[:600]}")
        print(
            "引用："
            + "、".join(
                f"[{index}] 第 {record.page_number} 页"
                for index, record in enumerate(answer.citations, start=1)
            )
        )
        print(
            "工具序列："
            + " → ".join(entry.get("tool", "") for entry in answer.trace)
        )

        if args.with_graph and graph_client is not None:
            _section("图谱查询结果")
            for query_type in (
                "method_experiments",
                "experiment_results",
                "conclusion_support",
            ):
                rows = graph_client.query_template(
                    query_type,
                    workspace_id=WORKSPACE_ID,
                    paper_ids=[paper.id],
                    limit=5,
                )
                print(f"  {query_type}：{len(rows)} 条")
                for row in rows[:2]:
                    print(
                        f"    {row.get('subject')} → {row.get('object')} "
                        f"（第 {row.get('page_number')} 页）"
                    )
    finally:
        if graph_client is not None:
            graph_client.close()
        conn.close()

    print("\n演示结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
