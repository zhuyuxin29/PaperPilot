from __future__ import annotations


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(result: dict) -> str:
    kind = result.get("kind")
    summary = result.get("summary", {})
    lines = [f"# 评测报告 · 配置 {result.get('config')}", ""]

    if kind == "retrieval":
        lines += [
            "## 汇总",
            "",
            "| 指标 | 值 |",
            "|---|---|",
            f"| 题目数 | {summary.get('questions')} |",
            f"| 有标注题目 | {summary.get('with_gold')} |",
            f"| Recall@K | {_fmt(summary.get('recall_at_k'))} |",
            f"| Hit@K | {_fmt(summary.get('hit_at_k'))} |",
            f"| 证据组覆盖 | {_fmt(summary.get('group_coverage'))} |",
            f"| 图谱参与题目 | {summary.get('graph_used')} |",
            f"| 错误数 | {summary.get('errors')} |",
            "",
            "## 逐题结果",
            "",
            "| 题目 | 类型 | Recall@K | Hit@K | 组覆盖 | 图谱 | 错误 |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in result["rows"]:
            lines.append(
                f"| {row['question_id']} | {row['question_type']} | "
                f"{_fmt(row['recall_at_k'])} | {_fmt(row['hit_at_k'])} | "
                f"{_fmt(row['group_coverage'])} | "
                f"{'是' if row['graph_used'] else '否'} | "
                f"{row['error'] or '-'} |"
            )
    else:
        lines += [
            "## 汇总",
            "",
            "| 指标 | 值 |",
            "|---|---|",
            f"| 题目数 | {summary.get('questions')} |",
            f"| 完成率 | {_fmt(summary.get('completion_rate'))} |",
            f"| 平均轮次 | {_fmt(summary.get('avg_steps'))} |",
            f"| 平均工具调用 | {_fmt(summary.get('avg_tool_calls'))} |",
            f"| 可答题带引用比例 | {_fmt(summary.get('citation_question_rate'))} |",
            f"| 无答案题拒答率 | {_fmt(summary.get('refusal_rate_unanswerable'))} |",
            f"| 标准证据覆盖率 | {_fmt(summary.get('gold_covered_rate'))} |",
            f"| 引用修正题数 | {summary.get('repaired_questions', 0)} |",
            f"| 平均耗时(ms) | {_fmt(summary.get('avg_latency_ms'))} |",
            f"| 错误数 | {summary.get('errors')} |",
            "",
            "## 逐题结果",
            "",
            "| 题目 | 类型 | 状态 | 工具调用 | 引用数 | 拒答 | 错误 |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in result["rows"]:
            lines.append(
                f"| {row['question_id']} | {row['question_type']} | "
                f"{row['status'] or '-'} | {row['tool_calls']} | "
                f"{len(row['citations'])} | "
                f"{'是' if row['refused'] else '否'} | {row['error'] or '-'} |"
            )

    lines += [
        "",
        "> 说明：Recall@K / Hit@K / 证据组覆盖只对人工标注了 gold 片段的题目计算；",
        "> 答案内容是否正确、引用是否支持结论，请结合原始结果 JSON 人工复核。",
        "",
    ]
    return "\n".join(lines)
