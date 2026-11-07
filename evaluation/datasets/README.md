# 评测数据集说明

## 格式

JSONL，每行一道题：

```json
{
  "question_id": "q_001",
  "paper_ids": ["paper_001"],
  "question": "这篇论文的主要方法是什么？",
  "question_type": "method",
  "answerable": true,
  "gold_chunk_ids": ["paper_001:r1:p3:c0002:v1:xxxxxxxx"],
  "gold_evidence_groups": [["paper_001:r1:p3:c0002:v1:xxxxxxxx"]],
  "reference_note": "方法章节第 3 页"
}
```

- `question_type`：`fact / term / method / relation / unanswerable`
- `gold_chunk_ids`：人工确认能支撑答案的片段；
- `gold_evidence_groups`：可替代证据组（同一事实的多个等价片段放一组，命中任一片段即算覆盖）；
- `unanswerable` 题目的 `answerable` 设为 `false`，`gold_chunk_ids` 留空。

## 标注步骤

1. 先跑 `python -m evaluation.annotate --paper-id paper_001 --grep 关键词` 找到候选片段；
2. 打开对应 PDF 核对，再把片段 ID 填进数据集；
3. 不要用模型输出反填标准答案。

## 规模建议

- 开发题 10 道（调参用）；
- 保留测试题 20 道（比较前冻结，不参与调参）；
- 题型覆盖：事实定位、术语解释、方法理解、关联查证、无法回答。

## 运行

```powershell
python -m evaluation.run --dataset evaluation/datasets/demo.jsonl --config A
python -m evaluation.run --dataset evaluation/datasets/demo.jsonl --config B
python -m evaluation.run --dataset evaluation/datasets/demo.jsonl --config C
python -m evaluation.run --dataset evaluation/datasets/demo.jsonl --config D
```

结果写入 `evaluation/results/`，同时生成 Markdown 报告。
