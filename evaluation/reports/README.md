# 评测报告归档

本目录保存 2026-11-07 的 22 题 A/B/C/D 四组评测报告（Markdown 汇总），
用于给 README 的指标表提供可追溯的原始记录。

- 数据集：`evaluation/datasets/eval_all.jsonl`（22 题人工确认集，含 3 道无答案题）
- 原始逐题 JSON、review.csv 与运行设置仍保存在本机 `evaluation/results/`，
  该目录按 `.gitignore` 不提交（体积与隐私原因）；需要逐题复核可重跑评测命令。
- 说明：这批报告生成时，`RETRIEVAL_TERM_BOOST` 的旧实现（规则词面加权）默认开启；
  当前代码默认 `RETRIEVAL_TERM_BOOST=0`（纯 RRF），重跑后指标可能变化。
- 指标仍是辅助性质：Recall/Hit/依据覆盖/拒答率可自动计算，
  答案正确性与引用是否支持结论仍需人工复核。

复现：

```powershell
python -m evaluation.run --dataset evaluation/datasets/eval_all.jsonl --config A --top-k 5
python -m evaluation.run --dataset evaluation/datasets/eval_all.jsonl --config B --top-k 5
python -m evaluation.run --dataset evaluation/datasets/eval_all.jsonl --config C
python -m evaluation.run --dataset evaluation/datasets/eval_all.jsonl --config D
```
