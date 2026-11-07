"""Explicit final-answer fixtures used by scripted chat providers."""
import json
import re


def final_json(text):
    if not text.strip():
        return text
    if text in (
        "当前文献中未找到相关证据。", "当前文献中未找到可支持该结论的证据。",
        "无法执行该操作。",
    ):
        return json.dumps({"status": "insufficient_evidence", "claims": []})
    ids = re.findall(r"ev_\d+", text)
    claim = re.sub(r"\[ev_\d+\]", "", text).strip()
    return json.dumps({"status": "answered", "claims": [{"text": claim, "evidence_ids": ids}]}, ensure_ascii=False)
