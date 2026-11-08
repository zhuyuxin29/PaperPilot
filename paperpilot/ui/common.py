"""UI 公共常量、格式化与共享辅助函数。"""

from __future__ import annotations

import copy
import base64
import html
import io
import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import streamlit as st
import pypdfium2 as pdfium

from paperpilot import storage, services
from paperpilot.agent.runner import AgentResult, run_agent
from paperpilot.client import ApiError, PaperPilotClient
from paperpilot.config import Settings, load_settings
from paperpilot.conversation import rewrite_followup
from paperpilot.evidence.registry import EvidenceRecord
from paperpilot.graph.builder import build_paper_graph
from paperpilot.graph.client import WORKSPACE_ID, GraphError, Neo4jClient
from paperpilot.ingestion import pipeline
from paperpilot.llm import create_chat_provider, create_embedding_provider
from paperpilot.schemas import SUMMARY_KEYS, PaperState, Summary
from paperpilot.summary.generator import (
    SUMMARY_LABELS,
    current_summary_model,
    generate_summary,
    load_summary,
    summary_path,
)
from paperpilot.tools import build_registry

STATE_LABELS: dict[PaperState, str] = {
    PaperState.UPLOADED: "待处理",
    PaperState.PARSING: "处理中",
    PaperState.READY: "可检索",
    PaperState.PARTIAL: "部分可检索",
    PaperState.FAILED: "失败",
}

GRAPH_LABELS = {
    "ready": "就绪",
    "partial": "部分",
    "failed": "失败",
    "building": "构建中",
}

TOOL_LABELS = {
    "search_chunks": "检索文献",
    "read_passages": "读取原文",
    "query_graph": "查询图谱",
    "get_key_points": "读取要点",
    "citation_repair": "引用修正",
}

STOP_LABELS = {
    "BUDGET_STEPS": "达到决策轮次上限，已停止",
    "BUDGET_TOOL_CALLS": "达到工具调用上限，已停止",
    "TIMEOUT": "执行超时，已停止",
    "REPEATED_CALL": "检测到重复调用，已停止",
    "INVALID_CITATION": "回答引用了不存在的证据，已拒绝发布",
    "EMPTY_ANSWER": "模型没有生成回答",
    "CANCELLED": "已手动停止",
}

SECTION_ICONS = {
    "supported": "✓",
    "not_found": "—",
    "insufficient_evidence": "?",
}

SECTION_STATUS_LABELS = {
    "supported": "有依据",
    "not_found": "论文未报告",
    "insufficient_evidence": "证据不足",
}

SECTION_COLORS = (
    "#1f3a5f",
    "#4b5d3a",
    "#7a4b2a",
    "#5b4a6b",
    "#7a3b3b",
    "#335e6b",
    "#5a5142",
)

_CSS = """
<style>
:root { --pp-ink: #2b2a26; --pp-navy: #1f3a5f; --pp-line: #ded4c2;
        --pp-line-soft: #e8e0d1; --pp-paper: #fbf8f1; --pp-panel: #fffdf8;
        --pp-muted: #7a7263; }
html, body, [data-testid="stAppViewContainer"] {
  background: var(--pp-paper); color: var(--pp-ink);
  font-family: "Noto Serif SC", "Source Han Serif SC", "Songti SC",
               "STSong", "SimSun", serif; }
h1, h2, h3, h4 { font-family: "Noto Serif SC", "Songti SC", "SimSun", serif;
                 color: #1f2a37; letter-spacing: .6px; }
.block-container { padding-top: 3.8rem; max-width: 1280px; }
/* 顶栏保留（侧栏展开按钮在顶栏工具条里），只隐藏 Deploy / 主菜单 / 页脚 */
header[data-testid="stHeader"] { display: block !important; background: transparent;
  box-shadow: none; }
[data-testid="stToolbar"] { display: flex !important; visibility: visible !important; }
[data-testid="stSidebarCollapseButton"] { display: flex !important;
  visibility: visible !important; }
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"],
[data-testid="stStatusWidget"], #MainMenu, footer {
  display: none !important; visibility: hidden !important; }
[data-testid="stSidebar"] { background: #f6f1e6; border-right: 1px solid var(--pp-line); }
[data-testid="stSidebar"] .block-container { padding-top: .9rem; }
.pp-brand { font-family: Georgia, "Times New Roman", serif; font-size: 1.5rem;
            font-weight: 700; color: var(--pp-navy); letter-spacing: 2.5px; }
.pp-sub { font-size: .78rem; color: var(--pp-muted); margin-bottom: .8rem;
          letter-spacing: .6px; }
.pp-status { font-size: .78rem; color: #5a6b4f; margin: .2rem 0 .6rem 0; }
.pp-hint { background: var(--pp-panel); border: 1px solid var(--pp-line);
           border-radius: 6px; padding: .85rem 1rem; color: #4b463c;
           font-size: .9rem; line-height: 1.7; }
.pp-cite-title { font-size: .8rem; color: var(--pp-muted); margin: .4rem 0 .1rem 0; }
.pp-side-title { font-family: "Noto Serif SC", "Songti SC", "SimSun", serif;
                 font-size: 1.02rem; font-weight: 700; color: var(--pp-navy);
                 letter-spacing: 1.5px; margin-bottom: .2rem; }
.pp-pdf-view { height: 72vh; min-height: 420px; overflow: auto;
               background: #f1eadc; border: 1px solid var(--pp-line);
               border-radius: 4px; padding: .45rem; }
.pp-pdf-view img { display: block; margin: 0 auto .3rem; background: #ffffff;
                   box-shadow: 0 1px 3px rgba(43, 42, 38, .12);
                   border-radius: 1px; max-width: none !important; }
.pp-pdf-page-no { text-align: center; font-size: .68rem; color: #9b9284;
                  margin: 0 0 .8rem 0;
                  font-family: Georgia, "Times New Roman", serif; }
div[data-testid="stColumn"]:has(.pp-source-marker) {
  position: sticky; top: 0.3rem; align-self: flex-start; margin-top: -3.3rem;
  background: #f8f4ea; border-left: 1px solid var(--pp-line);
  border-radius: 4px; padding: .7rem .8rem; }
/* 抽屉把手：透明图标按钮，接近原生侧栏的收起样式 */
.st-key-pp_close_source button,
.st-key-pp_open_chat button,
.st-key-pp_open_points button { border: 1px solid transparent !important;
  background: transparent !important; color: #8a8175 !important;
  padding: .1rem .35rem !important; min-height: 1.5rem !important;
  font-size: 1rem !important; line-height: 1 !important; box-shadow: none !important; }
.st-key-pp_close_source button:hover,
.st-key-pp_open_chat button:hover,
.st-key-pp_open_points button:hover { background: #efe8da !important;
  border-radius: 6px !important; color: var(--pp-navy) !important; }
/* 左侧导航 */
[data-testid="stRadio"] label > div:first-child { display: none; }
[data-testid="stRadio"] label { padding: .32rem .45rem; border-radius: 6px;
                                transition: background .15s; font-size: .92rem; }
[data-testid="stRadio"] label:hover { background: #efe8da; }
/* 聊天：隐藏头像，提问靠右、回答靠左 */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarDefault"] { display: none !important; }
[data-testid="stChatMessage"] { background: transparent !important;
  border: none !important; padding: 0 .1rem; gap: 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  display: flex; justify-content: flex-end; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
  background: #efe8da; border: 1px solid #e3d9c7;
  border-radius: 12px 12px 3px 12px; padding: .5rem .8rem;
  max-width: 78%; margin-left: auto; text-align: left; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
  background: var(--pp-panel); border: 1px solid var(--pp-line);
  border-radius: 3px 12px 12px 12px; padding: .55rem .85rem; max-width: 94%; }
[data-testid="stChatInput"] textarea { background: var(--pp-panel);
  border: 1px solid var(--pp-line); border-radius: 6px; }
.pp-relation { background: #f8f4ea; border-left: 3px solid var(--pp-navy);
               border-radius: 4px; padding: .5rem .7rem; margin: .35rem 0;
               font-size: .88rem; }
.pp-card-title { font-size: .82rem; color: var(--pp-muted); }
.pp-doc-title { font-family: "Noto Serif SC", "Source Han Serif SC", "Songti SC",
                "STSong", "SimSun", serif; font-size: 1.7rem; font-weight: 700;
                letter-spacing: 3px; color: #1f2a37; margin: 0 0 .15rem 0; }
.pp-doc-sub { font-family: Georgia, "Times New Roman", serif; font-size: .7rem;
              letter-spacing: 2.4px; text-transform: uppercase; color: #a39a8a;
              margin-bottom: .7rem; }
.pp-doc-meta { display: flex; gap: 1.4rem; flex-wrap: wrap;
               border-top: 1px solid var(--pp-line); border-bottom: 1px solid var(--pp-line);
               padding: .5rem 0; margin: .2rem 0 1.1rem 0;
               font-size: .79rem; color: #6b6457; }
.pp-sec { background: var(--pp-panel); border: 1px solid #e3dac8;
          border-left: 3px solid #1f3a5f; border-radius: 4px;
          padding: .85rem 1.1rem .95rem 1.1rem; margin-bottom: .9rem; }
.pp-sec-head { display: flex; align-items: baseline; gap: .55rem; margin-bottom: .35rem; }
.pp-sec-num { font-family: Georgia, "Times New Roman", serif; font-size: .76rem;
              letter-spacing: 1px; color: #1f3a5f; }
.pp-sec-title { font-family: "Noto Serif SC", "Source Han Serif SC", "Songti SC",
                "STSong", "SimSun", serif; font-size: 1.05rem; font-weight: 700;
                color: #1f2a37; letter-spacing: 1.4px; }
.pp-sec-status { margin-left: auto; font-size: .7rem; color: #6b6457;
                 background: #f2ebdc; border-radius: 2px; padding: .1rem .5rem;
                 letter-spacing: .3px; }
.pp-sec-note { font-size: .82rem; color: #8a8175; margin: .15rem 0 .3rem 0; }
.pp-sec-body p { font-family: "Noto Serif SC", "Source Han Serif SC", "Songti SC",
                 "STSong", "SimSun", serif; font-size: .94rem; line-height: 1.95;
                 color: #2f2c27; text-align: justify; margin: .28rem 0; }
.pp-sec details { margin-top: .5rem; border-top: 1px dashed var(--pp-line);
                  padding-top: .4rem; font-size: .78rem; color: #6b6457; }
.pp-sec summary { cursor: pointer; color: var(--pp-navy); letter-spacing: .4px; }
.pp-sec-pages { font-family: Georgia, "Times New Roman", serif; color: #4b463c;
                margin-top: .35rem; letter-spacing: .8px; }
button[kind="primary"] { border-radius: 4px; background: var(--pp-navy); }
.st-key-pp_width_toggle button { border-radius: 999px !important;
  border: 1px solid var(--pp-line) !important; background: var(--pp-panel) !important;
  color: #6b6457 !important; font-size: .8rem !important; letter-spacing: .5px;
  padding: .15rem 0 !important; box-shadow: none !important; }
.st-key-pp_width_toggle button[kind="primary"] { background: var(--pp-navy) !important;
  border-color: var(--pp-navy) !important; color: #ffffff !important; }
</style>
"""


# ---------------------------------------------------------------- 通用


def format_state(state: PaperState) -> str:
    return STATE_LABELS.get(state, state.value)


def paper_rows(
    conn: sqlite3.Connection, settings: Settings | None = None
) -> list[dict[str, object]]:
    """文献库列表：每行带页数、片段数与图谱状态，供界面表格展示。"""
    if settings is not None and _http_mode(settings):
        try:
            papers = _api_client(settings).list_papers()
        except ApiError:
            return []
        rows: list[dict[str, object]] = []
        for paper in papers:
            state = str(paper.get("state", ""))
            try:
                state_label = format_state(PaperState(state))
            except ValueError:
                state_label = state
            graph_status = str(paper.get("graph_status", "not_built"))
            rows.append(
                {
                    "编号": paper["id"],
                    "标题": paper["title"],
                    "状态": state_label,
                    "页数": paper["page_count"],
                    "片段数": paper["chunk_count"],
                    "图谱": (
                        "未构建"
                        if graph_status == "not_built"
                        else GRAPH_LABELS.get(graph_status, graph_status)
                    ),
                    "文件名": paper["file_name"],
                }
            )
        return rows

    rows = []
    for paper in storage.list_papers(conn):
        pages = storage.get_pages(conn, paper.id, paper.revision)
        chunks = storage.get_chunks(conn, paper.id, paper.revision)
        graph_state = storage.get_graph_state(conn, paper.id)
        rows.append(
            {
                "编号": paper.id,
                "标题": paper.title,
                "状态": format_state(paper.state),
                "页数": len(pages),
                "片段数": len(chunks),
                "图谱": (
                    GRAPH_LABELS.get(
                        str(graph_state["status"]), str(graph_state["status"])
                    )
                    if graph_state
                    else "未构建"
                ),
                "文件名": paper.file_name,
            }
        )
    return rows


def _paper_info(
    settings: Settings, conn: sqlite3.Connection, paper_id: str
) -> dict | None:
    if _http_mode(settings):
        try:
            paper = _api_client(settings).get_paper(paper_id)
        except ApiError:
            return None
        return {
            "id": paper["id"],
            "title": paper["title"],
            "file_name": paper["file_name"],
            "revision": paper["revision"],
            "page_count": paper["page_count"],
            "state": paper["state"],
        }
    paper = storage.get_paper(conn, paper_id)
    if paper is None:
        return None
    pages = storage.get_pages(conn, paper_id, paper.revision)
    return {
        "id": paper.id,
        "title": paper.title,
        "file_name": paper.file_name,
        "revision": paper.revision,
        "page_count": len(pages),
        "state": paper.state.value,
    }


def _paper_status_line(
    conn: sqlite3.Connection,
    paper_id: str,
    settings: Settings | None = None,
) -> str:
    rows = paper_rows(conn, settings)
    row = next((item for item in rows if item["编号"] == paper_id), None)
    return str(row["状态"]) if row else "未知"


def _pdf_bytes(settings: Settings, paper_id: str) -> bytes | None:
    path = settings.paper_dir(paper_id) / "source.pdf"
    return path.read_bytes() if path.is_file() else None


def _render_pdf_button(
    settings: Settings, paper_id: str, label: str = "下载原文 PDF", key: str = ""
) -> None:
    if _http_mode(settings):
        try:
            data = _api_client(settings).pdf_bytes(paper_id)
        except ApiError:
            return
    else:
        data = _pdf_bytes(settings, paper_id)
    if data is None:
        return
    st.download_button(
        label,
        data=data,
        file_name=f"{paper_id}.pdf",
        mime="application/pdf",
        key=key or f"pdf_{paper_id}_{uuid.uuid4().hex[:6]}",
    )


def _purge_paper(
    settings: Settings, conn: sqlite3.Connection, paper_id: str
) -> None:
    """删除一篇论文的全部本地数据与图谱节点。"""
    from paperpilot.maintenance import purge_paper
    purge_paper(settings, conn, paper_id)
    chats = st.session_state.get("chats", {})
    chats.pop(paper_id, None)


def _api_client(settings: Settings) -> PaperPilotClient:
    return PaperPilotClient(settings.api_url)


def _http_mode(settings: Settings) -> bool:
    return settings.api_mode == "http"


def _chats_path(settings: Settings, paper_id: str) -> Path:
    return settings.data_dir / "chats" / f"{paper_id}.json"


def _load_chat_history(settings: Settings, paper_id: str) -> list[dict]:
    from paperpilot.history import load_history
    return load_history(settings, paper_id)


def _save_chat_history(
    settings: Settings, paper_id: str, history: list[dict]
) -> None:
    path = _chats_path(settings, paper_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(history[-50:], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _renumber_citations(
    answer: str, citations: list[dict]
) -> tuple[str, dict[str, int]]:
    """把答案里的 [ev_xxx] 按引用出现顺序改写成 [1] [2]…"""
    mapping = {
        citation["evidence_id"]: index
        for index, citation in enumerate(citations, start=1)
    }
    text = answer or ""
    for evidence_id, number in mapping.items():
        text = text.replace(f"[{evidence_id}]", f"[{number}]")
    return text, mapping


def _summary_to_markdown(summary: Summary, paper_title: str) -> str:
    lines = [f"# {paper_title}", "", "## 七类阅读要点", ""]
    lines.append(
        f"> 覆盖：{summary.coverage.status} · "
        f"{summary.coverage.processed_units}/{summary.coverage.total_units} 片段"
    )
    lines.append("")
    for key in SUMMARY_KEYS:
        section = summary.sections[key]
        lines.append(f"### {SUMMARY_LABELS[key]}（{SECTION_STATUS_LABELS[section.status.value]}）")
        if section.note:
            lines.append("")
            lines.append(f"> {section.note}")
        for item in section.items:
            refs = " ".join(f"[{value}]" for value in item.evidence_ids)
            lines.append("")
            lines.append(f"- {item.text} {refs}".rstrip())
        lines.append("")
    lines.append("---")
    lines.append("由 PaperPilot 生成；引用编号对应生成时回读的原文片段。")
    return "\n".join(lines)


# ---------------------------------------------------------------- 入口

