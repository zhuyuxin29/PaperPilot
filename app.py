"""PaperPilot 界面：左侧导航 + 右侧工作区。

结构：
- 侧边栏：品牌、导入论文、当前论文、页面导航、设置；
- 文献问答：聊天气泡、引用展开、查阅过程、图谱关系、停止执行；
- 文献要点：纵向折叠卡片、查看来源、导出 JSON/Markdown；
- 我的文献：上传、选择、重新处理、删除。

边界：
- 界面只做展示与调用，业务规则在 ``paperpilot`` 包里；
- Agent 在后台线程运行，使用独立数据库连接，不阻塞界面；
- 密钥只在本次会话的密码输入框中使用，不写入日志或页面明文。
"""

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


def main() -> None:
    st.set_page_config(
        page_title="PaperPilot",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    settings = load_settings()
    try:
        settings.ensure_dirs()
        conn = storage.connect(settings.db_path)
        storage.ensure_schema(conn)
    except Exception as exc:
        st.error(f"数据目录不可用：{exc}")
        return

    try:
        with st.sidebar:
            st.markdown(
                '<div class="pp-brand">PaperPilot</div>'
                '<div class="pp-sub">你的文献阅读助手</div>',
                unsafe_allow_html=True,
            )

        nav = st.navigation(PAGES, position="hidden")

        with st.sidebar:
            for page, label, icon in (
                (CHAT_PAGE, "文献问答", "💬"),
                (POINTS_PAGE, "文献要点", "📄"),
                (LIBRARY_PAGE, "我的文献", "📚"),
            ):
                st.page_link(page, label=label, icon=icon)
            st.divider()
            _render_import_popover(settings, conn)
            _render_current_paper(settings, conn)
            st.divider()
            _render_settings(settings)

        nav.run()
    finally:
        conn.close()


# ---------------------------------------------------------------- 侧边栏


def _render_current_paper(
    settings: Settings, conn: sqlite3.Connection
) -> None:
    rows = paper_rows(conn, settings)
    paper_ids = [str(row["编号"]) for row in rows]
    if not paper_ids:
        st.caption("还没有文献")
        return
    titles = {str(row["编号"]): str(row["标题"]) for row in rows}
    if st.session_state.get("current_paper") not in paper_ids:
        st.session_state["current_paper"] = paper_ids[0]
    index = paper_ids.index(st.session_state["current_paper"])
    current = st.selectbox(
        "当前论文",
        paper_ids,
        index=index,
        format_func=lambda pid: f"{titles[pid][:24]}",
        key="paper_selector",
    )
    if current != st.session_state.get("current_paper"):
        st.session_state["current_paper"] = current
    st.markdown(
        f'<div class="pp-status">● {_paper_status_line(conn, current, settings)}</div>',
        unsafe_allow_html=True,
    )


def _render_import_popover(
    settings: Settings, conn: sqlite3.Connection
) -> None:
    with st.popover("＋ 导入论文", width="stretch"):
        uploaded = st.file_uploader(
            "选择 PDF 文件", type=["pdf"], key="sidebar_upload"
        )
        if uploaded is not None:
            st.caption(f"{uploaded.name} · {len(uploaded.getvalue())} 字节")
            if st.button("开始入库", type="primary", key="sidebar_ingest"):
                if _http_mode(settings):
                    try:
                        paper = _api_client(settings).upload_paper(
                            uploaded.getvalue(), uploaded.name
                        )
                    except ApiError as exc:
                        st.error(f"入库失败：{exc.message}")
                    else:
                        st.session_state["current_paper"] = paper["id"]
                        st.success(f"入库完成：{paper['id']}")
                        st.switch_page(CHAT_PAGE)
                else:
                    try:
                        result = services.prepare_paper(
                            uploaded.getvalue(), uploaded.name, settings, conn
                        )
                    except Exception as exc:
                        st.error(f"入库失败：{exc}")
                    else:
                        st.session_state["current_paper"] = result.paper.id
                        st.success(f"入库完成：{result.paper.id}")
                        st.switch_page(CHAT_PAGE)


def _render_settings(settings: Settings) -> None:
    with st.expander("⚙ 设置"):
        st.caption(f"对话模型：{current_summary_model(settings)}")
        st.caption(f"向量模型：{settings.embedding_model or 'hash'}")
        st.caption(f"运行模式：{settings.api_mode} · {settings.api_url}")
        st.caption(f"当前数据目录：{settings.data_dir}")
        st.caption("修改仅本次会话生效；持久化请编辑 .env。")

        mode = st.selectbox(
            "图谱模式",
            ["off", "auto", "on"],
            index=["off", "auto", "on"].index(settings.graph_mode)
            if settings.graph_mode in ("off", "auto", "on")
            else 0,
            key="graph_mode_select",
        )
        if mode != settings.graph_mode:
            settings.graph_mode = mode
            os.environ["GRAPH_MODE"] = mode

        chat_key = st.text_input(
            "CHAT_API_KEY", type="password", key="chat_key_input"
        )
        emb_key = st.text_input(
            "EMBEDDING_API_KEY", type="password", key="emb_key_input"
        )
        if st.button("应用密钥（仅本次会话）", key="apply_keys"):
            if chat_key:
                os.environ["CHAT_API_KEY"] = chat_key
            if emb_key:
                os.environ["EMBEDDING_API_KEY"] = emb_key
            st.success("已应用")
            st.rerun()

        if st.button("检查连接", key="check_conn"):
            fresh = load_settings()
            try:
                create_embedding_provider(fresh).embed(["ping"])
                st.success("向量模型连接正常")
            except Exception as exc:
                st.error(f"向量模型连接失败：{exc}")
            if fresh.graph_mode != "off":
                client = None
                try:
                    client = Neo4jClient.connect(fresh)
                    st.success("Neo4j 连接正常")
                except GraphError as exc:
                    st.error(f"Neo4j 连接失败：{exc.message}")
                finally:
                    if client is not None:
                        client.close()


# ---------------------------------------------------------------- 欢迎页


def _render_welcome() -> None:
    st.title("PaperPilot")
    st.markdown("#### 带证据的文献阅读助手")
    st.markdown(
        "上传 PDF 后，PaperPilot 会自动生成固定七类阅读要点，并允许你在文献内提问。"
        "每个结论都能展开查看原文片段与页码，没有依据的内容会如实标注。"
    )
    st.info("用左侧「＋ 导入论文」上传第一篇 PDF，然后从左侧选择文献开始问答。")
    st.markdown(
        '<div class="pp-hint">'
        "<b>三步上手</b><br>"
        "1. 左侧「＋ 导入论文」上传 PDF（自动解析、分块、建索引）<br>"
        "2. 在「文献问答」里直接提问，回答下方可展开引用原文<br>"
        "3. 在「文献要点」里生成七类要点，缺失内容会标注证据不足"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 文献问答


def _render_chat_page(
    settings: Settings, conn: sqlite3.Connection, paper_id: str
) -> None:
    paper = _paper_info(settings, conn, paper_id)
    if paper is None:
        st.error("文献不存在")
        return
    show_source = st.session_state.setdefault("show_source", False)
    head_cols = st.columns([6, 1])
    head_cols[0].subheader("文献问答")
    if not show_source:
        with head_cols[1]:
            with st.container(key="pp_open_chat"):
                if st.button("» 原文", help="展开原文"):
                    st.session_state["show_source"] = True
                    st.rerun()
    if show_source:
        width = int(st.session_state.get("source_width", 36))
        main_col, source_col = st.columns([100 - width, width], gap="large")
    else:
        main_col, source_col = st.container(), None
    with main_col:
        _poll_pending(settings, conn)

        chats = st.session_state.setdefault("chats", {})
        if paper_id not in chats or not st.session_state.get("pending"):
            chats[paper_id] = _load_chat_history(settings, paper_id)
        history: list[dict] = chats[paper_id]
        if not history:
            st.markdown(
                '<div class="pp-hint">直接输入问题，例如：'
                "“这篇论文采用了什么方法？”“哪些实验结果支持作者结论？”</div>",
                unsafe_allow_html=True,
            )
        for turn in history:
            _render_turn(settings, conn, paper_id, turn)

        if any(t.get("status") == "RUNNING" for t in history):
            st.info("之前的任务仍在执行，结果会保存在任务日志中。")
            if st.button("刷新任务记录"):
                st.rerun()
        pending = st.session_state.get("pending")
        if pending and pending.get("paper_id") == paper_id:
            if pending["thread"].is_alive():
                label = (
                    "正在请求后端…"
                    if _http_mode(settings)
                    else "正在检索与取证…"
                )
                with st.status(label, expanded=True) as status:
                    if _http_mode(settings):
                        st.write("API 模式：等待后端完成本次问答。")
                    else:
                        st.write("Agent 正在检索、回读原文并校验引用。")
                        if st.button("停止", key=f"stop_{paper_id}"):
                            pending["cancel"].set()
                            status.update(label="正在停止…", state="running")
                            st.rerun()
                time.sleep(0.4)
                st.rerun()
            else:
                _finalize_pending(settings, conn)
                st.rerun()

        question = st.chat_input(
            "输入你对这篇论文的问题……", key=f"chat_input_{paper_id}",
            disabled=bool(st.session_state.get("pending")) or any(t.get("status") == "RUNNING" for t in history)
        )
        if question:
            _start_pending(settings, paper_id, question.strip())
            st.rerun()

    if source_col is not None:
        with source_col:
            _render_source_panel(settings, conn, paper_id)


@st.cache_data(show_spinner=False)
def _render_pdf_page(
    pdf_path: str, page_index: int, scale: float = 1.6
) -> bytes | None:
    """把 PDF 指定页渲染成 PNG，供右侧原文栏显示。"""
    try:
        document = pdfium.PdfDocument(pdf_path)
        page = document[page_index]
        image = page.render(scale=scale).to_pil()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        document.close()
        return buffer.getvalue()
    except Exception:
        return None


def _render_source_panel(
    settings: Settings, conn: sqlite3.Connection, paper_id: str
) -> None:
    """右侧原文栏：整篇 PDF 连续滚动 + 缩放 + 收起。"""
    info = _paper_info(settings, conn, paper_id)
    if info is None:
        return
    st.markdown(
        '<div class="pp-source-marker"></div>', unsafe_allow_html=True
    )
    head = st.columns([4, 1])
    head[0].markdown(
        '<div class="pp-side-title">源文献</div>', unsafe_allow_html=True
    )
    with head[1]:
        with st.container(key="pp_close_source"):
            if st.button("«", help="收起原文"):
                st.session_state["show_source"] = False
                st.rerun()
    st.caption(info["file_name"])
    current_label = st.session_state.get("source_width_label", "标准")
    with st.container(key="pp_width_toggle"):
        width_cols = st.columns(3)
        for col, label, width in zip(
            width_cols, ("窄", "标准", "宽"), (30, 40, 50)
        ):
            if col.button(
                label,
                type="primary" if current_label == label else "secondary",
                key=f"source_width_{paper_id}_{label}",
                width="stretch",
            ):
                st.session_state["source_width_label"] = label
                st.session_state["source_width"] = width
                st.rerun()
    page_count = int(info["page_count"])
    if not page_count:
        st.info("没有可显示的页面。")
        return
    if _http_mode(settings):
        try:
            data = _api_client(settings).pdf_bytes(paper_id)
        except ApiError:
            st.info("无法从后端获取 PDF。")
            return
        tmp_dir = settings.data_dir / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = str(tmp_dir / f"api_{paper_id}.pdf")
        with open(pdf_path, "wb") as handle:
            handle.write(data)
    else:
        path = settings.paper_dir(paper_id) / "source.pdf"
        if not path.is_file():
            st.info("找不到原始 PDF 文件。")
            return
        pdf_path = str(path)

    zoom = int(st.session_state.get(f"source_zoom_{paper_id}", 100))

    parts = ['<div class="pp-pdf-view">']
    for index in range(page_count):
        image = _render_pdf_page(pdf_path, index)
        if image is None:
            continue
        encoded = base64.b64encode(image).decode("ascii")
        parts.append(
            f'<img src="data:image/png;base64,{encoded}" '
            f'style="width:{zoom}%" alt="第 {index + 1} 页">'
        )
        parts.append(
            f'<div class="pp-pdf-page-no">第 {index + 1} 页</div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
    _render_pdf_button(
        settings, paper_id, label="下载 PDF", key=f"pdf_side_{paper_id}"
    )
    st.divider()
    st.select_slider(
        "缩放",
        options=[75, 100, 125, 150],
        value=int(st.session_state.get(f"source_zoom_{paper_id}", 100)),
        key=f"source_zoom_{paper_id}",
    )


def _start_pending(settings: Settings, paper_id: str, question: str) -> None:
    if st.session_state.get("pending") is not None:
        st.warning("已有任务尚未结束，请等待完成或停止后再提交。")
        return
    cancel = threading.Event()
    holder: dict = {}
    settings_snapshot = copy.deepcopy(settings)
    history = st.session_state.get("chats", {}).get(paper_id, [])[-3:]

    def worker() -> None:
        try:
            holder["result"] = _run_question(
                settings_snapshot, paper_id, question, cancel, history
            )
        except Exception as exc:  # 网络/配置错误通过界面反馈
            holder["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    st.session_state["pending"] = {
        "thread": thread,
        "cancel": cancel,
        "holder": holder,
        "paper_id": paper_id,
        "question": question,
        "started_at": time.time(),
    }
    thread.start()


def _poll_pending(settings: Settings, conn: sqlite3.Connection) -> None:
    pending = st.session_state.get("pending")
    if not pending or pending["thread"].is_alive():
        return
    _finalize_pending(settings, conn)


def _finalize_pending(
    settings: Settings, conn: sqlite3.Connection
) -> None:
    pending = st.session_state.pop("pending", None)
    if not pending:
        return
    paper_id = pending["paper_id"]
    question = pending["question"]
    holder = pending["holder"]
    chats = st.session_state.setdefault("chats", {})
    history = chats.setdefault(paper_id, [])

    if "error" in holder:
        history.append(
            {
                "id": uuid.uuid4().hex[:8],
                "question": question,
                "answer": "",
                "status": "FAILED",
                "stop_reason": "CLIENT_ERROR",
                "citations": [],
                "trace": [],
                "graph_claims": [],
                "invalid_citations": [],
                "error": str(holder["error"]),
            }
        )
        _save_chat_history(settings, paper_id, history)
        return

    result: AgentResult = holder["result"]
    citations = [
        {
            "evidence_id": record.evidence_id,
            "chunk_id": record.chunk_id,
            "page_number": record.page_number,
            "quote": record.quote,
        }
        for record in result.citations
    ]
    graph_claims: list[dict] = []
    for entry in result.trace:
        claims = entry.get("graph_claims")
        if claims:
            graph_claims.extend(claims)
    history.append(
        {
            "id": uuid.uuid4().hex[:8],
            "question": question,
            "answer": result.answer,
            "run_id": result.run_id,
            "revision": storage.get_paper(conn, paper_id).revision if storage.get_paper(conn, paper_id) else None,
            "status": result.status,
            "stop_reason": result.stop_reason,
            "citations": citations,
            "trace": [dict(entry) for entry in result.trace],
            "graph_claims": graph_claims,
            "invalid_citations": list(result.invalid_citations),
            "repair_attempts": result.repair_attempts,
            "error": None,
        }
    )
    _save_chat_history(settings, paper_id, history)


def _run_question(
    settings: Settings,
    paper_id: str,
    question: str,
    cancel: threading.Event,
    history: list[dict] | None = None,
) -> AgentResult:
    history = history or []
    if _http_mode(settings):
        client = _api_client(settings)
        resolved = question
        if history:
            try:
                resolved = rewrite_followup(
                    create_chat_provider(settings), question, history
                )
            except Exception:
                resolved = question
        data = client.chat(paper_id, resolved)
        revision = next(
            (
                paper["revision"]
                for paper in client.list_papers()
                if paper["id"] == paper_id
            ),
            1,
        )
        citations = tuple(
            EvidenceRecord(
                evidence_id=item["evidence_id"],
                chunk_id=item["chunk_id"],
                paper_id=paper_id,
                revision=revision,
                page_number=item["page_number"],
                quote=item.get("quote", ""),
            )
            for item in data.get("citations", [])
        )
        return AgentResult(
            answer=data.get("answer", ""),
            run_id=data.get("run_id"), answer_kind=data.get("answer_kind"),
            citations=citations,
            status=data.get("status", "FAILED"),
            stop_reason=data.get("stop_reason"),
            trace=tuple(data.get("trace", [])),
            steps=int(data.get("steps", 0)),
            tool_calls=int(data.get("tool_calls", 0)),
            usage={},
            invalid_citations=tuple(data.get("invalid_citations", [])),
            repair_attempts=int(data.get("repair_attempts", 0)),
        )

    conn = storage.connect(settings.db_path)
    storage.ensure_schema(conn)
    graph_client = None
    try:
        embedding = create_embedding_provider(settings)
        chat = create_chat_provider(settings)
        paper = storage.get_paper(conn, paper_id)
        history = [turn for turn in history if turn.get("status") == "COMPLETED"
                   and turn.get("revision") == (paper.revision if paper else None)]
        resolved = (
            rewrite_followup(chat, question, history) if history else question
        )
        if settings.graph_mode != "off":
            try:
                graph_client = Neo4jClient.connect(settings)
            except GraphError:
                if settings.graph_mode == "on":
                    raise
                graph_client = None
        return run_agent(
            question=resolved,
            paper_ids=(paper_id,),
            conn=conn,
            settings=settings,
            embedding=embedding,
            chat=chat,
            tools=build_registry(graph_enabled=graph_client is not None and services.graph_ready(conn, (paper_id,))),
            graph=graph_client,
            cancel_event=cancel,
        )
    finally:
        if graph_client is not None:
            graph_client.close()
        conn.close()


def _render_turn(
    settings: Settings,
    conn: sqlite3.Connection,
    paper_id: str,
    turn: dict,
) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        status = turn.get("status")
        if status and status != "COMPLETED":
            reason = turn.get("stop_reason") or status
            st.warning(STOP_LABELS.get(reason, f"本次执行未完成：{reason}"))
            if st.button("重试这个问题", key=f"retry_{turn['id']}", disabled=bool(st.session_state.get("pending")) or status == "RUNNING"):
                _start_pending(settings, paper_id, turn["question"])
                st.rerun()
        answer = turn.get("answer") or ""
        citations = turn.get("citations") or []
        display, _ = _renumber_citations(answer, citations)
        if display and status == "COMPLETED":
            st.markdown(display)
        elif not turn.get("error"):
            st.markdown("（没有生成回答）")
        if turn.get("invalid_citations"):
            st.error(
                "回答引用了未登记的证据："
                + "、".join(turn["invalid_citations"])
            )
        if turn.get("error"):
            st.error(f"请求失败：{turn['error']}")
        graph_claims = turn.get("graph_claims") or []
        if citations or graph_claims:
            _render_citations(
                settings,
                paper_id,
                turn["id"],
                citations,
                graph_claims=graph_claims,
            )
        _render_trace(turn.get("trace") or [])


def _render_citations(
    settings: Settings,
    paper_id: str,
    turn_id: str,
    citations: list[dict],
    graph_claims: list[dict] | None = None,
) -> None:
    st.divider()
    with st.expander(f"引用与依据（{len(citations)} 条）", expanded=False):
        for index, citation in enumerate(citations, start=1):
            st.markdown(f"**[{index}] 第 {citation['page_number']} 页**")
        if graph_claims:
            st.markdown("**图谱关系**")
            for claim in graph_claims[:10]:
                st.markdown(
                    f"- {claim.get('subject', '?')} → "
                    f"{claim.get('object', '?')}"
                    f"（第 {claim.get('page_number', '?')} 页）"
                )
        _render_pdf_button(
            settings, paper_id, label="下载原文 PDF", key=f"pdf_{turn_id}"
        )



def _render_trace(trace: list[dict]) -> None:
    if not trace:
        return
    with st.expander("▸ 本次查阅过程"):
        labels = [
            TOOL_LABELS.get(entry.get("tool", ""), entry.get("tool", ""))
            for entry in trace
        ]
        st.caption(" → ".join(labels))
        rows = []
        for entry in trace:
            tool = entry.get("tool", "")
            detail = ""
            if entry.get("evidence_ids"):
                detail = f"{len(entry['evidence_ids'])} 条证据"
            elif entry.get("graph_claims"):
                detail = f"{len(entry['graph_claims'])} 条关系"
            rows.append(
                {
                    "步骤": entry.get("step", ""),
                    "动作": TOOL_LABELS.get(tool, tool),
                    "状态": entry.get("status", ""),
                    "结果": detail,
                    "耗时(ms)": entry.get("duration_ms", ""),
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")
        for entry in trace:
            error = entry.get("error")
            if error:
                st.caption(
                    f"⚠ {TOOL_LABELS.get(entry.get('tool', ''), entry.get('tool', ''))}："
                    f"{error.get('message', '')}"
                )


# ---------------------------------------------------------------- 文献要点


def _render_points_page(
    settings: Settings, conn: sqlite3.Connection, paper_id: str
) -> None:
    paper = _paper_info(settings, conn, paper_id)
    if paper is None:
        st.error("文献不存在")
        return
    show_source = st.session_state.setdefault("show_source", False)
    if show_source:
        width = int(st.session_state.get("source_width", 36))
        main_col, source_col = st.columns([100 - width, width], gap="large")
    else:
        main_col, source_col = st.container(), None
    with main_col:
        summary = _load_summary(settings, paper_id, paper["revision"])
        header_cols = st.columns([6, 2, 2, 2])
        header_cols[0].markdown(
            '<div class="pp-doc-title">七类文献要点</div>'
            '<div class="pp-doc-sub">Seven-Category Reading Notes</div>',
            unsafe_allow_html=True,
        )
        if header_cols[1].button(
            "生成 / 刷新要点", type="primary", key="gen_summary"
        ):
            _generate_summary(settings, conn, paper_id)
            summary = _load_summary(settings, paper_id, paper["revision"])

        if summary is not None:
            with header_cols[2].popover("导出"):
                st.download_button(
                    "下载 JSON",
                    data=json.dumps(
                        summary.model_dump(mode="json"),
                        ensure_ascii=False,
                        indent=2,
                    ).encode("utf-8"),
                    file_name=f"{paper_id}_summary.json",
                    mime="application/json",
                    key="export_summary_json",
                )
                st.download_button(
                    "下载 Markdown",
                    data=_summary_to_markdown(summary, paper["title"]).encode(
                        "utf-8"
                    ),
                    file_name=f"{paper_id}_summary.md",
                    mime="text/markdown",
                    key="export_summary_md",
                )

        if not show_source:
            with header_cols[3]:
                with st.container(key="pp_open_points"):
                    if st.button("» 原文", help="展开原文"):
                        st.session_state["show_source"] = True
                        st.rerun()

        if summary is None:
            st.markdown(
                '<div class="pp-hint">还没有生成要点，点击右上角'
                "「生成 / 刷新要点」开始。</div>",
                unsafe_allow_html=True,
            )
        else:
            coverage = summary.coverage
            status_text = "完整" if coverage.status == "complete" else "部分完成"
            unparsed = (
                f" · 未解析页：{', '.join(map(str, coverage.unparsed_pages))}"
                if coverage.unparsed_pages
                else ""
            )
            st.markdown(
                '<div class="pp-doc-meta">'
                f"<span>论文：{html.escape(paper['file_name'])}</span>"
                f"<span>覆盖：{status_text} · {coverage.processed_units}/"
                f"{coverage.total_units} 片段{unparsed}</span>"
                f"<span>生成模型：{html.escape(current_summary_model(settings))}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            if _http_mode(settings):
                client = _api_client(settings)
                chunks = {}
                for chunk_id in set(summary.evidence_map.values()):
                    try:
                        data = client.get_chunk(paper_id, chunk_id)
                    except ApiError:
                        continue
                    chunks[chunk_id] = SimpleNamespace(
                        page_number=data["page_number"]
                    )
            else:
                chunks = {
                    chunk.chunk_id: chunk
                    for chunk in storage.get_chunks(conn, paper_id)
                }
            for index, key in enumerate(SUMMARY_KEYS, start=1):
                _render_summary_section(index, key, summary, chunks)

    if source_col is not None:
        with source_col:
            _render_source_panel(settings, conn, paper_id)


def _section_pages(
    section, summary: Summary, chunks: dict
) -> list[int]:
    pages: set[int] = set()
    for item in section.items:
        for evidence_id in item.evidence_ids:
            chunk = chunks.get(summary.evidence_map.get(evidence_id, ""))
            if chunk is not None:
                pages.add(chunk.page_number)
    return sorted(pages)


def _render_summary_section(
    index: int, key: str, summary: Summary, chunks: dict
) -> None:
    section = summary.sections[key]
    status = section.status.value
    accent = SECTION_COLORS[(index - 1) % len(SECTION_COLORS)]
    paragraphs = "".join(
        f"<p>{html.escape(item.text).replace(chr(10), '<br>')}</p>"
        for item in section.items
    )
    if not paragraphs:
        paragraphs = '<p class="pp-sec-note">（本节没有可展示的条目）</p>'
    note = (
        f'<div class="pp-sec-note">{html.escape(section.note)}</div>'
        if section.note
        else ""
    )
    pages = _section_pages(section, summary, chunks)
    source = ""
    if pages:
        page_text = " · ".join(f"第 {page} 页" for page in pages)
        source = (
            "<details><summary>查看来源</summary>"
            f'<div class="pp-sec-pages">{page_text}</div></details>'
        )
    st.markdown(
        f'<div class="pp-sec" style="border-left-color:{accent}">'
        '<div class="pp-sec-head">'
        f'<span class="pp-sec-num" style="color:{accent}">{index:02d}</span>'
        f'<span class="pp-sec-title">{SUMMARY_LABELS[key]}</span>'
        '<span class="pp-sec-status">'
        f"{SECTION_STATUS_LABELS.get(status, status)}</span>"
        "</div>"
        f'{note}<div class="pp-sec-body">{paragraphs}</div>{source}'
        "</div>",
        unsafe_allow_html=True,
    )


def _load_summary(
    settings: Settings, paper_id: str, revision: int
) -> Summary | None:
    if _http_mode(settings):
        try:
            payload = _api_client(settings).get_summary(paper_id)
        except ApiError:
            return None
        if payload is None:
            return None
        try:
            return Summary.model_validate(payload)
        except Exception:
            return None
    path = summary_path(
        settings, paper_id, revision, current_summary_model(settings)
    )
    if not path.is_file():
        return None
    try:
        return load_summary(settings, path, paper_id=paper_id, revision=revision, model_id=current_summary_model(settings))
    except Exception:
        return None


def _generate_summary(
    settings: Settings, conn: sqlite3.Connection, paper_id: str
) -> None:
    if _http_mode(settings):
        try:
            with st.spinner("正在请求后端生成要点…"):
                _api_client(settings).generate_summary(paper_id, force=True)
        except ApiError as exc:
            st.error(f"生成失败：{exc.message}")
            return
        st.success("要点已生成")
        return

    graph_client = None
    if settings.graph_mode != "off":
        try:
            graph_client = Neo4jClient.connect(settings)
        except GraphError as exc:
            if settings.graph_mode == "on":
                st.error(f"图谱必须可用：{exc.message}")
                return
            graph_client = None
    try:
        with st.spinner("正在阅读全文并生成要点…"):
            result = generate_summary(
                conn,
                settings,
                create_chat_provider(settings),
                paper_id,
                force=True,
                graph=graph_client,
            )
    except Exception as exc:
        st.error(f"生成失败：{exc}")
        return
    finally:
        if graph_client is not None:
            graph_client.close()
    st.success("要点已生成")


# ---------------------------------------------------------------- 我的文献


def _render_library_page(
    settings: Settings, conn: sqlite3.Connection
) -> None:
    st.subheader("我的文献")
    st.caption("上传、选择、重新处理或删除文献；用左上角「＋ 导入论文」上传新文件。")
    rows = paper_rows(conn, settings)
    if not rows:
        st.info("还没有文献。")
        return
    for row in rows:
        paper_id = str(row["编号"])
        with st.container(border=True):
            cols = st.columns([5, 2, 2, 2])
            cols[0].markdown(f"**{row['标题']}**")
            cols[0].caption(
                f"{row['文件名']} · {row['页数']} 页 · {row['片段数']} 片段 · "
                f"图谱：{row['图谱']}"
            )
            cols[1].caption(row["状态"])
            if cols[2].button("打开问答", key=f"open_{paper_id}"):
                st.session_state["current_paper"] = paper_id
                st.switch_page(CHAT_PAGE)
            with cols[3].popover("更多"):
                _render_pdf_button(
                    settings, paper_id, key=f"pdf_lib_{paper_id}"
                )
                if st.button("重新处理", key=f"reprocess_{paper_id}"):
                    if _http_mode(settings):
                        try:
                            _api_client(settings).reprocess_paper(paper_id)
                        except ApiError as exc:
                            st.error(f"重新处理失败：{exc.message}")
                            st.stop()
                        st.success("重新处理完成")
                        st.rerun()
                    else:
                        _reprocess_paper(settings, conn, paper_id)
                confirm = st.checkbox("确认删除", key=f"confirm_{paper_id}")
                if st.button(
                    "删除文献",
                    key=f"delete_{paper_id}",
                    disabled=not confirm,
                ):
                    if _http_mode(settings):
                        try:
                            _api_client(settings).delete_paper(paper_id)
                        except ApiError as exc:
                            st.error(f"删除失败：{exc.message}")
                            st.stop()
                    else:
                        _purge_paper(settings, conn, paper_id)
                    st.success("已删除")
                    st.rerun()


def _reprocess_paper(
    settings: Settings, conn: sqlite3.Connection, paper_id: str
) -> None:
    data = _pdf_bytes(settings, paper_id)
    if data is None:
        st.error("找不到原始 PDF，无法重新处理。")
        return
    paper = storage.get_paper(conn, paper_id)
    file_name = paper.file_name if paper else f"{paper_id}.pdf"
    try:
        result = services.reprocess_paper(settings, conn, paper_id)
    except Exception as exc:
        st.error(f"重新处理失败：{exc}")
        return
    st.session_state["current_paper"] = result.paper.id
    st.success(f"重新处理完成：{result.paper.id} · {result.chunk_count} 个片段")
    st.rerun()


def _open_page_context() -> tuple[Settings, sqlite3.Connection]:
    settings = load_settings()
    settings.ensure_dirs()
    conn = storage.connect(settings.db_path)
    storage.ensure_schema(conn)
    return settings, conn


def _page_chat() -> None:
    settings, conn = _open_page_context()
    try:
        paper_id = st.session_state.get("current_paper")
        if not paper_id or _paper_info(settings, conn, paper_id) is None:
            _render_welcome()
            return
        _render_chat_page(settings, conn, paper_id)
    finally:
        conn.close()


def _page_points() -> None:
    settings, conn = _open_page_context()
    try:
        paper_id = st.session_state.get("current_paper")
        if not paper_id or _paper_info(settings, conn, paper_id) is None:
            _render_welcome()
            return
        _render_points_page(settings, conn, paper_id)
    finally:
        conn.close()


def _page_library() -> None:
    settings, conn = _open_page_context()
    try:
        _render_library_page(settings, conn)
    finally:
        conn.close()


CHAT_PAGE = st.Page(
    _page_chat, title="文献问答", icon="💬", url_path="chat", default=True
)
POINTS_PAGE = st.Page(
    _page_points, title="文献要点", icon="📄", url_path="points"
)
LIBRARY_PAGE = st.Page(
    _page_library, title="我的文献", icon="📚", url_path="library"
)
PAGES = [CHAT_PAGE, POINTS_PAGE, LIBRARY_PAGE]


if __name__ == "__main__":
    # 直接 `python app.py` 时自动启动 Streamlit 并打开浏览器；
    # 被 `streamlit run app.py` 启动时 runtime 已存在，直接执行页面。
    try:
        from streamlit import runtime as _streamlit_runtime

        _running_in_streamlit = _streamlit_runtime.exists()
    except Exception:
        _running_in_streamlit = False

    if not _running_in_streamlit:
        import sys as _sys

        from streamlit.web import cli as _streamlit_cli

        _sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        raise SystemExit(_streamlit_cli.main())

    main()
