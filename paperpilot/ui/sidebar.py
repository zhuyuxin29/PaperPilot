"""侧边栏：品牌、导入文献、当前论文与设置。"""

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
from .common import *  # noqa: F401,F403
from .common import (
    _api_client, _http_mode, _paper_info, _paper_status_line, _pdf_bytes,
    _render_pdf_button, _purge_paper, _chats_path, _load_chat_history,
    _save_chat_history, _renumber_citations, _summary_to_markdown,
)


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


def _render_import_popover(settings, conn, chat_page) -> None:
    with st.popover("＋ 导入论文", width="stretch"):
        uploaded = st.file_uploader(
            "选择 PDF 文件", type=["pdf"], key="sidebar_upload"
        )
        if uploaded is not None:
            st.caption(f"{uploaded.name} · {len(uploaded.getvalue())} 字节")
            if st.button("开始入库", type="primary", key="sidebar_ingest"):
                if _http_mode(settings):
                    try:
                        client = _api_client(settings)
                        paper = client.upload_paper(
                            uploaded.getvalue(), uploaded.name, background=True
                        )
                        if paper.get("state") in ("parsing", "uploaded"):
                            with st.spinner("后端正在解析、向量化并建索引…"):
                                paper = client.wait_for_paper(paper["id"])
                    except ApiError as exc:
                        st.error(f"入库失败：{exc.message}")
                    else:
                        if paper.get("state") == "failed":
                            st.error("入库失败，请检查 PDF 或后端日志。")
                        else:
                            st.session_state["current_paper"] = paper["id"]
                            st.success(f"入库完成：{paper['id']}")
                            st.switch_page(chat_page)
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
                        st.switch_page(chat_page)


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


def render_sidebar(settings, conn, pages) -> None:
    """渲染完整侧边栏；pages 为 st.Page 列表（问答/要点/文献库）。"""
    with st.sidebar:
        st.markdown(
            '<div class="pp-brand">PaperPilot</div>'
            '<div class="pp-sub">你的文献阅读助手</div>',
            unsafe_allow_html=True,
        )
        for page, label, icon in zip(
            pages, ("文献问答", "文献要点", "我的文献"), ("💬", "📄", "📚")
        ):
            st.page_link(page, label=label, icon=icon)
        st.divider()
        _render_import_popover(settings, conn, pages[0])
        _render_current_paper(settings, conn)
        st.divider()
        _render_settings(settings)
