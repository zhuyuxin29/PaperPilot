"""我的文献页：上传、选择、重新处理与删除。"""

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


def _render_library_page(settings, conn, chat_page=None) -> None:
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
                st.switch_page(chat_page)
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
