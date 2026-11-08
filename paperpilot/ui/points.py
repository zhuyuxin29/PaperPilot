"""文献要点页：七类要点卡片、导出与生成。"""

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

