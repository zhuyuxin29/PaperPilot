from __future__ import annotations

import html
import json
import sqlite3

from types import SimpleNamespace
import streamlit as st

from .common import _api_client, _http_mode, _paper_info, _summary_to_markdown
from paperpilot import storage
from paperpilot.client import ApiError
from paperpilot.config import Settings
from paperpilot.schemas import SUMMARY_KEYS
from paperpilot.summary.generator import current_summary_model
from .source import _render_source_panel
from .summary import (
    _generate_summary,
    _load_summary,
    _render_summary_section,
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
