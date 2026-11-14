from __future__ import annotations

import html
import sqlite3

import streamlit as st

from .common import SECTION_COLORS, SECTION_STATUS_LABELS, _api_client, _http_mode
from paperpilot.client import ApiError
from paperpilot.config import Settings
from paperpilot.graph.client import GraphError, Neo4jClient
from paperpilot.llm import create_chat_provider
from paperpilot.schemas import Summary
from paperpilot.summary.generator import SUMMARY_LABELS, current_summary_model, generate_summary, load_summary, summary_path

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
