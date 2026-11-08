"""文献操作：重新处理等后台动作。"""

from __future__ import annotations

import sqlite3

import streamlit as st

from .common import _pdf_bytes
from paperpilot import storage, services
from paperpilot.config import Settings

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
