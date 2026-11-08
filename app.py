"""PaperPilot 应用入口：组装导航、侧边栏与三个页面。

业务逻辑在 ``paperpilot`` 包；页面实现拆到 ``paperpilot/ui``。
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from paperpilot import storage
from paperpilot.config import Settings, load_settings
from paperpilot.ui.chat import _render_chat_page, _start_pending
from paperpilot.ui.common import (
    _CSS,
    _paper_info,
    format_state,
    paper_rows,
)
from paperpilot.ui.library import _render_library_page
from paperpilot.ui.points import _render_points_page
from paperpilot.ui.sidebar import _render_welcome, render_sidebar

__all__ = [
    "format_state",
    "paper_rows",
    "_page_chat",
    "_page_points",
    "_page_library",
    "_start_pending",
]


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
        _render_library_page(settings, conn, CHAT_PAGE)
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
        render_sidebar(settings, conn, PAGES)
        st.navigation(PAGES, position="hidden").run()
    finally:
        conn.close()


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

        try:
            from streamlit.web import cli as _st_cli

            _sys.argv = ["streamlit", "run", __file__]
            _st_cli.main()
        except SystemExit:
            raise
    else:
        main()
