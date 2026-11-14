from __future__ import annotations

import base64
import io
import sqlite3

import pypdfium2 as pdfium
import streamlit as st

from .common import _api_client, _http_mode, _paper_info, _render_pdf_button
from paperpilot.client import ApiError
from paperpilot.config import Settings

@st.cache_data(show_spinner=False)
def _render_pdf_page(
    pdf_path: str, page_index: int, scale: float = 1.6
) -> bytes | None:
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
    page_key = f"source_page_{paper_id}"
    current_page = int(st.session_state.get(page_key, 1))
    current_page = max(1, min(page_count, current_page))
    if page_count > 1:
        current_page = int(
            st.slider(
                "页码",
                min_value=1,
                max_value=page_count,
                value=current_page,
                key=page_key,
            )
        )
    st.caption(f"第 {current_page} / {page_count} 页")


    image = _render_pdf_page(pdf_path, current_page - 1)
    if image is None:
        st.info("这一页暂时无法渲染。")
    else:
        encoded = base64.b64encode(image).decode("ascii")
        st.markdown(
            '<div class="pp-pdf-view">'
            f'<img src="data:image/png;base64,{encoded}" '
            f'style="width:{zoom}%" alt="第 {current_page} 页">'
            f'<div class="pp-pdf-page-no">第 {current_page} 页</div>'
            "</div>",
            unsafe_allow_html=True,
        )
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
