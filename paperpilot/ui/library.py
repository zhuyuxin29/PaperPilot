from __future__ import annotations

import streamlit as st

from .common import _api_client, _http_mode, _render_pdf_button, _purge_paper, paper_rows
from paperpilot.client import ApiError
from .paper_actions import _reprocess_paper

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
