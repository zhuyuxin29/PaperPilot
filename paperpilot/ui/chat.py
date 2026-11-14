"""文献问答页、源文献面板与后台任务轮询。"""

from __future__ import annotations

import copy
import sqlite3
import threading
import time
import uuid

import streamlit as st

from .common import (
    STOP_LABELS,
    TOOL_LABELS,
    _api_client,
    _http_mode,
    _paper_info,
    _render_pdf_button,
    _load_chat_history,
    _save_chat_history,
    _renumber_citations,
)
from paperpilot import storage, services
from paperpilot.agent.runner import AgentResult, run_agent
from paperpilot.config import Settings
from paperpilot.conversation import rewrite_followup
from paperpilot.evidence.registry import EvidenceRecord
from paperpilot.graph.client import GraphError, Neo4jClient
from paperpilot.llm import create_chat_provider, create_embedding_provider
from paperpilot.tools import build_registry
from .source import _render_source_panel

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
            _pending_poller(settings, paper_id)

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


def _poll_pending(settings: Settings, conn: sqlite3.Connection | None = None) -> None:
    """轮询并收尾后台任务；连接参数保留兼容，收尾时自行创建线程内连接。"""
    pending = st.session_state.get("pending")
    if not pending or pending["thread"].is_alive():
        return
    _finalize_pending(settings)


def _render_pending_status(settings: Settings, pending: dict, paper_id: str) -> None:
    label = (
        "正在请求后端…" if _http_mode(settings) else "正在检索与取证…"
    )
    with st.status(label, expanded=True) as status:
        if _http_mode(settings):
            st.write("API 模式：等待后端完成本次问答。")
        else:
            st.write("Agent 正在检索、回读原文并校验引用。")
            if st.button("停止", key=f"stop_{paper_id}"):
                pending["cancel"].set()
                status.update(label="正在停止…", state="running")


@st.fragment(run_every=0.8)
def _pending_poller(settings: Settings, paper_id: str) -> None:
    """局部轮询后台问答任务，避免整页 rerun。"""
    pending = st.session_state.get("pending")
    if not pending or pending.get("paper_id") != paper_id:
        return
    if pending["thread"].is_alive():
        _render_pending_status(settings, pending, paper_id)
        return
    _poll_pending(settings)
    st.rerun()


def _finalize_pending(settings: Settings) -> None:
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
    # fragment 可能运行在与页面不同的线程：SQLite 连接必须在当前线程创建。
    conn = storage.connect(settings.db_path)
    try:
        paper = storage.get_paper(conn, paper_id)
        revision = paper.revision if paper else None
    finally:
        conn.close()
    history.append(
        {
            "id": uuid.uuid4().hex[:8],
            "question": question,
            "answer": result.answer,
            "run_id": result.run_id,
            "revision": revision,
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

