from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from paperpilot import storage, services
from paperpilot.agent.runner import run_agent
from paperpilot.config import Settings, load_settings
from paperpilot.graph.builder import build_paper_graph
from paperpilot.graph.client import GraphError, Neo4jClient
from paperpilot.ingestion import pipeline
from paperpilot.llm import (
    ChatError,
    EmbeddingError,
    create_chat_provider,
    create_embedding_provider,
)
from paperpilot.maintenance import purge_paper
from paperpilot.rag.retrieval import RetrievalError
from paperpilot.schemas import Summary
from paperpilot.summary.generator import (
    SummaryError,
    current_summary_model,
    generate_summary,
    load_summary,
    summary_path,
)
from paperpilot.tools import build_registry

from .schemas import (
    ChatIn,
    ChatOut,
    CitationOut,
    HealthOut,
    PaperOut,
    SummaryOut,
)


logger = logging.getLogger(__name__)
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_GRAPH_STATE_UNSET = object()


def _read_upload_limited(
    upload: UploadFile, max_bytes: int | None = None
) -> bytes:
    limit = pipeline.MAX_PDF_BYTES if max_bytes is None else max_bytes
    declared = getattr(upload, "size", None)
    if isinstance(declared, int) and declared > limit:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"文件超过大小上限（{limit} 字节）",
            },
        )
    blocks: list[bytes] = []
    total = 0
    while True:
        block = upload.file.read(_UPLOAD_CHUNK_BYTES)
        if not block:
            break
        total += len(block)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "FILE_TOO_LARGE",
                    "message": f"文件超过大小上限（{limit} 字节）",
                },
            )
        blocks.append(block)
    return b"".join(blocks)


@lru_cache
def get_settings() -> Settings:
    settings = load_settings()
    settings.ensure_dirs()
    return settings


def get_conn(
    settings: Settings = Depends(get_settings),
) -> Iterator[sqlite3.Connection]:
    conn = storage.connect(settings.db_path)
    storage.ensure_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


app = FastAPI(
    title="PaperPilot API",
    version="0.1.0",
    description="带证据的文献阅读 Agent：文献、要点、问答与图谱接口。",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _paper_out(
    conn: sqlite3.Connection,
    paper,
    counts: tuple[int, int] | None = None,
    graph_state: dict | None | object = _GRAPH_STATE_UNSET,
) -> PaperOut:
    page_count, chunk_count = (
        counts if counts is not None else storage.get_paper_counts(
            conn, paper.id, paper.revision
        )
    )
    if graph_state is _GRAPH_STATE_UNSET:
        graph_state = storage.get_graph_state(conn, paper.id)
    return PaperOut(
        id=paper.id,
        title=paper.title,
        file_name=paper.file_name,
        state=paper.state.value,
        revision=paper.revision,
        page_count=page_count,
        chunk_count=chunk_count,
        graph_status=(
            str(graph_state["status"]) if graph_state else "not_built"
        ),
    )


def _get_paper(conn: sqlite3.Connection, paper_id: str):
    paper = storage.get_paper(conn, paper_id)
    if paper is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "PAPER_NOT_FOUND", "message": f"文献不存在：{paper_id}"},
        )
    return paper


def _graph_client(settings: Settings):
    if settings.graph_mode == "off":
        return None
    try:
        return Neo4jClient.connect(settings)
    except GraphError as exc:
        if settings.graph_mode == "on":
            raise HTTPException(status_code=503, detail={"code": exc.code, "message": exc.message}) from exc
        return None


def _summary_out(summary: Summary) -> SummaryOut:
    payload = summary.model_dump(mode="json")
    return SummaryOut(**payload)


@app.get("/health", response_model=HealthOut, tags=["system"])
def health(settings: Settings = Depends(get_settings)) -> HealthOut:
    return HealthOut(status="ok", graph_mode=settings.graph_mode)


@app.get("/papers", response_model=list[PaperOut], tags=["papers"])
def list_papers(conn: sqlite3.Connection = Depends(get_conn)) -> list[PaperOut]:
    counts = storage.list_paper_counts(conn)
    graph_states = storage.list_graph_states(conn)
    return [
        _paper_out(
            conn, paper, counts.get(paper.id), graph_states.get(paper.id)
        )
        for paper in storage.list_papers(conn)
    ]


@app.post(
    "/papers",
    response_model=PaperOut,
    status_code=201,
    tags=["papers"],
)
def upload_paper(
    file: UploadFile = File(..., description="PDF 文件"),
    response: Response = None,
    background: bool = False,
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_conn),
) -> PaperOut:
    data = _read_upload_limited(file)
    if not data:
        raise HTTPException(
            status_code=400,
            detail={"code": "EMPTY_FILE", "message": "上传文件为空"},
        )
    if background:
        paper, started = services.start_background_ingest(
            conn, settings, data, file.filename or "upload.pdf"
        )
        response.status_code = 202 if started else 200
        return _paper_out(conn, paper)
    try:
        result = services.prepare_paper(
            data, file.filename or "upload.pdf", settings, conn
        )
    except pipeline.IngestError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail={"code": exc.code,
            "message": "原文已保留，向量索引未完成；检查模型配置后重新上传同一文件可重试。"}) from exc
    except Exception as exc:
        logger.exception("上传入库失败")
        raise HTTPException(
            status_code=500,
            detail={"code": "INGEST_FAILED", "message": "入库失败，请稍后重试。"},
        ) from exc
    if result.duplicated:
        response.status_code = 200
    return _paper_out(conn, result.paper)


@app.get("/papers/{paper_id}", response_model=PaperOut, tags=["papers"])
def get_paper(
    paper_id: str,
    conn: sqlite3.Connection = Depends(get_conn),
) -> PaperOut:
    return _paper_out(conn, _get_paper(conn, paper_id))


@app.delete("/papers/{paper_id}", status_code=204, tags=["papers"])
def delete_paper(
    paper_id: str,
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_conn),
) -> None:
    _get_paper(conn, paper_id)
    purge_paper(settings, conn, paper_id)


@app.get("/papers/{paper_id}/pdf", tags=["papers"])
def download_pdf(
    paper_id: str,
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_conn),
) -> FileResponse:
    paper = _get_paper(conn, paper_id)
    path = settings.paper_dir(paper_id) / "source.pdf"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"code": "PDF_NOT_FOUND", "message": "找不到原始 PDF"},
        )
    return FileResponse(
        path, media_type="application/pdf", filename=paper.file_name
    )


@app.post(
    "/papers/{paper_id}/reprocess",
    response_model=PaperOut,
    tags=["papers"],
)
def reprocess_paper(
    paper_id: str,
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_conn),
) -> PaperOut:
    paper = _get_paper(conn, paper_id)
    path = settings.paper_dir(paper_id) / "source.pdf"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"code": "PDF_NOT_FOUND", "message": "找不到原始 PDF"},
        )
    try:
        result = services.reprocess_paper(settings, conn, paper_id)
    except pipeline.IngestError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except Exception as exc:
        logger.exception("重新处理论文失败：%s", paper_id)
        raise HTTPException(
            status_code=500,
            detail={"code": "REPROCESS_FAILED", "message": "重新处理失败，请稍后重试。"},
        ) from exc
    return _paper_out(conn, result.paper)


@app.get("/papers/{paper_id}/chunks/{chunk_id}", tags=["papers"])
def get_chunk(
    paper_id: str,
    chunk_id: str,
    conn: sqlite3.Connection = Depends(get_conn),
) -> dict:
    paper = _get_paper(conn, paper_id)
    chunk = storage.get_chunk(conn, chunk_id)
    if (
        chunk is None
        or chunk.paper_id != paper.id
        or chunk.revision != paper.revision
    ):
        raise HTTPException(
            status_code=404,
            detail={"code": "CHUNK_NOT_FOUND", "message": "片段不存在"},
        )
    return {
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "page_number": chunk.page_number,
        "text": chunk.text,
    }


@app.post("/papers/{paper_id}/graph", tags=["graph"])
def build_graph(
    paper_id: str,
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_conn),
) -> dict:
    paper = _get_paper(conn, paper_id)
    if settings.graph_mode == "off":
        raise HTTPException(
            status_code=409,
            detail={"code": "GRAPH_DISABLED", "message": "图谱模式为 off"},
        )
    graph = _graph_client(settings)
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "GRAPH_UNAVAILABLE", "message": "无法连接 Neo4j"},
        )
    try:
        result = build_paper_graph(
            conn,
            settings,
            create_chat_provider(settings),
            graph,
            paper.id,
        )
    except ChatError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    finally:
        graph.close()
    return {
        "status": result.status,
        "chunks_processed": result.chunks_processed,
        "claims_total": result.claims_total,
        "claims_validated": result.claims_validated,
        "claims_pending": result.claims_pending,
        "errors": list(result.errors),
    }


@app.get(
    "/papers/{paper_id}/summary", response_model=SummaryOut, tags=["summary"]
)
def get_summary(
    paper_id: str,
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_conn),
) -> SummaryOut:
    paper = _get_paper(conn, paper_id)
    model_id = current_summary_model(settings)
    path = summary_path(settings, paper_id, paper.revision, model_id)
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "SUMMARY_NOT_GENERATED",
                "message": "还没有生成七类要点",
            },
        )
    try:
        summary = load_summary(
            settings,
            path,
            paper_id=paper_id,
            revision=paper.revision,
            model_id=model_id,
        )
    except Exception as exc:
        logger.exception("要点缓存读取失败：%s", paper_id)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SUMMARY_BROKEN",
                "message": "要点缓存已过期或损坏，请重新生成。",
            },
        ) from exc
    return _summary_out(summary)


@app.post(
    "/papers/{paper_id}/summary",
    response_model=SummaryOut,
    tags=["summary"],
)
def create_summary(
    paper_id: str,
    force: bool = True,
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_conn),
) -> SummaryOut:
    paper = _get_paper(conn, paper_id)
    graph = _graph_client(settings)
    try:
        result = generate_summary(
            conn,
            settings,
            create_chat_provider(settings),
            paper.id,
            force=force,
            graph=graph,
        )
    except ChatError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except SummaryError as exc:
        logger.warning("要点生成失败：%s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": exc.code,
                "message": "要点生成失败，请稍后重试或检查模型配置。",
            },
        ) from exc
    finally:
        if graph is not None:
            graph.close()
    return _summary_out(result.summary)


@app.post(
    "/papers/{paper_id}/chat", response_model=ChatOut, tags=["chat"]
)
def chat(
    paper_id: str,
    payload: ChatIn,
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_conn),
) -> ChatOut:
    paper = _get_paper(conn, paper_id)
    graph = _graph_client(settings)
    try:
        result = run_agent(
            question=payload.question,
            paper_ids=(paper.id,),
            conn=conn,
            settings=settings,
            embedding=create_embedding_provider(settings),
            chat=create_chat_provider(settings),
            tools=build_registry(graph_enabled=graph is not None and services.graph_ready(conn, (paper.id,))),
            graph=graph,
        )
    except ChatError as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except GraphError as exc:
        raise HTTPException(status_code=503, detail={"code": exc.code, "message": exc.message}) from exc
    except RetrievalError as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    finally:
        if graph is not None:
            graph.close()

    return ChatOut(
        run_id=result.run_id, answer_kind=result.answer_kind,
        status=result.status,
        stop_reason=result.stop_reason,
        answer=result.answer,
        citations=[
            CitationOut(
                evidence_id=record.evidence_id,
                chunk_id=record.chunk_id,
                page_number=record.page_number,
                quote=record.quote,
            )
            for record in result.citations
        ],
        invalid_citations=list(result.invalid_citations),
        steps=result.steps,
        tool_calls=result.tool_calls,
        repair_attempts=result.repair_attempts,
        trace=[dict(entry) for entry in result.trace],
    )
