from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .schemas import Chunk, Page, Paper, PaperState

SCHEMA_VERSION = 2

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS papers (
        id          TEXT PRIMARY KEY,
        title       TEXT NOT NULL,
        file_name   TEXT NOT NULL,
        sha256      TEXT NOT NULL,
        revision    INTEGER NOT NULL DEFAULT 1,
        page_count  INTEGER NOT NULL DEFAULT 0,
        state       TEXT NOT NULL DEFAULT 'uploaded',
        created_at  TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_papers_sha256 ON papers(sha256)",
    """
    CREATE TABLE IF NOT EXISTS pages (
        paper_id      TEXT NOT NULL,
        revision      INTEGER NOT NULL,
        page_index    INTEGER NOT NULL,
        page_number   INTEGER NOT NULL,
        text          TEXT NOT NULL DEFAULT '',
        is_blank      INTEGER NOT NULL DEFAULT 0,
        quality_flags TEXT NOT NULL DEFAULT '[]',
        PRIMARY KEY (paper_id, revision, page_index),
        FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id       TEXT PRIMARY KEY,
        paper_id       TEXT NOT NULL,
        revision       INTEGER NOT NULL,
        page_index     INTEGER NOT NULL,
        page_number    INTEGER NOT NULL,
        ordinal        INTEGER NOT NULL,
        text           TEXT NOT NULL,
        start_offset   INTEGER NOT NULL,
        end_offset     INTEGER NOT NULL,
        token_estimate INTEGER,
        UNIQUE (paper_id, revision, ordinal),
        FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id, revision, page_index)",
    """
    CREATE TABLE IF NOT EXISTS paper_graph_state (
        paper_id     TEXT PRIMARY KEY,
        revision     INTEGER NOT NULL,
        status       TEXT NOT NULL,
        entity_count INTEGER NOT NULL DEFAULT 0,
        claim_count  INTEGER NOT NULL DEFAULT 0,
        updated_at   TEXT NOT NULL,
        error        TEXT,
        FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
    )
    """,
)


class SchemaVersionError(RuntimeError):
    pass


class StorageError(RuntimeError):
    def __init__(self, path: Path | str, reason: str | None = None) -> None:
        self.path = Path(path)
        message = (
            f"无法打开数据库文件：{self.path}。"
            "请检查该目录是否存在且可写，或修改 .env 中的 PAPERPILOT_DATA_DIR。"
        )
        if reason:
            message += f"（原始错误：{reason}）"
        super().__init__(message)


def connect(db_path: Path | str) -> sqlite3.Connection:
    if isinstance(db_path, str) and db_path == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        path = Path(db_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(path)
            conn.execute("PRAGMA journal_mode = WAL")
        except OSError as exc:
            raise StorageError(path, str(exc)) from exc
        except sqlite3.OperationalError as exc:
            raise StorageError(path, str(exc)) from exc
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> int:
    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)

    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
    else:
        version = int(row["value"])
        if version > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"数据库结构版本 {version} 高于代码支持的 {SCHEMA_VERSION}，"
                "请升级程序或恢复备份，不要直接写入"
            )
        if version < SCHEMA_VERSION:
            raise SchemaVersionError(
                f"数据库结构版本 {version} 低于代码支持的 {SCHEMA_VERSION}；"
                "请先备份并迁移数据，或重建数据目录（PAPERPILOT_DATA_DIR）。"
            )
    conn.commit()
    return current_schema_version(conn)


def current_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()
    return int(row["value"]) if row is not None else 0


def get_sequence(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = ?", (f"seq:{name}",)
    ).fetchone()
    return int(row["value"]) if row is not None else 0


def set_sequence(conn: sqlite3.Connection, name: str, value: int) -> None:
    with conn:
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (f"seq:{name}", str(int(value))),
        )


def save_paper(conn: sqlite3.Connection, paper: Paper) -> None:
    with conn:
        _upsert_paper(conn, paper)


def save_pages(conn: sqlite3.Connection, pages: Iterable[Page]) -> None:
    with conn:
        for page in pages:
            _upsert_page(conn, page)


def save_chunks(conn: sqlite3.Connection, chunks: Iterable[Chunk]) -> None:
    with conn:
        for chunk in chunks:
            _upsert_chunk(conn, chunk)


def save_document(
    conn: sqlite3.Connection,
    paper: Paper,
    pages: Sequence[Page] = (),
    chunks: Sequence[Chunk] = (),
) -> None:
    with conn:
        _upsert_paper(conn, paper)
        for page in pages:
            _upsert_page(conn, page)
        for chunk in chunks:
            _upsert_chunk(conn, chunk)


def update_paper_state(
    conn: sqlite3.Connection, paper_id: str, state: PaperState
) -> None:
    with conn:
        cursor = conn.execute(
            "UPDATE papers SET state = ? WHERE id = ?", (state.value, paper_id)
        )
        if cursor.rowcount == 0:
            raise KeyError(f"文献不存在：{paper_id}")


def delete_paper(conn: sqlite3.Connection, paper_id: str) -> bool:
    with conn:
        cursor = conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        return cursor.rowcount > 0


def save_graph_state(
    conn: sqlite3.Connection,
    paper_id: str,
    revision: int,
    status: str,
    *,
    entity_count: int = 0,
    claim_count: int = 0,
    error: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO paper_graph_state
                (paper_id, revision, status, entity_count, claim_count, updated_at, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                revision = excluded.revision,
                status = excluded.status,
                entity_count = excluded.entity_count,
                claim_count = excluded.claim_count,
                updated_at = excluded.updated_at,
                error = excluded.error
            """,
            (
                paper_id,
                revision,
                status,
                entity_count,
                claim_count,
                now,
                error,
            ),
        )


def get_graph_state(
    conn: sqlite3.Connection, paper_id: str
) -> dict[str, object] | None:
    row = conn.execute(
        "SELECT * FROM paper_graph_state WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    return dict(row) if row is not None else None


def list_graph_states(
    conn: sqlite3.Connection,
) -> dict[str, dict[str, object]]:
    rows = conn.execute("SELECT * FROM paper_graph_state").fetchall()
    return {str(row["paper_id"]): dict(row) for row in rows}


def get_paper(conn: sqlite3.Connection, paper_id: str) -> Paper | None:
    row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    return _paper_from_row(row) if row is not None else None


def find_paper_by_sha256(
    conn: sqlite3.Connection, sha256: str
) -> Paper | None:
    row = conn.execute(
        "SELECT * FROM papers WHERE sha256 = ? ORDER BY created_at LIMIT 1",
        (sha256.strip().lower(),),
    ).fetchone()
    return _paper_from_row(row) if row is not None else None


def list_papers(conn: sqlite3.Connection) -> list[Paper]:
    rows = conn.execute(
        "SELECT * FROM papers ORDER BY created_at DESC, id"
    ).fetchall()
    return [_paper_from_row(row) for row in rows]


def get_pages(
    conn: sqlite3.Connection, paper_id: str, revision: int = 1
) -> list[Page]:
    rows = conn.execute(
        "SELECT * FROM pages WHERE paper_id = ? AND revision = ? "
        "ORDER BY page_index",
        (paper_id, revision),
    ).fetchall()
    return [_page_from_row(row) for row in rows]


def get_chunks(
    conn: sqlite3.Connection,
    paper_id: str,
    revision: int = 1,
    *,
    page_index: int | None = None,
) -> list[Chunk]:
    if page_index is None:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE paper_id = ? AND revision = ? "
            "ORDER BY ordinal",
            (paper_id, revision),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE paper_id = ? AND revision = ? "
            "AND page_index = ? ORDER BY ordinal",
            (paper_id, revision, page_index),
        ).fetchall()
    return [_chunk_from_row(row) for row in rows]


def get_chunk_ids(
    conn: sqlite3.Connection, paper_id: str, revision: int = 1
) -> list[str]:
    rows = conn.execute(
        "SELECT chunk_id FROM chunks WHERE paper_id = ? AND revision = ? "
        "ORDER BY ordinal",
        (paper_id, revision),
    ).fetchall()
    return [str(row["chunk_id"]) for row in rows]


def get_chunks_by_ids(
    conn: sqlite3.Connection, chunk_ids: Sequence[str]
) -> dict[str, Chunk]:
    unique = [chunk_id for chunk_id in dict.fromkeys(chunk_ids) if chunk_id]
    if not unique:
        return {}
    result: dict[str, Chunk] = {}

    for start in range(0, len(unique), 400):
        batch = unique[start : start + 400]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", batch
        ).fetchall()
        for row in rows:
            chunk = _chunk_from_row(row)
            result[chunk.chunk_id] = chunk
    return result


def get_paper_counts(
    conn: sqlite3.Connection, paper_id: str, revision: int = 1
) -> tuple[int, int]:
    row = conn.execute(
        "SELECT (SELECT COUNT(*) FROM pages WHERE paper_id = ? AND revision = ?) "
        "AS page_count, "
        "(SELECT COUNT(*) FROM chunks WHERE paper_id = ? AND revision = ?) "
        "AS chunk_count",
        (paper_id, revision, paper_id, revision),
    ).fetchone()
    return (int(row["page_count"]), int(row["chunk_count"])) if row else (0, 0)


def list_paper_counts(
    conn: sqlite3.Connection,
) -> dict[str, tuple[int, int]]:
    rows = conn.execute(
        """
        SELECT p.id AS id,
               (SELECT COUNT(*) FROM pages pg
                WHERE pg.paper_id = p.id AND pg.revision = p.revision) AS page_count,
               (SELECT COUNT(*) FROM chunks ch
                WHERE ch.paper_id = p.id AND ch.revision = p.revision) AS chunk_count
        FROM papers p
        """
    ).fetchall()
    return {
        str(row["id"]): (int(row["page_count"]), int(row["chunk_count"]))
        for row in rows
    }


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> Chunk | None:
    row = conn.execute(
        "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    return _chunk_from_row(row) if row is not None else None


def _upsert_paper(conn: sqlite3.Connection, paper: Paper) -> None:
    conn.execute(
        """
        INSERT INTO papers (id, title, file_name, sha256, revision, page_count, state, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            file_name = excluded.file_name,
            sha256 = excluded.sha256,
            revision = excluded.revision,
            page_count = excluded.page_count,
            state = excluded.state,
            created_at = excluded.created_at
        """,
        (
            paper.id,
            paper.title,
            paper.file_name,
            paper.sha256,
            paper.revision,
            paper.page_count,
            paper.state.value,
            paper.created_at.isoformat(),
        ),
    )


def _upsert_page(conn: sqlite3.Connection, page: Page) -> None:
    conn.execute(
        """
        INSERT INTO pages (paper_id, revision, page_index, page_number, text, is_blank, quality_flags)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id, revision, page_index) DO UPDATE SET
            page_number = excluded.page_number,
            text = excluded.text,
            is_blank = excluded.is_blank,
            quality_flags = excluded.quality_flags
        """,
        (
            page.paper_id,
            page.revision,
            page.page_index,
            page.page_number,
            page.text,
            int(page.is_blank),
            json.dumps(list(page.quality_flags), ensure_ascii=False),
        ),
    )


def _upsert_chunk(conn: sqlite3.Connection, chunk: Chunk) -> None:
    conn.execute(
        """
        INSERT INTO chunks (chunk_id, paper_id, revision, page_index, page_number,
                            ordinal, text, start_offset, end_offset, token_estimate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            paper_id = excluded.paper_id,
            revision = excluded.revision,
            page_index = excluded.page_index,
            page_number = excluded.page_number,
            ordinal = excluded.ordinal,
            text = excluded.text,
            start_offset = excluded.start_offset,
            end_offset = excluded.end_offset,
            token_estimate = excluded.token_estimate
        """,
        (
            chunk.chunk_id,
            chunk.paper_id,
            chunk.revision,
            chunk.page_index,
            chunk.page_number,
            chunk.ordinal,
            chunk.text,
            chunk.start_offset,
            chunk.end_offset,
            chunk.token_estimate,
        ),
    )


def _paper_from_row(row: sqlite3.Row) -> Paper:
    return Paper(
        id=row["id"],
        title=row["title"],
        file_name=row["file_name"],
        sha256=row["sha256"],
        revision=row["revision"],
        page_count=row["page_count"],
        state=PaperState(row["state"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _page_from_row(row: sqlite3.Row) -> Page:
    return Page(
        paper_id=row["paper_id"],
        revision=row["revision"],
        page_index=row["page_index"],
        page_number=row["page_number"],
        text=row["text"],
        is_blank=bool(row["is_blank"]),
        quality_flags=tuple(json.loads(row["quality_flags"] or "[]")),
    )


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        paper_id=row["paper_id"],
        revision=row["revision"],
        page_index=row["page_index"],
        page_number=row["page_number"],
        ordinal=row["ordinal"],
        text=row["text"],
        start_offset=row["start_offset"],
        end_offset=row["end_offset"],
        token_estimate=row["token_estimate"],
    )
