"""论文数据清理（删除论文时复用）。

清理范围：SQLite 记录、论文目录、索引文件、要点缓存、Neo4j 节点。
"""

from __future__ import annotations

import shutil
import sqlite3

from . import storage
from .config import Settings
from .services import document_lock


def purge_paper(settings, conn, paper_id, *, clear_graph=True):
    with document_lock(settings):
        return _purge_paper(settings, conn, paper_id, clear_graph=clear_graph)



def _purge_paper(
    settings: Settings,
    conn: sqlite3.Connection,
    paper_id: str,
    *,
    clear_graph: bool = True,
) -> None:
    if clear_graph and settings.graph_mode != "off":
        client = None
        try:
            from .graph.client import WORKSPACE_ID, Neo4jClient

            client = Neo4jClient.connect(settings)
            client.clear_paper_graph(WORKSPACE_ID, paper_id)
        except Exception:
            pass
        finally:
            if client is not None:
                client.close()

    storage.delete_paper(conn, paper_id)
    shutil.rmtree(settings.paper_dir(paper_id), ignore_errors=True)
    for path in settings.index_dir.glob(f"{paper_id}*"):
        path.unlink(missing_ok=True)
    for path in settings.summaries_dir.glob(f"{paper_id}.*"):
        path.unlink(missing_ok=True)
    (settings.data_dir / "chats" / f"{paper_id}.json").unlink(missing_ok=True)
