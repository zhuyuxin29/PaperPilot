import json
import os
import uuid
from datetime import datetime, timezone

_PROCESS = f"{os.getpid()}:{uuid.uuid4().hex}"


def ensure(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS agent_runs (
        run_id TEXT PRIMARY KEY, owner TEXT NOT NULL, question TEXT NOT NULL,
        paper_ids TEXT NOT NULL, status TEXT NOT NULL,
        started_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        trace_json TEXT NOT NULL DEFAULT '[]', result_json TEXT
    )''')
    conn.commit()


def _now():
    return datetime.now(timezone.utc).isoformat()


def start(conn, question, paper_ids):
    ensure(conn)
    prune(conn)
    identifier = uuid.uuid4().hex
    with conn:
        conn.execute("INSERT INTO agent_runs(run_id,owner,question,paper_ids,status,started_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                     (identifier, _PROCESS, question, json.dumps(list(paper_ids)), "RUNNING", _now(), _now()))
    return identifier


def prune(conn, keep: int = 500):
    ensure(conn)
    if keep < 1:
        raise ValueError("keep 必须大于等于 1")
    with conn:
        conn.execute(
            "DELETE FROM agent_runs WHERE run_id IN ("
            "SELECT run_id FROM agent_runs ORDER BY started_at DESC LIMIT -1 OFFSET ?)",
            (keep,),
        )


def delete_for_paper(conn, paper_id: str) -> int:
    ensure(conn)
    rows = conn.execute("SELECT run_id, paper_ids FROM agent_runs").fetchall()
    targets = [
        row["run_id"] for row in rows
        if paper_id in json.loads(row["paper_ids"])
    ]
    if targets:
        with conn:
            conn.executemany(
                "DELETE FROM agent_runs WHERE run_id = ?",
                [(run_id,) for run_id in targets],
            )
    return len(targets)


def progress(conn, identifier, trace):
    with conn:
        conn.execute("UPDATE agent_runs SET trace_json=?,updated_at=? WHERE run_id=?",
                     (json.dumps(trace, ensure_ascii=False), _now(), identifier))


def finish(conn, identifier, status, result):
    with conn:
        conn.execute("UPDATE agent_runs SET status=?,result_json=?,updated_at=? WHERE run_id=?",
                     (status, json.dumps(result, ensure_ascii=False), _now(), identifier))


def mark_interrupted(conn, run_id):
    ensure(conn)
    with conn:
        conn.execute("UPDATE agent_runs SET status='INTERRUPTED',updated_at=? WHERE run_id=? AND status='RUNNING'",
                     (_now(), run_id))


def _owner_alive(owner):
    try:
        pid = int(owner.split(":", 1)[0])
    except (ValueError, AttributeError):
        return True
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if handle:
            kernel.CloseHandle(handle)
            return True
        return ctypes.get_last_error() != 87
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def recover_orphans(conn):
    ensure(conn)
    for identifier, owner in conn.execute("SELECT run_id,owner FROM agent_runs WHERE status='RUNNING'").fetchall():
        if not _owner_alive(owner):
            mark_interrupted(conn, identifier)


def recent(conn, paper_id=None, limit=50):
    ensure(conn)
    rows = conn.execute("SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
    result = []
    names = [column[1] for column in conn.execute("PRAGMA table_info(agent_runs)")]
    for row in rows:
        item = dict(zip(names, row))
        if paper_id is not None and paper_id not in json.loads(item["paper_ids"]):
            continue
        result.append(item)
    return result
