"""Durable task journal. Never store credentials or hidden model reasoning."""
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
    identifier = uuid.uuid4().hex
    with conn:
        conn.execute("INSERT INTO agent_runs(run_id,owner,question,paper_ids,status,started_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                     (identifier, _PROCESS, question, json.dumps(list(paper_ids)), "RUNNING", _now(), _now()))
    return identifier


def progress(conn, identifier, trace):
    with conn:
        conn.execute("UPDATE agent_runs SET trace_json=?,updated_at=? WHERE run_id=?",
                     (json.dumps(trace, ensure_ascii=False), _now(), identifier))


def finish(conn, identifier, status, result):
    with conn:
        conn.execute("UPDATE agent_runs SET status=?,result_json=?,updated_at=? WHERE run_id=?",
                     (status, json.dumps(result, ensure_ascii=False), _now(), identifier))


def mark_interrupted(conn, run_id):
    """Explicit recovery only: a different process may still own an active run.

    Callers must confirm the old process has exited; do not blindly mark all
    RUNNING tasks interrupted on every Streamlit rerun or API startup.
    """
    ensure(conn)
    with conn:
        conn.execute("UPDATE agent_runs SET status='INTERRUPTED',updated_at=? WHERE run_id=? AND status='RUNNING'",
                     (_now(), run_id))


def _owner_alive(owner):
    try:
        pid = int(owner.split(":", 1)[0])
    except (ValueError, AttributeError):
        return True  # Unknown owner is not proof that it has stopped.
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
        return ctypes.get_last_error() != 87  # Access denied does not mean dead.
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
