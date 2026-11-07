"""Merge human-readable chat archives with the durable execution journal."""
import json
from . import storage
from .agent import runs


def load_history(settings, paper_id):
    path = settings.data_dir / "chats" / f"{paper_id}.json"
    history = []
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            history = value if isinstance(value, list) else []
        except (ValueError, OSError):
            pass
    conn = storage.connect(settings.db_path)
    try:
        runs.recover_orphans(conn)
        by_id = {turn.get("run_id"): index for index, turn in enumerate(history) if turn.get("run_id")}
        for row in reversed(runs.recent(conn, paper_id)):
            result = json.loads(row["result_json"] or "{}")
            turn = {
                "id": row["run_id"], "run_id": row["run_id"],
                "question": row["question"], "status": row["status"],
                "answer": result.get("answer", "") if row["status"] == "COMPLETED" else "",
                "stop_reason": result.get("stop_reason") or row["status"],
                "citations": result.get("citations", []),
                "trace": result.get("trace", json.loads(row["trace_json"])),
                "invalid_citations": result.get("invalid_citations", []),
                "repair_attempts": result.get("repair_attempts", 0),
                "revision": result.get("source_revisions", {}).get(paper_id),
                "error": result.get("code"),
            }
            turn["graph_claims"] = [claim for item in turn["trace"] for claim in (item.get("graph_claims") or [])]
            if row["run_id"] in by_id:
                history[by_id[row["run_id"]]] = turn
            else:
                history.append(turn)
        return history[-50:]
    finally:
        conn.close()
