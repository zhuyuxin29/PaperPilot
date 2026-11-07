"""Inspect persisted runs and mark only proven-dead owners interrupted."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paperpilot import storage
from paperpilot.config import load_settings
from paperpilot.agent import runs


def main():
    parser = argparse.ArgumentParser(description="查看 PaperPilot 任务日志")
    parser.add_argument("--paper-id")
    parser.add_argument("--recover", action="store_true", help="将已退出进程遗留的运行标记为 INTERRUPTED；不会重发模型请求")
    args = parser.parse_args()
    settings = load_settings()
    conn = storage.connect(settings.db_path)
    try:
        if args.recover:
            runs.recover_orphans(conn)
        for item in runs.recent(conn, args.paper_id):
            print(json.dumps({key:item[key] for key in ("run_id","status","question","started_at","updated_at")}, ensure_ascii=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
