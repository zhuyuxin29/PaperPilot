from __future__ import annotations

import argparse

from paperpilot import storage
from paperpilot.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="按关键词查找片段，辅助标注")
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--grep", default=None, help="关键词（不区分大小写）")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    settings = load_settings()
    conn = storage.connect(settings.db_path)
    storage.ensure_schema(conn)
    try:
        chunks = storage.get_chunks(conn, args.paper_id)
        if not chunks:
            print(f"这篇论文没有片段：{args.paper_id}")
            return
        keyword = (args.grep or "").lower()
        shown = 0
        for chunk in chunks:
            if keyword and keyword not in chunk.text.lower():
                continue
            snippet = chunk.text[:120].replace("\n", " ")
            print(f"{chunk.chunk_id}\t第{chunk.page_number}页\t{snippet}")
            shown += 1
            if shown >= args.limit:
                break
        if shown == 0:
            print("没有匹配的片段")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
