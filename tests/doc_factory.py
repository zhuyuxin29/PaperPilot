from __future__ import annotations

from paperpilot.schemas import Chunk, Page, Paper


def make_document(
    paper_id: str = "paper_001",
    texts: tuple[str, ...] = ("alpha beta", "gamma delta"),
    revision: int = 1,
    sha256: str = "a" * 64,
) -> tuple[Paper, list[Page], list[Chunk]]:
    paper = Paper(
        id=paper_id,
        title=f"title {paper_id}",
        file_name=f"{paper_id}.pdf",
        sha256=sha256,
        revision=revision,
        page_count=len(texts),
    )
    pages: list[Page] = []
    chunks: list[Chunk] = []
    for index, text in enumerate(texts):
        pages.append(
            Page(
                paper_id=paper_id,
                revision=revision,
                page_index=index,
                page_number=index + 1,
                text=text,
            )
        )
        chunks.append(
            Chunk(
                chunk_id=f"{paper_id}:r{revision}:p{index + 1}:c{index:04d}:v1:{index:08x}",
                paper_id=paper_id,
                revision=revision,
                page_index=index,
                page_number=index + 1,
                ordinal=index,
                text=text,
                start_offset=0,
                end_offset=len(text),
            )
        )
    return paper, pages, chunks
