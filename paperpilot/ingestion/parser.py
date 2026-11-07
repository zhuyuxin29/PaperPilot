"""PDF 解析（模块 4）：按页提取文本，生成 Page 契约。

职责：
- 打开 PDF、处理加密文档（可给密码）；
- 逐页提取文本，保留原始文本（不做清洗）；
- 标注空白页与提取异常，避免把扫描页当成“解析成功的空文本”。

边界：
- 不做 OCR、不做分块、不做清洗；偏移映射由分块模块负责；
- 不写数据库，返回解析结果由调用方决定如何入库。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from ..schemas import Page


class PdfParseError(Exception):
    """PDF 解析失败的基类；``code`` 供上层映射成统一错误码。"""

    code = "PDF_PARSE_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class PdfEncryptedError(PdfParseError):
    """文档已加密，且未提供可用密码。"""

    code = "PDF_ENCRYPTED"


class PdfCorruptedError(PdfParseError):
    """文件损坏或不是有效的 PDF。"""

    code = "PDF_CORRUPTED"


class PdfEmptyError(PdfParseError):
    """PDF 没有任何页面。"""

    code = "PDF_EMPTY"


@dataclass(frozen=True)
class ParseResult:
    """一次解析的完整结果。"""

    paper_id: str
    revision: int
    pages: tuple[Page, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def blank_page_numbers(self) -> tuple[int, ...]:
        return tuple(page.page_number for page in self.pages if page.is_blank)


def parse_pdf(
    path: Path | str,
    paper_id: str,
    revision: int = 1,
    password: str | None = None,
) -> ParseResult:
    """按页解析 PDF；失败时抛 ``PdfParseError`` 子类。"""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(str(pdf_path))

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # pypdf 对损坏文件抛出多种异常
        raise PdfCorruptedError(f"无法读取 PDF：{exc}") from exc

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt(password or "")
        except Exception as exc:
            raise PdfEncryptedError(f"加密文档解密失败：{exc}") from exc
        if not unlocked:
            raise PdfEncryptedError("PDF 已加密，需要提供正确密码")

    try:
        total = len(reader.pages)
    except Exception as exc:
        raise PdfCorruptedError(f"无法读取页面列表：{exc}") from exc
    if total == 0:
        raise PdfEmptyError("PDF 没有任何页面")

    pages: list[Page] = []
    for page_index in range(total):
        quality_flags: list[str] = []
        try:
            raw_text = reader.pages[page_index].extract_text() or ""
        except Exception:
            raw_text = ""
            quality_flags.append("extract_error")

        is_blank = not raw_text.strip()
        if is_blank:
            quality_flags.append("blank")

        pages.append(
            Page(
                paper_id=paper_id,
                revision=revision,
                page_index=page_index,
                page_number=page_index + 1,
                text=raw_text,
                is_blank=is_blank,
                quality_flags=tuple(dict.fromkeys(quality_flags)),
            )
        )

    return ParseResult(paper_id=paper_id, revision=revision, pages=tuple(pages))
