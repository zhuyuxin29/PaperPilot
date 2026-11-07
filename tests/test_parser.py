"""模块 4（parser）验收测试。

测试用 PDF 由 ``tests/pdf_factory.py`` 现场生成，不依赖外部样例文件。
"""

from __future__ import annotations

import io
import logging
import tempfile
import unittest
from pathlib import Path

from paperpilot.ingestion import parser
from pdf_factory import build_pdf

# pypdf 对损坏文件会打 warning 日志，测试里只保留错误级别，避免干扰输出
logging.getLogger("pypdf").setLevel(logging.ERROR)


class ParserTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_pdf(self, data: bytes, name: str = "demo.pdf") -> Path:
        path = self.tmp_path / name
        path.write_bytes(data)
        return path


class ParsePdfTest(ParserTestBase):
    def test_three_pages_with_blank_page(self) -> None:
        pdf_path = self.write_pdf(
            build_pdf(["Hello PaperPilot", None, "Third page text"])
        )
        result = parser.parse_pdf(pdf_path, paper_id="paper_001")

        self.assertEqual(result.page_count, 3)
        self.assertEqual([p.page_number for p in result.pages], [1, 2, 3])

        first, second, third = result.pages
        self.assertIn("Hello PaperPilot", first.text)
        self.assertFalse(first.is_blank)
        self.assertEqual(first.quality_flags, ())

        self.assertTrue(second.is_blank)
        self.assertEqual(second.text, "")
        self.assertIn("blank", second.quality_flags)

        self.assertIn("Third page text", third.text)
        self.assertFalse(third.is_blank)

        self.assertEqual(result.blank_page_numbers, (2,))
        self.assertEqual(result.paper_id, "paper_001")
        self.assertEqual(result.revision, 1)

    def test_revision_is_propagated(self) -> None:
        pdf_path = self.write_pdf(build_pdf(["Page one"]))
        result = parser.parse_pdf(pdf_path, paper_id="paper_001", revision=3)
        self.assertTrue(all(page.revision == 3 for page in result.pages))

    def test_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            parser.parse_pdf(self.tmp_path / "nope.pdf", paper_id="paper_001")

    def test_corrupted_file(self) -> None:
        path = self.write_pdf(b"this is definitely not a pdf", "broken.pdf")
        with self.assertRaises(parser.PdfCorruptedError) as ctx:
            parser.parse_pdf(path, paper_id="paper_001")
        self.assertEqual(ctx.exception.code, "PDF_CORRUPTED")

    def test_encrypted_pdf_requires_password(self) -> None:
        path = self.write_pdf(
            build_pdf(["Secret page"], encrypt_password="secret"), "locked.pdf"
        )

        with self.assertRaises(parser.PdfEncryptedError) as ctx:
            parser.parse_pdf(path, paper_id="paper_001")
        self.assertEqual(ctx.exception.code, "PDF_ENCRYPTED")

        unlocked = parser.parse_pdf(path, paper_id="paper_001", password="secret")
        self.assertEqual(unlocked.page_count, 1)
        self.assertIn("Secret page", unlocked.pages[0].text)

    def test_empty_pdf(self) -> None:
        from pypdf import PdfWriter

        writer = PdfWriter()
        buffer = io.BytesIO()
        try:
            writer.write(buffer)
        except Exception:  # 生成 0 页 PDF 失败则跳过该用例
            self.skipTest("当前 pypdf 无法生成 0 页 PDF")

        path = self.write_pdf(buffer.getvalue(), "empty.pdf")
        with self.assertRaises((parser.PdfEmptyError, parser.PdfCorruptedError)):
            parser.parse_pdf(path, paper_id="paper_001")


class ErrorCodeTest(unittest.TestCase):
    def test_error_codes(self) -> None:
        self.assertEqual(parser.PdfParseError("x").code, "PDF_PARSE_ERROR")
        self.assertEqual(parser.PdfEncryptedError("x").code, "PDF_ENCRYPTED")
        self.assertEqual(parser.PdfCorruptedError("x").code, "PDF_CORRUPTED")
        self.assertEqual(parser.PdfEmptyError("x").code, "PDF_EMPTY")


if __name__ == "__main__":
    unittest.main()
