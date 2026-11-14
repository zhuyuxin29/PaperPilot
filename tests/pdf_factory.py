from __future__ import annotations

import io


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(
    page_texts: list[str | None],
    *,
    encrypt_password: str | None = None,
) -> bytes:
    page_count = len(page_texts)
    catalog_num, pages_num = 1, 2
    page_nums = list(range(3, 3 + page_count))
    next_num = 3 + page_count
    content_nums: dict[int, int] = {}
    for index, text in enumerate(page_texts):
        if text:
            content_nums[index] = next_num
            next_num += 1
    font_num = next_num

    def page_object(index: int) -> str:
        parts = [
            "<< /Type /Page",
            f"/Parent {pages_num} 0 R",
            "/MediaBox [0 0 612 792]",
        ]
        if index in content_nums:
            parts.append(f"/Contents {content_nums[index]} 0 R")
            parts.append(f"/Resources << /Font << /F1 {font_num} 0 R >> >>")
        else:
            parts.append("/Resources << >>")
        parts.append(">>")
        return " ".join(parts)

    objects: dict[int, bytes] = {}
    kids = " ".join(f"{num} 0 R" for num in page_nums)
    objects[catalog_num] = f"<< /Type /Catalog /Pages {pages_num} 0 R >>".encode()
    objects[pages_num] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode()
    )
    for index, num in enumerate(page_nums):
        objects[num] = page_object(index).encode()
    for index, num in content_nums.items():
        stream = (
            f"BT /F1 12 Tf 72 720 Td "
            f"({_escape_pdf_text(page_texts[index] or '')}) Tj ET"
        ).encode()
        objects[num] = (
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
    objects[font_num] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objects[num] + b"\nendobj\n"

    xref_position = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in range(1, size):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {size} /Root {catalog_num} 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode()
    data = bytes(out)

    if encrypt_password:
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(io.BytesIO(data))
        writer = PdfWriter()
        writer.append(reader)
        writer.encrypt(
            user_password=encrypt_password,
            owner_password=f"{encrypt_password}-owner",
        )
        buffer = io.BytesIO()
        writer.write(buffer)
        return buffer.getvalue()
    return data
