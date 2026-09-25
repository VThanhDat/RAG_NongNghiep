"""
loader/pdf/strategies/pdfplumber.py
===================================
pdfplumber — best table extraction (text layer).
"""

from __future__ import annotations

from langchain_core.documents import Document

from loader.core.base import BaseLoader, Language
from loader.utils.text import clean_text


class PDFPlumberLoader(BaseLoader):
    """
    PDF extraction via pdfplumber — best table extraction for text-layer PDFs.

    Chuyển đổi bảng phát hiện được thành Markdown inline với văn bản xung quanh.
    Không có OCR. Chậm hơn PyMuPDF.
    Dùng cho: báo cáo tài chính, data sheet, PDF nhiều bảng (text layer).

    Tham số:
        extract_tables : Bật/tắt trích xuất bảng → Markdown.
    """

    def __init__(self, language: Language = "both", extract_tables: bool = True):
        super().__init__(language)
        self.extract_tables = extract_tables

    def load(self, file_path: str) -> list[Document]:
        import pdfplumber

        docs = []
        with pdfplumber.open(file_path) as pdf:
            total = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                parts: list[str] = []

                raw = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                if raw:
                    parts.append(raw)

                if self.extract_tables:
                    for table in page.extract_tables():
                        if not table:
                            continue
                        header, *rows = table
                        header = [str(h or "") for h in header]
                        md = [
                            "| " + " | ".join(header) + " |",
                            "| " + " | ".join(["---"] * len(header)) + " |",
                        ]
                        for row in rows:
                            cells = ([str(c or "") for c in row] + [""] * len(header))[:len(header)]
                            md.append("| " + " | ".join(cells) + " |")
                        parts.append("\n".join(md))

                content = clean_text("\n\n".join(parts))
                if content:
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": file_path, "page": page_num,
                            "total_pages": total, "file_type": "pdf",
                            "pdf_strategy": "pdfplumber",
                            "has_table": self.extract_tables and bool(page.extract_tables()),
                        },
                    ))
        return self._stamp(docs)
