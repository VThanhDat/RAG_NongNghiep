"""
loader/pdf/strategies/pypdf.py
==============================
PyPDF — text layer, zero extra deps.
"""

from __future__ import annotations

from langchain_core.documents import Document

from loader.core.base import BaseLoader
from loader.utils.text import clean_text


class PyPDFLoader(BaseLoader):
    """
    Text-layer PDF extraction via LangChain built-in PyPDFLoader.

    Không cần cài thêm gì ngoài pypdf (đã bao gồm trong langchain-community).
    Trả về 1 Document / trang. Không có OCR, không trích bảng.
    Dùng cho: PDF digital text thuần, khi ưu tiên tốc độ.
    """

    def load(self, file_path: str) -> list[Document]:
        from langchain_community.document_loaders import PyPDFLoader as _PyPDF

        docs = _PyPDF(file_path).load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata.update({"file_type": "pdf", "pdf_strategy": "pypdf"})
        return self._stamp(docs)
