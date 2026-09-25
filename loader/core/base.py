"""
loader/core/base.py
==============
Abstract base class cho tất cả document loader.

Giao kèo (contract) của mọi loader:
    loader = SomeLoader(language="both", **options)
    docs   = loader.load(file_path)    -> list[Document]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document

Language = Literal["vi","en","both"]

class BaseLoader(ABC):
    """
    Base class cho mọi document loader.

    Tham số
    -------
    language : Ngôn ngữ tài liệu nguồn.
               \"vi\" = Tiếng Việt, \"en\" = Tiếng Anh, \"both\" = hỗn hợp.
               Được lưu vào metadata[\"language\"] để các stage sau biết ngôn ngữ nguồn.
    """

    def __init__(self, language: Language = "both"):
        self.language = language
        
    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        """
        Load một file và trả về danh sách Document.

        Tham số
        -------
        file_path : Đường dẫn tuyệt đối hoặc tương đối đến file nguồn.

        Trả về
        ------
        List[Document] — một file có thể cho ra nhiều document
        (ví dụ: 1 document/trang, 1 document/sheet bảng tính).
        """
        
    def _stamp(self, docs: list[Document]) -> list[Document]:
        """
        Đóng dấu (stamp) metadata chung cho tất cả document.

        Metadata này giúp các stage sau truy vết nguồn, trang và ngôn ngữ tài liệu.
        """
        created_at = datetime.now(timezone.utc).isoformat()
        for doc in docs:
            meta = doc.metadata
            source = meta.get("source") or meta.get("file_path") or ""
            path = Path(str(source)) if source else None
            suffix = path.suffix.lower().lstrip(".") if path else ""

            file_type = meta.get("file_type") or meta.get("document_type") or suffix or "unknown"
            title = meta.get("title") or (path.stem if path else None)

            page = meta.get("page")
            if meta.get("page_number") is None and page is not None:
                if isinstance(page, int) and meta.get("pdf_strategy") == "pypdf":
                    meta["page_number"] = page + 1
                else:
                    meta["page_number"] = page

            meta.setdefault("source", str(source))
            meta.setdefault("file_name", path.name if path else "")
            meta.setdefault("page_number", None)
            meta.setdefault("file_type", file_type)
            meta.setdefault("document_type", file_type)
            meta.setdefault("title", title)
            meta.setdefault("section", None)
            meta.setdefault("created_at", created_at)
            meta.setdefault("language", self.language)
        return docs
