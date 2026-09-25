"""
Shared helpers for post-retrieval processing.
"""
from __future__ import annotations

from langchain_core.documents import Document


def normalize_text(text: str) -> str:
    return " ".join((text or "").split()).casefold()


def deduplicate(docs: list[Document]) -> list[Document]:
    """Remove exact duplicate documents by normalized text."""
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in docs:
        key = normalize_text(doc.page_content)
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique
