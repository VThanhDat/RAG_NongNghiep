"""Exact post-split deduplication for chunking output."""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document


def deduplicate_chunks(
    chunks: list[Document],
    method: str = "exact",
) -> list[Document]:
    """
    Remove exactly duplicated chunk text.

    Only ``method="exact"`` is supported in the simplified agriculture RAG
    pipeline. The first occurrence is kept and order is preserved.
    """
    if method != "exact":
        raise ValueError("Only exact chunk deduplication is supported.")

    seen: set[str] = set()
    unique: list[Document] = []
    for chunk in chunks:
        h = hashlib.md5(chunk.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    return unique
