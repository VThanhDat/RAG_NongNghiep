"""
Record conversion helpers for chunking output.
"""

from __future__ import annotations

from langchain_core.documents import Document


def to_text_records(chunks: list[Document]) -> list[dict]:
    """Convert chunk Documents to {"text": ..., "metadata": ...} records."""
    return [{"text": chunk.page_content, "metadata": dict(chunk.metadata)} for chunk in chunks]
