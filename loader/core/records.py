from __future__ import annotations

from langchain_core.documents import Document


def to_text_records(docs: list[Document]) -> list[dict]:
    """Convert LangChain Documents to {"text": ..., "metadata": ...} records."""
    return [{"text": doc.page_content, "metadata": dict(doc.metadata)} for doc in docs]
