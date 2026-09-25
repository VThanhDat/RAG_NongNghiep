"""
Order final context documents before generation.
"""
from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.core.base import BasePostProcessor


class ContextOrderer(BasePostProcessor):
    def __init__(self, ordering: str = "relevance"):
        if ordering not in {"relevance", "original"}:
            raise ValueError("Supported context ordering: relevance, original.")
        self.ordering = ordering

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        if self.ordering == "original":
            return docs
        return sorted(docs, key=self._score, reverse=True)

    @staticmethod
    def _score(doc: Document) -> float:
        for key in ("score", "rrf_score", "relevance_score", "bm25_score"):
            value = doc.metadata.get(key)
            if value is not None:
                return float(value)
        rank = doc.metadata.get("rank")
        if rank:
            return 1.0 / float(rank)
        return 0.0
