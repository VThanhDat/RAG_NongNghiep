"""
Filter retrieved documents by retrieval score.
"""
from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.core.base import BasePostProcessor


class ScoreFilter(BasePostProcessor):
    def __init__(self, threshold: float = 0.0, keep_min: int = 1):
        self.threshold = threshold
        self.keep_min = keep_min

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs or self.threshold <= 0:
            return docs

        passed = [doc for doc in docs if self._score(doc) >= self.threshold]
        if not passed and self.keep_min > 0:
            return docs[: self.keep_min]
        return passed

    @staticmethod
    def _score(doc: Document) -> float:
        for key in ("score", "rrf_score", "relevance_score", "bm25_score"):
            value = doc.metadata.get(key)
            if value is not None:
                return float(value)
        return 0.0
