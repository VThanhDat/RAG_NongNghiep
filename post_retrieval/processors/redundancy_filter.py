"""
Lightweight near-duplicate filter using token Jaccard similarity.
"""
from __future__ import annotations

from langchain_core.documents import Document

from post_retrieval.core.base import BasePostProcessor


class RedundancyFilter(BasePostProcessor):
    def __init__(self, top_n: int = 5, threshold: float = 0.9):
        self.top_n = top_n
        self.threshold = threshold

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        kept: list[Document] = []
        for doc in docs:
            if any(self._jaccard(doc.page_content, existing.page_content) >= self.threshold for existing in kept):
                continue
            kept.append(doc)
            if len(kept) >= self.top_n:
                break
        return kept

    @staticmethod
    def _jaccard(left: str, right: str) -> float:
        left_tokens = set((left or "").casefold().split())
        right_tokens = set((right or "").casefold().split())
        union = left_tokens | right_tokens
        if not union:
            return 0.0
        return len(left_tokens & right_tokens) / len(union)
