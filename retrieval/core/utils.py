"""
Shared helpers for the simplified retrieval stage.
"""
from __future__ import annotations

from collections import defaultdict
import re

from langchain_core.documents import Document


def is_low_value_document(doc: Document) -> bool:
    """Return whether a chunk is navigation-only rather than evidence.

    Table-of-contents chunks often contain the exact query terms and can win
    BM25 despite having no explanatory content. They should not be sent to
    the answer model as evidence.
    """
    text = " ".join((doc.page_content or "").split())
    if not text:
        return True

    lower = text.casefold()
    heading_count = len(re.findall(r"(?:^|\s)\d+(?:\.\d+)*[.)]", text))
    if "mục lục" in lower and heading_count >= 3:
        return True
    if len(text) < 180 and heading_count >= 2:
        return True
    return False


def deduplicate(docs: list[Document]) -> list[Document]:
    """Remove duplicate documents based on normalized page content."""
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in docs:
        key = " ".join(doc.page_content.split())
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def reciprocal_rank_fusion(ranked_lists: list[list[Document]], k: int = 60) -> list[Document]:
    """Merge ranked lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = " ".join(doc.page_content.split())
            scores[key] += 1.0 / (k + rank)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    fused: list[Document] = []
    for rank, key in enumerate(sorted_keys, start=1):
        doc = doc_map[key]
        score = scores[key]
        fused.append(
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rrf_score": score, "rank": rank, "score": score},
            )
        )
    return fused


def add_rank_scores(docs: list[Document]) -> list[Document]:
    """Ensure every retrieved document has rank and score metadata."""
    result: list[Document] = []
    for rank, doc in enumerate(docs, start=1):
        score = (
            doc.metadata.get("rrf_score")
            or doc.metadata.get("relevance_score")
            or doc.metadata.get("bm25_score")
            or (1.0 / rank)
        )
        result.append(
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rank": rank, "score": float(score)},
            )
        )
    return result
