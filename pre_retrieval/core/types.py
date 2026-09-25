"""
pre_retrieval/core/types.py
===========================
Shared data containers for the pre-retrieval pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class PreRetrievalContext:
    """Optional context for the pre-retrieval pipeline (chat history, etc.)."""
    chat_history: list = field(default_factory=list)
    user_id: str | None = None
    session_id: str | None = None
    extra: dict = field(default_factory=dict)

@dataclass
class TransformResult:
    """Output of any pre-retrieval transformer."""
    original_query:    str
    queries:           list[str]      = field(default_factory=list)
    clean_query:       str | None     = None
    rewritten_query:   str | None     = None
    expanded_terms:    list[str]      = field(default_factory=list)
    language:          str | None     = None
    intent:            str | None     = None
    metadata:          dict           = field(default_factory=dict)
    metadata_filter:   dict | None    = None
    retrieval_path:    str | None     = None
    extra:             dict           = field(default_factory=dict)
    is_valid:          bool           = True
    rejection_reason:  str | None     = None
    warnings:          list[str]      = field(default_factory=list)

    def all_queries(self) -> list[str]:
        if not self.is_valid:
            return []
        seen:   set[str]  = set()
        result: list[str] = []
        preferred_query = self.rewritten_query or self.clean_query or self.original_query
        for q in [preferred_query] + self.queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                result.append(q.strip())
        return result

    def to_retriever_payload(self) -> dict:
        metadata = dict(self.metadata)
        metadata.update({
            "queries": self.all_queries() if self.is_valid else [],
            "metadata_filter": self.metadata_filter,
            "retrieval_path": self.retrieval_path,
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
            "warnings": list(self.warnings),
            "extra": dict(self.extra),
        })
        return {
            "original_query": self.original_query,
            "clean_query": self.clean_query or self.original_query.strip(),
            "rewritten_query": self.rewritten_query or self.clean_query or self.original_query.strip(),
            "expanded_terms": list(self.expanded_terms),
            "language": self.language or "unknown",
            "intent": self.intent or "unknown",
            "metadata": metadata,
        }

    def as_dict(self) -> dict:
        return self.to_retriever_payload()
