"""
retrieval/core/types.py
=======================
Shared data containers for the Retrieval stage.

RetrievalResult is the output contract of this stage — it wraps the list of
retrieved documents together with metadata about how the retrieval was performed.
The Reranker or Generator stage consumes this object directly.

Design mirrors TransformResult from pre_retrieval/core/types.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    """
    Output of the Retrieval stage.

    Attributes
    ----------
    query        : The primary query used for retrieval (rewritten or original).
    documents    : Ranked list of retrieved Document objects, each with
                   ``rank`` and ``score`` in their metadata.
    strategy     : The retrieval strategy that was used (e.g. "hybrid").
    total_found  : How many candidate docs were considered before top-k cut-off.
    latency_ms   : Wall-clock time (ms) spent in the retrieve() call.
    metadata_filter : The metadata filter that was applied, if any.
    extra        : Free-form dict for downstream stages to attach extra data.
    is_empty     : True when no documents were found (empty result).
    warnings     : Non-fatal issues encountered during retrieval.
    """

    query:           str
    documents:       list[Document]      = field(default_factory=list)
    strategy:        str                 = "unknown"
    total_found:     int                 = 0
    latency_ms:      float               = 0.0
    metadata_filter: dict | None         = None
    extra:           dict                = field(default_factory=dict)
    warnings:        list[str]           = field(default_factory=list)

    # -------------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return len(self.documents) == 0

    @property
    def top_doc(self) -> Document | None:
        """Return the highest-ranked document, or None if empty."""
        return self.documents[0] if self.documents else None

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def scores(self) -> list[float]:
        """Return the list of scores for all retrieved documents."""
        return [doc.metadata.get("score", 0.0) for doc in self.documents]

    def to_context_string(self, separator: str = "\n\n---\n\n") -> str:
        """
        Flatten the document list into a single string for prompt injection.

        Each chunk is prefixed with its rank and source for traceability.
        """
        parts = []
        for doc in self.documents:
            rank   = doc.metadata.get("rank", "?")
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{rank}] ({source})\n{doc.page_content.strip()}")
        return separator.join(parts)

    def as_dict(self) -> dict:
        return {
            "query":           self.query,
            "strategy":        self.strategy,
            "total_found":     self.total_found,
            "latency_ms":      round(self.latency_ms, 2),
            "is_empty":        self.is_empty,
            "metadata_filter": self.metadata_filter,
            "warnings":        list(self.warnings),
            "documents": [
                {
                    "rank":    doc.metadata.get("rank"),
                    "score":   doc.metadata.get("score"),
                    "source":  doc.metadata.get("source"),
                    "text":    doc.page_content[:300],
                    "metadata": doc.metadata,
                }
                for doc in self.documents
            ],
        }
