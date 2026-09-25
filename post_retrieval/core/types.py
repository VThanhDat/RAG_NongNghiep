"""
post_retrieval/types.py
========================
Shared output type for the Post-Retrieval stage.

PostRetrievalResult is the contract this stage exposes to the Generator.
It wraps the final list of documents together with pipeline diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class PostRetrievalResult:
    """
    Output of the PostRetrievalPipeline.

    Attributes
    ----------
    query         : Primary query used for this pipeline run.
    documents     : Final ranked list of Documents ready for prompt injection.
    steps_run     : Names of processors that were executed, in order.
    input_count   : Number of documents received from Retrieval.
    output_count  : Number of documents returned after all filtering.
    latency_ms    : Total wall-clock time (ms) for the pipeline.
    warnings      : Non-fatal issues (e.g. all docs below score threshold).
    """

    query:        str
    documents:    list[Document]  = field(default_factory=list)
    steps_run:    list[str]       = field(default_factory=list)
    input_count:  int             = 0
    output_count: int             = 0
    latency_ms:   float           = 0.0
    warnings:     list[str]       = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.documents) == 0

    @property
    def top_doc(self) -> Document | None:
        return self.documents[0] if self.documents else None

    def to_context_string(self, separator: str = "\n\n---\n\n") -> str:
        """
        Flatten all documents into a single context string for generation.

        Each chunk is prefixed with its rank and source for citation readiness.
        """
        parts = []
        for doc in self.documents:
            rank   = doc.metadata.get("rank", "?")
            source = doc.metadata.get("source", "unknown")
            title  = doc.metadata.get("title", "")
            prefix = f"[{rank}] {title} ({source})" if title else f"[{rank}] ({source})"
            parts.append(f"{prefix}\n{doc.page_content.strip()}")
        return separator.join(parts)

    def as_dict(self) -> dict:
        return {
            "query":        self.query,
            "steps_run":    self.steps_run,
            "input_count":  self.input_count,
            "output_count": self.output_count,
            "latency_ms":   round(self.latency_ms, 2),
            "is_empty":     self.is_empty,
            "warnings":     list(self.warnings),
            "documents": [
                {
                    "rank":     doc.metadata.get("rank"),
                    "score":    doc.metadata.get("score") or doc.metadata.get("rerank_score"),
                    "source":   doc.metadata.get("source"),
                    "title":    doc.metadata.get("title"),
                    "page":     doc.metadata.get("page_number", doc.metadata.get("page")),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "text":     doc.page_content[:300],
                    "metadata": doc.metadata,
                }
                for doc in self.documents
            ],
        }
