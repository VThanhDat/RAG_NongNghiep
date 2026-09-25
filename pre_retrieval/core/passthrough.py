"""
pre_retrieval/core/passthrough.py
=================================
Passthrough — no transformation applied.
"""
from __future__ import annotations
from pre_retrieval.core.base import BaseTransformer
from pre_retrieval.core.types import TransformResult

class PassthroughTransformer(BaseTransformer):
    """No-op transformer — passes the query through as-is."""
    def transform(self, query: str, **kwargs) -> TransformResult:
        return TransformResult(original_query=query, queries=[query])
