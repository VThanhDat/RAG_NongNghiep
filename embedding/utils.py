"""Backward-compatible utility exports for embedding modules."""

from __future__ import annotations

from embedding.core.records import stable_text_id
from embedding.core.text import (
    embedding_text_quality_flags,
    is_embedding_text_valid,
    normalize_embedding_text,
)
from embedding.core.validation import (
    REQUIRED_INDEX_METADATA,
    l2_normalize_vector,
    validate_embedding_record,
    validate_vector_dimensions,
)

__all__ = [
    "REQUIRED_INDEX_METADATA",
    "embedding_text_quality_flags",
    "is_embedding_text_valid",
    "l2_normalize_vector",
    "normalize_embedding_text",
    "stable_text_id",
    "validate_embedding_record",
    "validate_vector_dimensions",
]
