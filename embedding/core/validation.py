"""
Vector validation helpers.
"""

from __future__ import annotations

from typing import Any

REQUIRED_INDEX_METADATA = (
    "source",
    "file_name",
    "page_number",
    "title",
    "section",
    "chunk_id",
    "chunk_index",
)


def l2_normalize_vector(vector: list[float]) -> list[float]:
    """L2-normalize a dense vector for cosine/dot-product indexes."""
    import math

    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm == 0:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]


def validate_vector_dimensions(
    vectors: list[list[float]],
    expected_dimension: int | None,
) -> int:
    """
    Validate vector dimensionality and return the detected dimension.

    Raises ValueError for empty vectors, inconsistent dimensions, or mismatch
    with the expected Vector DB dimension.
    """
    if not vectors:
        return expected_dimension or 0

    detected = len(vectors[0])
    if detected == 0:
        raise ValueError("Embedding vector is empty.")
    bad = [index for index, vector in enumerate(vectors) if len(vector) != detected]
    if bad:
        raise ValueError(
            f"Inconsistent embedding dimensions: expected {detected}, "
            f"bad indexes={bad[:10]}"
        )
    if expected_dimension is not None and detected != expected_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dimension}, got {detected}."
        )
    return detected


def validate_embedding_record(
    record: dict[str, Any],
    *,
    expected_dimension: int | None,
    required_metadata: tuple[str, ...] = REQUIRED_INDEX_METADATA,
    strict_metadata: bool = False,
) -> list[str]:
    """Validate one index-ready embedding record and return warning labels."""
    warnings: list[str] = []
    vector = record.get("embedding")
    if not isinstance(vector, list):
        raise ValueError("Embedding record is missing a dense vector list.")
    validate_vector_dimensions([vector], expected_dimension)

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Embedding record is missing metadata.")

    missing = [
        key
        for key in required_metadata
        if key not in metadata or metadata.get(key) in (None, "")
    ]
    if missing:
        message = f"Embedding metadata missing required keys: {missing}"
        if strict_metadata:
            raise ValueError(message)
        metadata["missing_metadata"] = missing
        warnings.append("missing_metadata")

    return warnings
