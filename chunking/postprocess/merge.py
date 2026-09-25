"""
Post-split merging for tiny chunks.
"""

from __future__ import annotations

from langchain_core.documents import Document

from chunking.core.metadata import enrich_chunk_metadata, estimate_token_count


def merge_tiny_chunks(
    chunks: list[Document],
    *,
    min_tokens: int = 100,
    max_tokens: int = 1000,
    token_encoding: str = "cl100k_base",
) -> list[Document]:
    """
    Merge very small chunks into the previous compatible chunk.

    Compatibility is intentionally conservative: chunks are merged only when
    they share source, page, section, and hierarchy level. This avoids mixing
    unrelated headings/pages while reducing orphan heading/bullet fragments.
    """
    if not chunks:
        return chunks

    merged: list[Document] = []
    for chunk in chunks:
        current = Document(
            page_content=chunk.page_content,
            metadata=dict(chunk.metadata),
        )
        current_tokens = _token_count(current, token_encoding)
        previous = merged[-1] if merged else None
        previous_tokens = _token_count(previous, token_encoding) if previous else 0

        if (
            current_tokens < min_tokens
            and merged
            and _can_merge(merged[-1], current)
            and previous_tokens + current_tokens <= max_tokens
        ):
            previous.metadata.setdefault("merged_from_chunk_ids", [previous.metadata.get("chunk_id")])
            previous.metadata["merged_from_chunk_ids"].append(current.metadata.get("chunk_id"))
            previous.page_content = _join(previous.page_content, current.page_content)
            previous.metadata["merged_tiny_chunks"] = True
            continue

        if (
            previous is not None
            and previous_tokens < min_tokens
            and _can_merge(previous, current)
            and previous_tokens + current_tokens <= max_tokens
        ):
            previous.metadata.setdefault("merged_from_chunk_ids", [previous.metadata.get("chunk_id")])
            previous.metadata["merged_from_chunk_ids"].append(current.metadata.get("chunk_id"))
            previous.page_content = _join(previous.page_content, current.page_content)
            previous.metadata["merged_tiny_chunks"] = True
            continue

        current.metadata.setdefault("merged_from_chunk_ids", [current.metadata.get("chunk_id")])
        merged.append(current)

    return enrich_chunk_metadata(merged, token_encoding=token_encoding)


def _merge_key(chunk: Document) -> tuple:
    metadata = chunk.metadata
    return (
        metadata.get("source"),
        metadata.get("page_number") or metadata.get("page"),
        metadata.get("chunk_level"),
        metadata.get("parent_id"),
    )


def _can_merge(left: Document, right: Document) -> bool:
    return _merge_key(left) == _merge_key(right) and _sections_compatible(
        left.metadata.get("section"),
        right.metadata.get("section"),
    )


def _sections_compatible(left: object, right: object) -> bool:
    if left == right:
        return True
    if not left or not right:
        return True
    left_s = str(left)
    right_s = str(right)
    return left_s.startswith(f"{right_s} >") or right_s.startswith(f"{left_s} >")


def _token_count(chunk: Document, token_encoding: str) -> int:
    value = chunk.metadata.get("token_count")
    if isinstance(value, int) and value > 0:
        return value
    token_count, _ = estimate_token_count(chunk.page_content, token_encoding)
    return token_count


def _join(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left}\n\n{right}"
