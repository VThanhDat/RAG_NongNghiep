"""
Chunk metadata helpers.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

from langchain_core.documents import Document


@lru_cache(maxsize=8)
def _get_tiktoken_encoding(encoding_name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None


def estimate_token_count(text: str, encoding_name: str = "cl100k_base") -> tuple[int, str]:
    """
    Count tokens when the tokenizer is available, otherwise use a safe estimate.

    The fallback avoids network/cache failures during indexing. It is intentionally
    conservative for Vietnamese and mixed technical text.
    """
    encoding = _get_tiktoken_encoding(encoding_name)
    if encoding is not None:
        try:
            return len(encoding.encode(text)), f"tiktoken:{encoding_name}"
        except Exception:
            pass

    # Rough multilingual fallback: 3 chars/token is safer than 4 for Vietnamese.
    return max(1, (len(text) + 2) // 3), "approx_chars_per_3"


def enrich_chunk_metadata(
    chunks: list[Document],
    *,
    token_encoding: str = "cl100k_base",
) -> list[Document]:
    """Stamp stable chunk metadata used by indexing and source tracing."""
    per_source_index: dict[str, int] = {}
    for chunk in chunks:
        source = str(
            chunk.metadata.get("source")
            or chunk.metadata.get("file_name")
            or "unknown"
        )
        page_number = chunk.metadata.get("page_number")
        if page_number is not None:
            chunk.metadata["page"] = page_number
        elif chunk.metadata.get("page") is not None:
            chunk.metadata["page_number"] = chunk.metadata["page"]

        chunk_index = per_source_index.get(source, 0)
        per_source_index[source] = chunk_index + 1

        source_hash = hashlib.md5(source.encode("utf-8", errors="replace")).hexdigest()[:10]
        text_hash = hashlib.md5(
            chunk.page_content.encode("utf-8", errors="replace")
        ).hexdigest()[:10]

        chunk.metadata["chunk_id"] = f"{source_hash}:{chunk_index:06d}:{text_hash}"
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["char_count"] = len(chunk.page_content)
        token_count, token_method = estimate_token_count(
            chunk.page_content,
            encoding_name=token_encoding,
        )
        chunk.metadata["token_count"] = token_count
        chunk.metadata["token_count_method"] = token_method
    return chunks
