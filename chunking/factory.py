"""
Factory and config builder for the simplified agriculture RAG chunker.

Only recursive chunking is supported. It is a good default for PDF text:
paragraphs -> lines -> sentences -> words -> characters.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from chunking.core.base import BaseChunker

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, tuple[str, str]] = {
    "recursive": ("chunking.strategies.recursive", "RecursiveChunker"),
}


def get_chunker(strategy: str, **kwargs: Any) -> BaseChunker:
    """Instantiate a chunker by strategy name."""
    entry = _REGISTRY.get(strategy)
    if entry is None:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Valid: {valid}")

    module_path, class_name = entry
    import importlib

    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)


def chunk_documents_from_config(docs: list[Document], cfg: dict) -> list[Document]:
    """Build and run the recursive chunker from config.yaml."""
    chunk_cfg = cfg["indexing"]["chunking"]
    strategy = chunk_cfg.get("strategy", "recursive")
    token_encoding = chunk_cfg.get("token_encoding", "cl100k_base")

    chunker = get_chunker(
        strategy,
        chunk_size=chunk_cfg.get("chunk_size", 1000),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 150),
        token_encoding=token_encoding,
    )
    chunks = chunker.split(docs)

    if chunk_cfg.get("merge_tiny_chunks", False):
        from chunking.postprocess.merge import merge_tiny_chunks

        chunks = merge_tiny_chunks(
            chunks,
            min_tokens=chunk_cfg.get("min_chunk_tokens", 80),
            max_tokens=chunk_cfg.get("max_chunk_tokens", 1000),
            token_encoding=token_encoding,
        )

    if chunk_cfg.get("deduplicate_chunks", False):
        from chunking.postprocess.deduplication import deduplicate_chunks

        chunks = deduplicate_chunks(
            chunks,
            method=chunk_cfg.get("deduplicate_method", "exact"),
        )

    logger.info("Chunking ('%s'): %d docs -> %d chunks.", strategy, len(docs), len(chunks))
    return chunks
