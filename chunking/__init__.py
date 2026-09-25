"""
Document chunking package for the agriculture RAG indexing pipeline.

Public API:
    from chunking import get_chunker, chunk_documents_from_config
    from chunking import deduplicate_chunks, to_text_records
"""

from chunking.core.records import to_text_records
from chunking.factory import get_chunker, chunk_documents_from_config
from chunking.postprocess.deduplication import deduplicate_chunks

__all__ = [
    "get_chunker",
    "chunk_documents_from_config",
    "deduplicate_chunks",
    "to_text_records",
]
