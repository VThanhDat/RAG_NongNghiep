"""
Post-retrieval stage for RAG Nong Nghiep.

Supported processing:
    exact deduplication -> score filtering -> Jaccard redundancy filtering
    -> relevance ordering -> top_n context documents
"""

from post_retrieval.factory import build_pipeline, build_pipeline_from_config
from post_retrieval.pipeline import PostRetrievalPipeline
from post_retrieval.core.types import PostRetrievalResult

__all__ = [
    "build_pipeline",
    "build_pipeline_from_config",
    "PostRetrievalPipeline",
    "PostRetrievalResult",
]
