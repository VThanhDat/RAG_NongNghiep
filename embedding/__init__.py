"""
Text embedding package for the RAG indexing pipeline.

Default model:
    cohere / embed-v4.0

Folder layout:
    core/       shared base classes, text prep, record ids, vector validation
    providers/  dense embedding provider wrappers
    sparse/     BM25 sparse embedding for hybrid retrieval

Public API:
    from embedding import get_embedder, get_embedder_from_config, EmbeddingPipeline
"""

from embedding.factory import get_embedder, get_embedder_from_config
from embedding.pipeline import EmbeddingPipeline

__all__ = ["get_embedder", "get_embedder_from_config", "EmbeddingPipeline"]
