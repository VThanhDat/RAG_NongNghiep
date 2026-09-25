"""
Retrieval stage for RAG Nong Nghiep.

Public API:
    from retrieval import get_retriever, build_retriever_from_config, RetrievalResult

Supported strategy:
    hybrid = Pinecone dense search + BM25 sparse search + RRF fusion
"""

from retrieval.core.types import RetrievalResult
from retrieval.core.utils import add_rank_scores
from retrieval.factory import build_retriever_from_config, get_retriever

__all__ = [
    "get_retriever",
    "build_retriever_from_config",
    "add_rank_scores",
    "RetrievalResult",
]
