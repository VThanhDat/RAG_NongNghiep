"""
vector_db/
==========
Vector storage package for the RAG indexing pipeline.

Public API
----------
    from vector_db import get_vector_store, get_vector_store_from_config

    # Direct usage
    store = get_vector_store(provider="pinecone", chunks=chunks, embedder=pipeline)
    retriever = store.as_retriever(search_kwargs={"k": 10})

    # Config-driven (used by main.py)
    store = get_vector_store_from_config(chunks, pipeline, cfg)

Providers
---------
  pinecone           Managed production vector database.
"""

from vector_db.core.base import BaseVectorStore, REQUIRED_VECTOR_METADATA
from vector_db.core.evaluation import RetrievalCase, evaluate_retrieval
from vector_db.factory import get_vector_store, get_vector_store_from_config

__all__ = [
    "BaseVectorStore",
    "REQUIRED_VECTOR_METADATA",
    "RetrievalCase",
    "evaluate_retrieval",
    "get_vector_store",
    "get_vector_store_from_config",
]
