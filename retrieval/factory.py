"""
Factory for the simplified retrieval stage.

Only the local hybrid retriever is supported:
Pinecone dense search + BM25 sparse search + RRF fusion.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.core.base import BaseRetriever

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY = "hybrid"
_REGISTRY = {
    DEFAULT_STRATEGY: ("retrieval.strategies.hybrid", "HybridRetriever"),
}


def get_retriever(
    strategy: str,
    vector_store: VectorStore,
    documents: list[Document] | None = None,
    **kwargs: Any,
) -> BaseRetriever:
    if strategy != DEFAULT_STRATEGY:
        raise ValueError("Only hybrid retrieval is supported.")
    if not documents:
        raise ValueError("Hybrid retrieval requires the indexed chunk documents for BM25.")

    module_path, class_name = _REGISTRY[strategy]
    cls = getattr(importlib.import_module(module_path), class_name)
    logger.info("Retriever: strategy=%s top_k=%s", strategy, kwargs.get("top_k", 5))
    return cls(vector_store=vector_store, documents=documents, **kwargs)


def build_retriever_from_config(
    cfg: dict,
    vector_store: VectorStore,
    documents: list[Document] | None = None,
) -> BaseRetriever:
    ret_cfg = cfg.get("query_pipeline", {}).get("retrieval", {})
    strategy = ret_cfg.get("strategy", DEFAULT_STRATEGY)
    top_k = ret_cfg.get("top_k", 10)

    return get_retriever(
        strategy=strategy,
        vector_store=vector_store,
        documents=documents,
        top_k=top_k,
        candidate_k=ret_cfg.get("candidate_k", top_k * 3),
        rrf_k=ret_cfg.get("rrf_k", 60),
        score_threshold=ret_cfg.get("score_threshold", 0.0),
    )
