"""Factory functions for the Pinecone vector DB stage."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "pinecone"

_REGISTRY: dict[str, tuple[str, str]] = {
    "pinecone": ("vector_db.providers.pinecone_store", "PineconeVectorStore"),
}


def _metric(cfg_metric: str | None, default: str = "cosine") -> str:
    return str(cfg_metric or default).strip().lower()


def get_vector_store(
    provider: str,
    chunks: list[Document],
    embedder,
    force_reindex: bool = False,
    **kwargs: Any,
) -> VectorStore:
    """Build or load the managed Pinecone vector store."""
    if provider not in _REGISTRY:
        raise ValueError(f"Unsupported vector DB provider: {provider}.")

    import importlib

    module_path, class_name = _REGISTRY[provider]
    cls = getattr(importlib.import_module(module_path), class_name)
    allowed = {
        "collection_name",
        "index_name",
        "namespace",
        "dimension",
        "metric",
        "cloud",
        "region",
        "api_key",
        "batch_size",
        "strict_metadata",
    }
    init_kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    inst = cls(force_reindex=force_reindex, **init_kwargs)

    if hasattr(embedder, "enable_sparse") and embedder.enable_sparse:
        logger.info("Fitting BM25 sparse embedder on %d documents.", len(chunks))
        embedder.fit_sparse([chunk.page_content for chunk in chunks])

    logger.info(
        "VectorStore: provider=%s, collection=%s, chunks=%d",
        provider,
        kwargs.get("collection_name", "rag"),
        len(chunks),
    )
    return inst.get_or_create(chunks, embedder)


def get_vector_store_from_config(
    chunks: list[Document],
    embedder,
    cfg: dict,
    force_reindex: bool = False,
) -> VectorStore:
    """Build a Pinecone vector store from ``indexing.vector_db`` config."""
    indexing = cfg.get("indexing", {})
    db_cfg = indexing.get("vector_db", {})
    emb_cfg = indexing.get("embedding", {})
    provider = db_cfg.get("provider", DEFAULT_PROVIDER)
    metric = _metric(db_cfg.get("metric") or emb_cfg.get("metric"), "cosine")

    return get_vector_store(
        provider=provider,
        chunks=chunks,
        embedder=embedder,
        force_reindex=force_reindex or db_cfg.get("force_reindex", False),
        collection_name=db_cfg.get("collection_name", "rag_collection_cohere"),
        index_name=db_cfg.get("index_name"),
        namespace=db_cfg.get("namespace"),
        dimension=db_cfg.get("dimension") or emb_cfg.get("expected_dimension", 1024),
        metric=metric,
        cloud=db_cfg.get("cloud", "aws"),
        region=db_cfg.get("region", "us-east-1"),
        api_key=db_cfg.get("api_key"),
        batch_size=db_cfg.get("batch_size", 96),
        strict_metadata=db_cfg.get(
            "strict_metadata",
            emb_cfg.get("strict_metadata", False),
        ),
    )


def load_vector_store_from_config(
    chunks: list[Document],
    embedder,
    cfg: dict,
) -> VectorStore:
    """Load an existing Pinecone vector store and refit BM25 on chunks."""
    indexing = cfg.get("indexing", {})
    db_cfg = indexing.get("vector_db", {})
    emb_cfg = indexing.get("embedding", {})
    provider = db_cfg.get("provider", DEFAULT_PROVIDER)
    if provider not in _REGISTRY:
        raise ValueError(f"Unsupported vector DB provider: {provider}.")

    if hasattr(embedder, "enable_sparse") and embedder.enable_sparse:
        logger.info("Fitting BM25 sparse embedder on %d loaded documents.", len(chunks))
        embedder.fit_sparse([chunk.page_content for chunk in chunks])

    import importlib

    module_path, class_name = _REGISTRY[provider]
    cls = getattr(importlib.import_module(module_path), class_name)
    init_kwargs = {
        "collection_name": db_cfg.get("collection_name", "rag_collection_cohere"),
        "metric": _metric(db_cfg.get("metric") or emb_cfg.get("metric"), "cosine"),
        "force_reindex": False,
        "strict_metadata": db_cfg.get("strict_metadata", emb_cfg.get("strict_metadata", False)),
    }
    init_kwargs.update(
        {
            "index_name": db_cfg.get("index_name"),
            "namespace": db_cfg.get("namespace"),
            "dimension": db_cfg.get("dimension") or emb_cfg.get("expected_dimension", 1024),
            "cloud": db_cfg.get("cloud", "aws"),
            "region": db_cfg.get("region", "us-east-1"),
            "api_key": db_cfg.get("api_key"),
            "batch_size": db_cfg.get("batch_size", 96),
        }
    )
    inst = cls(**init_kwargs)
    return inst.load(embedder)
