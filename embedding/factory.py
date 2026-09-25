"""
Factory functions for the dense embedding stage.

Supported dense embedding:
  - provider: cohere
  - model: embed-v4.0

Optional sparse BM25 is local and free.
"""

from __future__ import annotations

import logging
from typing import Any

from embedding.core.base import BaseEmbedder

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "cohere"
DEFAULT_MODEL = "embed-v4.0"

_REGISTRY: dict[str, tuple[str, str]] = {
    "cohere": ("embedding.providers.cohere", "CohereEmbedder"),
}


def get_embedder(provider: str, model_name: str, **kwargs: Any) -> BaseEmbedder:
    """Instantiate a supported API-backed dense embedder."""
    supported_models = {
        "cohere": {"embed-v4.0"},
    }
    if provider not in _REGISTRY:
        raise ValueError(f"Unsupported embedding provider: {provider}.")
    if model_name not in supported_models.get(provider, set()):
        raise ValueError(f"Unsupported model '{model_name}' for provider '{provider}'.")

    import importlib

    module_path, class_name = _REGISTRY[provider]
    cls = getattr(importlib.import_module(module_path), class_name)
    logger.info("Embedding: provider=%s  model=%s", provider, model_name)
    return cls(model_name=model_name, **kwargs)


def get_embedder_from_config(cfg: dict) -> "EmbeddingPipeline":
    """Build an EmbeddingPipeline from the ``indexing.embedding`` config."""
    from embedding.pipeline import EmbeddingPipeline

    emb_cfg = cfg.get("indexing", {}).get("embedding", {})
    provider = emb_cfg.get("provider", DEFAULT_PROVIDER)
    model = emb_cfg.get("model_name", DEFAULT_MODEL)

    if provider == "cohere":
        dense_kwargs = {
            "api_key": emb_cfg.get("api_key"),
            "output_dimension": emb_cfg.get("output_dimension", 1024),
            "batch_size": emb_cfg.get("batch_size", 96),
        }
        expected_dimension = emb_cfg.get(
            "expected_dimension",
            dense_kwargs["output_dimension"],
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}.")

    return EmbeddingPipeline(
        dense_provider=provider,
        dense_model=model,
        dense_kwargs=dense_kwargs,
        enable_sparse=emb_cfg.get("enable_sparse", True),
        sparse_method=emb_cfg.get("sparse_method", "bm25"),
        expected_dimension=expected_dimension,
        metric=emb_cfg.get("metric", "cosine"),
        normalize_dense=emb_cfg.get("normalize_dense", True),
        min_text_chars=emb_cfg.get("min_text_chars", 20),
        use_cache=emb_cfg.get("cache_embeddings", True),
        cache_path=emb_cfg.get("cache_path"),
        strict_metadata=emb_cfg.get("strict_metadata", False),
    )
