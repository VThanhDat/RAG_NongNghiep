"""Factory for the local BM25 sparse embedder."""

from __future__ import annotations

from typing import Literal

from embedding.sparse.bm25 import BM25Embedder

SparseMethod = Literal["bm25"]


def get_sparse_embedder(method: SparseMethod = "bm25") -> BM25Embedder:
    """Return a BM25 sparse embedder instance."""
    if method == "bm25":
        return BM25Embedder()
    raise ValueError("Only BM25 sparse embeddings are supported.")
