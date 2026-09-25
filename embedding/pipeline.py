"""
embedding/pipeline.py
=====================
EmbeddingPipeline — combines a dense embedder with an optional sparse embedder.

This is the object that vector_db.py and retrieval.py use.
It produces both dense and sparse vectors in a single interface.

Usage
-----
    pipeline = EmbeddingPipeline(
        dense_provider="cohere",
        dense_model="embed-v4.0",
        enable_sparse=True,
        sparse_method="bm25",
    )
    pipeline.fit_sparse(corpus_texts)        # required for BM25
    result = pipeline.embed_documents(texts)
    # result["dense"]  -> list[list[float]]
    # result["sparse"] -> list[dict[str, float]] or None
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from embedding.core.cache import EmbeddingCache, make_embedding_cache_key
from embedding.core.records import stable_text_id
from embedding.core.text import (
    embedding_text_quality_flags,
    is_embedding_text_valid,
    normalize_embedding_text,
)
from embedding.core.validation import (
    REQUIRED_INDEX_METADATA,
    l2_normalize_vector,
    validate_embedding_record,
    validate_vector_dimensions,
)
from embedding.sparse import SparseMethod, get_sparse_embedder


_KNOWN_DENSE_DIMENSIONS: dict[tuple[str, str], int] = {
    ("cohere", "embed-v4.0"): 1024,
}


class EmbeddingPipeline:
    """
    Unified dense API + optional BM25 sparse embedding interface.

    Parameters
    ----------
    dense_provider : Dense embedding provider, default "cohere".
    dense_model    : Dense embedding model, default "embed-v4.0".
    dense_kwargs   : Extra kwargs forwarded to the dense embedder constructor.
    enable_sparse  : Also compute BM25 sparse vectors.
    sparse_method  : Must be "bm25".
    """

    def __init__(
        self,
        dense_provider: str          = "cohere",
        dense_model:    str          = "embed-v4.0",
        dense_kwargs:   dict | None  = None,
        enable_sparse:  bool         = False,
        sparse_method:  SparseMethod = "bm25",
        expected_dimension: int | None = None,
        metric: str = "cosine",
        normalize_dense: bool = True,
        min_text_chars: int = 20,
        use_cache: bool = True,
        cache_path: str | None = None,
        strict_metadata: bool = False,
        required_metadata: tuple[str, ...] = REQUIRED_INDEX_METADATA,
    ):
        from embedding.factory import get_embedder

        self.dense_provider = dense_provider
        self.dense_model = dense_model
        self.enable_sparse = enable_sparse
        self.sparse_method = sparse_method
        self.metric = metric
        self.normalize_dense = normalize_dense
        self.min_text_chars = min_text_chars
        self.use_cache = use_cache
        self.cache_path = cache_path
        self.strict_metadata = strict_metadata
        self.required_metadata = required_metadata
        self.expected_dimension = expected_dimension or _KNOWN_DENSE_DIMENSIONS.get(
            (dense_provider, dense_model)
        )
        self._dense = get_embedder(dense_provider, dense_model, **(dense_kwargs or {}))
        self._sparse = get_sparse_embedder(sparse_method) if enable_sparse else None
        self._dense_cache: dict[str, list[float]] = {}
        self._persistent_cache = EmbeddingCache(cache_path) if cache_path else None

    # ------------------------------------------------------------------
    # Corpus fitting (BM25 requires this before embedding)
    # ------------------------------------------------------------------

    def fit_sparse(self, corpus_texts: list[str]) -> "EmbeddingPipeline":
        """Fit BM25 IDF statistics on the corpus."""
        if self._sparse:
            self._sparse.fit(corpus_texts)
        return self

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> dict[str, Any]:
        """
        Embed a list of document texts.

        Returns
        -------
        {
            "dense":  list[list[float]],           # always present
            "sparse": list[dict[str, float]] | None # only if enable_sparse
        }
        """
        clean_texts = [normalize_embedding_text(text) for text in texts]
        invalid = [
            index
            for index, text in enumerate(clean_texts)
            if not is_embedding_text_valid(text, min_chars=self.min_text_chars)
        ]
        if invalid:
            raise ValueError(
                f"Refusing to embed empty or too-short chunks at indexes={invalid[:10]}."
            )

        return {
            "dense": self._embed_dense_documents(clean_texts),
            "sparse": self._sparse.embed_documents(clean_texts) if self._sparse else None,
        }

    def embed_query(self, query: str) -> dict[str, Any]:
        """
        Embed a single query string.

        Returns
        -------
        {
            "dense":  list[float],
            "sparse": dict[str, float] | None
        }
        """
        clean_query = normalize_embedding_text(query)
        if not is_embedding_text_valid(clean_query, min_chars=min(8, self.min_text_chars)):
            raise ValueError("Refusing to embed an empty or too-short query.")

        dense = self._dense.embed_query(clean_query)
        if self.normalize_dense:
            dense = l2_normalize_vector(dense)
        validate_vector_dimensions([dense], self.expected_dimension)
        return {
            "dense": dense,
            "sparse": self._sparse.embed_query(clean_query) if self._sparse else None,
        }

    def embed_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Embed {"text": ..., "metadata": ...} records for indexing.

        Invalid empty/tiny chunks are skipped. The returned records are ready for
        vector DB insertion and keep vector/text/metadata aligned.
        """
        prepared: list[tuple[str, str, dict[str, Any]]] = []
        for index, record in enumerate(records):
            text = normalize_embedding_text(str(record.get("text") or ""))
            if not is_embedding_text_valid(text, min_chars=self.min_text_chars):
                continue

            metadata = dict(record.get("metadata") or {})
            record_id = (
                str(record.get("id"))
                if record.get("id")
                else str(metadata.get("chunk_id") or stable_text_id(text))
            )
            metadata.setdefault("chunk_id", record_id)
            metadata.setdefault("chunk_index", index)
            metadata["embedding_model"] = self.dense_model
            metadata["embedding_provider"] = self.dense_provider
            metadata["embedding_dimension"] = self.expected_dimension
            metadata["embedding_metric"] = self.metric
            quality_flags = embedding_text_quality_flags(text)
            if quality_flags:
                metadata["embedding_text_quality_flags"] = quality_flags
            prepared.append((record_id, text, metadata))

        if not prepared:
            return []

        texts = [item[1] for item in prepared]
        if (
            self._sparse
            and getattr(self._sparse, "_bm25", None) is None
            and getattr(self._sparse, "_idf", None) is None
        ):
            self.fit_sparse(texts)

        dense_vectors = self._embed_dense_documents(texts)
        sparse_vectors = self._sparse.embed_documents(texts) if self._sparse else None

        output: list[dict[str, Any]] = []
        for index, (record_id, text, metadata) in enumerate(prepared):
            metadata["embedding_dimension"] = len(dense_vectors[index])
            item = {
                "id": record_id,
                "embedding": dense_vectors[index],
                "text": text,
                "metadata": metadata,
            }
            if sparse_vectors is not None:
                item["sparse_embedding"] = sparse_vectors[index]
            warnings = validate_embedding_record(
                item,
                expected_dimension=self.expected_dimension,
                required_metadata=self.required_metadata,
                strict_metadata=self.strict_metadata,
            )
            if warnings:
                item["validation_warnings"] = warnings
            output.append(item)
        return output

    def _embed_dense_documents(self, texts: list[str]) -> list[list[float]]:
        if not self.use_cache:
            vectors = self._dense.embed_documents(texts)
            return self._postprocess_dense_vectors(vectors)

        vectors_by_text: dict[str, list[float]] = {}
        missing: list[str] = []
        for text in texts:
            cached = self._get_cached_dense_vector(text)
            if cached is None:
                missing.append(text)
            else:
                vectors_by_text[text] = cached
        if missing:
            unique_missing = list(dict.fromkeys(missing))
            embedded = self._postprocess_dense_vectors(
                self._dense.embed_documents(unique_missing)
            )
            for text, vector in zip(unique_missing, embedded):
                self._set_cached_dense_vector(text, vector)
                vectors_by_text[text] = vector

        for text in texts:
            if text not in vectors_by_text:
                vectors_by_text[text] = self._dense_cache[text]
        return [vectors_by_text[text] for text in texts]

    def _cache_key(self, text: str) -> str:
        return make_embedding_cache_key(
            provider=self.dense_provider,
            model=self.dense_model,
            expected_dimension=self.expected_dimension,
            normalize_dense=self.normalize_dense,
            text=text,
        )

    def _get_cached_dense_vector(self, text: str) -> list[float] | None:
        if text in self._dense_cache:
            return self._dense_cache[text]
        if self._persistent_cache is None:
            return None
        vector = self._persistent_cache.get(self._cache_key(text))
        if vector is not None:
            validate_vector_dimensions([vector], self.expected_dimension)
            self._dense_cache[text] = vector
        return vector

    def _set_cached_dense_vector(self, text: str, vector: list[float]) -> None:
        self._dense_cache[text] = vector
        if self._persistent_cache is not None:
            self._persistent_cache.set(self._cache_key(text), vector)

    def _postprocess_dense_vectors(self, vectors: list[list[float]]) -> list[list[float]]:
        if self.normalize_dense:
            vectors = [l2_normalize_vector(vector) for vector in vectors]
        detected = validate_vector_dimensions(vectors, self.expected_dimension)
        if self.expected_dimension is None and detected:
            self.expected_dimension = detected
        return vectors

    @property
    def langchain_embedder(self) -> Embeddings:
        """
        The underlying LangChain Embeddings object.

        Pass this to a LangChain-compatible vector store when needed.
        """
        return self._dense.embedder
