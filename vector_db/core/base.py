"""
vector_db/base.py
=================
Abstract base class cho mọi vector store provider.

Giao kèo:
    store = get_vector_store(provider, chunks, embedder, **cfg)
    results = store.similarity_search(query, k=5)
"""

from __future__ import annotations

import json
import hashlib
import types
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


REQUIRED_VECTOR_METADATA = (
    "source",
    "file_name",
    "page_number",
    "title",
    "section",
    "chunk_id",
    "chunk_index",
)


class BaseVectorStore(ABC):
    """
    Abstract wrapper around a LangChain VectorStore.

    Parameters
    ----------
    collection_name : Name of the collection / index / table.
    force_reindex   : Wipe existing data and rebuild from scratch.
    """

    def __init__(
        self,
        collection_name: str = "rag",
        force_reindex: bool = False,
        strict_metadata: bool = False,
    ):
        self.collection_name = collection_name
        self.force_reindex   = force_reindex
        self.strict_metadata = strict_metadata

    @abstractmethod
    def get_or_create(
        self,
        chunks:   list[Document],
        embedder,                   # EmbeddingPipeline
    ) -> VectorStore:
        """
        Return a populated VectorStore, creating or loading as needed.

        Parameters
        ----------
        chunks   : Chunk Documents from the chunking stage.
        embedder : ``EmbeddingPipeline`` from the embedding stage.
        """

    # ── Shared utilities ──────────────────────────────────────────────────────

    def _langchain_embedder(self, embedder):
        """Extract the LangChain Embeddings object from an EmbeddingPipeline."""
        # EmbeddingPipeline exposes .langchain_embedder
        if hasattr(embedder, "langchain_embedder"):
            return embedder.langchain_embedder
        # Plain LangChain Embeddings object passed directly
        return embedder

    def prepare_documents(self, docs: list[Document]) -> tuple[list[Document], list[str]]:
        """
        Validate/sanitize documents and return stable vector IDs.

        ``chunk_id`` is used as the primary vector ID so updates/deletes can be
        addressed deterministically across re-indexing runs.
        """
        report = self.validate_documents(docs, strict_metadata=self.strict_metadata)
        if report["duplicate_chunk_ids"]:
            raise ValueError(
                "Duplicate chunk_id values found before vector insertion: "
                f"{report['duplicate_chunk_ids'][:10]}"
            )
        sanitized = self.sanitize_metadata(docs)
        ids = [self.document_id(doc) for doc in sanitized]
        for doc, doc_id in zip(sanitized, ids):
            doc.id = doc_id
            doc.metadata.setdefault("chunk_id", doc_id)
        return sanitized, ids

    @staticmethod
    def fingerprint_context(embedder, *, metric: str | None = None) -> dict[str, Any]:
        """Build a fingerprint context from embedding/vector settings."""
        return {
            "embedding_provider": getattr(embedder, "dense_provider", None),
            "embedding_model": getattr(embedder, "dense_model", None),
            "embedding_dimension": getattr(embedder, "expected_dimension", None),
            "embedding_metric": metric or getattr(embedder, "metric", None),
            "normalize_dense": getattr(embedder, "normalize_dense", None),
        }

    @staticmethod
    def document_id(doc: Document) -> str:
        """Return a stable vector ID for one document."""
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id not in (None, ""):
            return str(chunk_id)
        hasher = hashlib.md5()
        hasher.update(str(doc.metadata.get("source", "")).encode("utf-8", errors="replace"))
        hasher.update(doc.page_content.encode("utf-8", errors="replace"))
        return hasher.hexdigest()

    @staticmethod
    def sanitize_metadata(docs: list[Document]) -> list[Document]:
        """
        Normalize Document metadata so every value is str | int | float | bool.

        Vector stores commonly reject metadata
        that contains list/dict/None values.  This method converts them:
          - None        → key dropped entirely
          - list / dict → JSON string via json.dumps()
          - anything else (e.g. Path, datetime) → str()

        Returns a new list of Documents; originals are not mutated.
        """
        _SCALAR = (str, int, float, bool)
        result: list[Document] = []
        for doc in docs:
            clean: dict = {}
            for k, v in doc.metadata.items():
                if v is None:
                    continue
                if isinstance(v, _SCALAR):
                    clean[k] = v
                elif isinstance(v, (list, dict)):
                    clean[k] = json.dumps(v, ensure_ascii=False, default=str)
                else:
                    clean[k] = str(v)
            result.append(Document(page_content=doc.page_content, metadata=clean))
        return result

    @staticmethod
    def validate_documents(
        docs: list[Document],
        *,
        required_metadata: tuple[str, ...] = REQUIRED_VECTOR_METADATA,
        strict_metadata: bool = False,
    ) -> dict[str, Any]:
        """Return an integrity report for documents before vector insertion."""
        seen: set[str] = set()
        duplicate_chunk_ids: list[str] = []
        missing_metadata: list[dict[str, Any]] = []
        empty_text_indexes: list[int] = []

        for index, doc in enumerate(docs):
            if not doc.page_content or not doc.page_content.strip():
                empty_text_indexes.append(index)

            metadata = doc.metadata or {}
            chunk_id = metadata.get("chunk_id")
            if chunk_id not in (None, ""):
                chunk_id_str = str(chunk_id)
                if chunk_id_str in seen:
                    duplicate_chunk_ids.append(chunk_id_str)
                seen.add(chunk_id_str)

            missing = [
                key
                for key in required_metadata
                if key not in metadata or metadata.get(key) in (None, "")
            ]
            if missing:
                missing_metadata.append({
                    "index": index,
                    "chunk_id": chunk_id,
                    "missing": missing,
                })

        report = {
            "total": len(docs),
            "empty_text_indexes": empty_text_indexes,
            "missing_metadata": missing_metadata,
            "duplicate_chunk_ids": duplicate_chunk_ids,
        }
        if strict_metadata and (empty_text_indexes or missing_metadata or duplicate_chunk_ids):
            raise ValueError(f"Vector DB document integrity check failed: {report}")
        return report

    @staticmethod
    def format_search_result(result: Any) -> dict[str, Any]:
        """Normalize a LangChain search result into the API shape used by RAG."""
        score = None
        doc = result
        if isinstance(result, tuple) and len(result) >= 2:
            doc, score = result[0], result[1]
        metadata = dict(getattr(doc, "metadata", {}) or {})
        doc_id = metadata.get("chunk_id") or getattr(doc, "id", None)
        return {
            "id": doc_id,
            "score": float(score) if score is not None else None,
            "text": getattr(doc, "page_content", ""),
            "metadata": metadata,
        }

    def attach_common_ops(self, store: VectorStore) -> VectorStore:
        """Attach convenience ops without hiding the underlying LangChain store."""

        def search(
            vector_store,
            query: str,
            k: int = 5,
            filter: dict | None = None,
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            search_kwargs = {"k": k, **kwargs}
            if filter is not None:
                search_kwargs["filter"] = filter
            if hasattr(vector_store, "similarity_search_with_score"):
                raw = vector_store.similarity_search_with_score(query, **search_kwargs)
            else:
                raw = vector_store.similarity_search(query, **search_kwargs)
            return [BaseVectorStore.format_search_result(item) for item in raw]

        def delete_by_chunk_id(vector_store, chunk_id: str) -> Any:
            if not hasattr(vector_store, "delete"):
                raise NotImplementedError("This vector store does not expose delete(ids=...).")
            return vector_store.delete(ids=[str(chunk_id)])

        def delete_by_source(vector_store, source: str) -> Any:
            collection = getattr(vector_store, "_collection", None)
            if collection is not None and hasattr(collection, "delete"):
                return collection.delete(where={"source": source})
            docstore = getattr(vector_store, "docstore", None)
            docstore_dict = getattr(docstore, "_dict", None)
            if isinstance(docstore_dict, dict) and hasattr(vector_store, "delete"):
                ids = [
                    doc_id
                    for doc_id, doc in docstore_dict.items()
                    if getattr(doc, "metadata", {}).get("source") == source
                ]
                if ids:
                    return vector_store.delete(ids=ids)
                return False
            raise NotImplementedError(
                "delete_by_source is only available for stores exposing metadata deletes."
            )

        def upsert_documents(vector_store, documents: list[Document]) -> list[str]:
            prepared, ids = self.prepare_documents(documents)
            if hasattr(vector_store, "delete"):
                try:
                    vector_store.delete(ids=ids)
                except Exception:
                    pass
            if not hasattr(vector_store, "add_documents"):
                raise NotImplementedError("This vector store does not expose add_documents(...).")
            vector_store.add_documents(prepared, ids=ids)
            return ids

        for name, func in {
            "search": search,
            "delete_by_chunk_id": delete_by_chunk_id,
            "delete_by_source": delete_by_source,
            "upsert_documents": upsert_documents,
        }.items():
            try:
                setattr(store, name, types.MethodType(func, store))
            except Exception:
                pass
        return store
