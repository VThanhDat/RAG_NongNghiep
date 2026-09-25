"""Managed Pinecone vector store provider."""

from __future__ import annotations

import os
import time
import logging
from typing import Any

from langchain_core.documents import Document

from embedding.core.text import is_embedding_text_valid, normalize_embedding_text
from vector_db.core.base import BaseVectorStore

logger = logging.getLogger(__name__)


class PineconeVectorStore(BaseVectorStore):
    """Pinecone adapter used by the existing hybrid retriever."""

    def __init__(
        self,
        collection_name: str = "rag_collection",
        index_name: str | None = None,
        namespace: str | None = None,
        dimension: int = 768,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        api_key: str | None = None,
        batch_size: int = 100,
        force_reindex: bool = False,
        strict_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(collection_name, force_reindex, strict_metadata)
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "rag-nong-nghiep")
        self.namespace = namespace or os.getenv("PINECONE_NAMESPACE", collection_name)
        self.dimension = int(dimension)
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.batch_size = max(1, int(batch_size))
        self._client = None
        self._index = None
        self._embedder = None
        self.n_vectors = 0

    def _connect(self, embedder=None):
        if self._index is not None:
            return self._index
        if not self.api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not configured. Add it to .env or the deployment secret store."
            )
        from pinecone import Pinecone, ServerlessSpec

        self._client = Pinecone(api_key=self.api_key)
        names = self._index_names()
        if self.index_name not in names:
            dimension = int(getattr(embedder, "expected_dimension", None) or self.dimension)
            self._client.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
            for _ in range(60):
                description = self._client.describe_index(self.index_name)
                ready = getattr(getattr(description, "status", None), "ready", None)
                if ready is None and isinstance(description, dict):
                    ready = (description.get("status") or {}).get("ready")
                if ready:
                    break
                time.sleep(1)
        description = self._client.describe_index(self.index_name)
        actual_dimension = getattr(description, "dimension", None)
        if actual_dimension is None and isinstance(description, dict):
            actual_dimension = description.get("dimension")
        expected_dimension = int(getattr(embedder, "expected_dimension", None) or self.dimension)
        if actual_dimension is not None and int(actual_dimension) != expected_dimension:
            raise ValueError(
                f"Pinecone index '{self.index_name}' has dimension {actual_dimension}, "
                f"but the configured embedding has dimension {expected_dimension}. "
                "Create a new index or reconfigure the embedding dimension."
            )
        self._index = self._client.Index(self.index_name)
        return self._index

    def _index_names(self) -> set[str]:
        listed = self._client.list_indexes()
        names = getattr(listed, "names", None)
        if callable(names):
            return set(names())
        if names is not None:
            return set(names)
        if isinstance(listed, dict):
            return {str(item.get("name")) for item in listed.get("indexes", [])}
        return {
            str(getattr(item, "name", ""))
            for item in listed
            if getattr(item, "name", None)
        }

    def get_or_create(self, chunks: list[Document], embedder):
        self._embedder = embedder
        self._connect(embedder)
        if self.force_reindex:
            try:
                self._index.delete(delete_all=True, namespace=self.namespace)
            except Exception as exc:
                # Pinecone returns 404 when force-reindex is used before the
                # namespace has received its first upsert. Upsert creates it.
                message = str(exc).lower()
                if "namespace not found" in message or "404" in message:
                    logger.info(
                        "Pinecone namespace '%s' does not exist yet; continuing with first upsert.",
                        self.namespace,
                    )
                else:
                    raise
        self.n_vectors = len(self._upsert_documents(chunks, embedder))
        return self

    def load(self, embedder):
        self._embedder = embedder
        self._connect(embedder)
        return self

    def _upsert_documents(self, chunks: list[Document], embedder) -> list[str]:
        min_text_chars = int(getattr(embedder, "min_text_chars", 20) or 20)
        valid_chunks: list[Document] = []
        skipped = 0
        for chunk in chunks:
            normalized = normalize_embedding_text(chunk.page_content)
            if not is_embedding_text_valid(normalized, min_chars=min_text_chars):
                skipped += 1
                continue
            valid_chunks.append(
                Document(page_content=normalized, metadata=dict(chunk.metadata or {}))
            )
        if skipped:
            logger.warning(
                "Pinecone: skipped %d empty/too-short chunks before embedding (min_text_chars=%d).",
                skipped,
                min_text_chars,
            )

        documents, ids = self.prepare_documents(valid_chunks)
        if not documents:
            return []
        vectors = embedder.embed_documents([doc.page_content for doc in documents])["dense"]
        payload = [
            {
                "id": doc_id,
                "values": vector,
                "metadata": {**doc.metadata, "_text": doc.page_content},
            }
            for doc_id, doc, vector in zip(ids, documents, vectors)
        ]
        for start in range(0, len(payload), self.batch_size):
            self._index.upsert(
                vectors=payload[start : start + self.batch_size],
                namespace=self.namespace,
            )
        return ids

    def add_documents(self, documents: list[Document], ids: list[str] | None = None, **kwargs):
        prepared = documents
        if ids:
            prepared = [
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "chunk_id": doc_id},
                )
                for doc, doc_id in zip(documents, ids)
            ]
        return self._upsert_documents(prepared, self._embedder)

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        **kwargs,
    ):
        documents = [
            Document(
                page_content=text,
                metadata=(metadatas[index] if metadatas and index < len(metadatas) else {}),
            )
            for index, text in enumerate(texts)
        ]
        if ids:
            for doc, doc_id in zip(documents, ids):
                doc.metadata["chunk_id"] = doc_id
        return self._upsert_documents(documents, self._embedder)

    def _query_vector(self, query: str) -> list[float]:
        result = self._embedder.embed_query(query)
        return result["dense"] if isinstance(result, dict) else result

    @staticmethod
    def _pinecone_filter(filter_value: dict | None) -> dict | None:
        if not filter_value:
            return None
        converted: dict = {}
        for key, expected in filter_value.items():
            if isinstance(expected, (list, tuple, set)):
                converted[key] = {"$in": list(expected)}
            elif isinstance(expected, dict):
                converted[key] = expected
            else:
                converted[key] = {"$eq": expected}
        return converted

    @staticmethod
    def _match_value(match, key: str, default=None):
        value = getattr(match, key, None)
        if value is None and isinstance(match, dict):
            value = match.get(key)
        return default if value is None else value

    def _search(self, query: str, k: int = 4, filter: dict | None = None):
        query_kwargs = {
            "vector": self._query_vector(query),
            "top_k": k,
            "include_metadata": True,
            "namespace": self.namespace,
        }
        pinecone_filter = self._pinecone_filter(filter)
        if pinecone_filter:
            query_kwargs["filter"] = pinecone_filter
        response = self._index.query(**query_kwargs)
        matches = self._match_value(response, "matches", [])
        results = []
        for match in matches:
            metadata = dict(self._match_value(match, "metadata", {}) or {})
            text = metadata.pop("_text", "")
            score = float(self._match_value(match, "score", 0.0) or 0.0)
            results.append(
                (
                    Document(
                        page_content=str(text),
                        metadata={**metadata, "relevance_score": score},
                    ),
                    score,
                )
            )
        return results

    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None, **kwargs):
        return [doc for doc, _ in self._search(query, k=k, filter=filter)]

    def similarity_search_with_score(self, query: str, k: int = 4, filter: dict | None = None, **kwargs):
        return self._search(query, k=k, filter=filter)

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4, filter: dict | None = None, **kwargs):
        return self._search(query, k=k, filter=filter)

    def delete(self, ids: list[str] | None = None, delete_all: bool = False, **kwargs):
        if delete_all:
            return self._index.delete(delete_all=True, namespace=self.namespace)
        if ids:
            return self._index.delete(ids=[str(item) for item in ids], namespace=self.namespace)
        return None

    def delete_by_source(self, source: str):
        return self._index.delete(
            filter={"source": {"$eq": source}},
            namespace=self.namespace,
        )

    def upsert_documents(self, documents: list[Document]) -> list[str]:
        return self._upsert_documents(documents, self._embedder)
