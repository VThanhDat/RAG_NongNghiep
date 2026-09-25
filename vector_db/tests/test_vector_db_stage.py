"""
Smoke tests for the Pinecone Vector DB stage.

These tests avoid real Pinecone calls by exercising shared wrapper logic
and factory configuration mapping with tiny in-memory fakes.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vector_db.factory as factory
from vector_db.core.base import BaseVectorStore
from vector_db.core.evaluation import evaluate_retrieval


class FakeCollection:
    def __init__(self) -> None:
        self.deleted_where: dict | None = None

    def delete(self, where: dict) -> bool:
        self.deleted_where = where
        return True


class FakeLangChainStore(VectorStore):
    def __init__(self) -> None:
        self._collection = FakeCollection()
        self.deleted_ids: list[str] = []
        self.added_ids: list[str] = []

    @classmethod
    def from_texts(cls, texts: list[str], embedding: Any, **kwargs: Any):
        return cls()

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any):
        return [_doc("chunk-001")]

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any):
        return [(_doc("chunk-001"), 0.91)]

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None, **kwargs: Any):
        ids = list(kwargs.get("ids") or [])
        self.added_ids.extend(ids)
        return ids

    def add_documents(self, documents: list[Document], **kwargs: Any):
        ids = list(kwargs.get("ids") or [])
        self.added_ids.extend(ids)
        return ids

    def delete(self, ids: list[str] | None = None, **kwargs: Any):
        self.deleted_ids.extend(ids or [])
        return True


class DummyVectorStore(BaseVectorStore):
    def get_or_create(self, chunks: list[Document], embedder) -> VectorStore:
        return self.attach_common_ops(FakeLangChainStore())


class FakeEmbedder:
    expected_dimension = 1024
    metric = "cosine"
    dense_provider = "cohere"
    dense_model = "embed-v4.0"
    normalize_dense = True
    enable_sparse = False


def _doc(chunk_id: str = "chunk-001", text: str = "Noi dung ve benh than thu tren sau rieng.") -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "nong-nghiep.pdf",
            "file_name": "nong-nghiep.pdf",
            "page_number": 1,
            "title": "Tai lieu nong nghiep",
            "section": "Phong tru sau benh",
            "chunk_id": chunk_id,
            "chunk_index": 0,
            "document_type": "pdf",
            "category": "nong_nghiep",
        },
    )


def test_prepare_documents_uses_chunk_id_as_stable_vector_id() -> None:
    wrapper = DummyVectorStore()
    docs, ids = wrapper.prepare_documents([_doc("chunk-001")])

    if ids != ["chunk-001"]:
        raise AssertionError(f"Unexpected vector IDs: {ids}")
    if docs[0].id != "chunk-001":
        raise AssertionError("Prepared Document.id must match metadata.chunk_id.")


def test_prepare_documents_rejects_duplicate_chunk_ids() -> None:
    wrapper = DummyVectorStore()
    try:
        wrapper.prepare_documents([_doc("dup"), _doc("dup", text="Noi dung khac.")])
    except ValueError as exc:
        if "Duplicate chunk_id" not in str(exc):
            raise
        return
    raise AssertionError("Duplicate chunk_id values must be rejected.")


def test_common_ops_output_delete_filter_and_upsert() -> None:
    store = DummyVectorStore().get_or_create([_doc()], FakeEmbedder())

    result = store.search("than thu sau rieng", k=1)[0]
    if result["id"] != "chunk-001" or result["score"] != 0.91:
        raise AssertionError(f"Unexpected normalized search result: {result}")
    if not {"id", "score", "text", "metadata"}.issubset(result):
        raise AssertionError(f"Search output shape is incomplete: {result}")

    store.delete_by_chunk_id("chunk-001")
    if store.deleted_ids != ["chunk-001"]:
        raise AssertionError("delete_by_chunk_id must call delete(ids=[chunk_id]).")

    store.delete_by_source("nong-nghiep.pdf")
    if store._collection.deleted_where != {"source": "nong-nghiep.pdf"}:
        raise AssertionError("delete_by_source must call metadata delete by source.")

    ids = store.upsert_documents([_doc("chunk-002")])
    if ids != ["chunk-002"] or "chunk-002" not in store.added_ids:
        raise AssertionError("upsert_documents must add documents with stable IDs.")


def test_factory_defaults_to_pinecone_config() -> None:
    captured: dict[str, Any] = {}
    module = types.ModuleType("vector_db.tests.fake_pinecone_provider")

    class FakePineconeStore:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def get_or_create(self, chunks: list[Document], embedder) -> dict[str, Any]:
            return captured

    module.FakePineconeStore = FakePineconeStore
    sys.modules[module.__name__] = module
    original = dict(factory._REGISTRY)
    try:
        factory._REGISTRY["pinecone"] = (module.__name__, "FakePineconeStore")
        result = factory.get_vector_store_from_config(
            [_doc()],
            FakeEmbedder(),
            {"indexing": {"embedding": {"expected_dimension": 1024}}},
        )
    finally:
        factory._REGISTRY.clear()
        factory._REGISTRY.update(original)
        sys.modules.pop(module.__name__, None)

    if result["collection_name"] != "rag_collection_cohere":
        raise AssertionError("Missing vector_db config should default to the Cohere collection.")
    if result["dimension"] != 1024:
        raise AssertionError("Pinecone dimension should default to the Cohere dimension.")
    if result["metric"] != "cosine":
        raise AssertionError("Pinecone metric should default to cosine.")


def test_factory_rejects_non_pinecone_provider() -> None:
    try:
        factory.get_vector_store(
            provider="unsupported",
            chunks=[_doc()],
            embedder=FakeEmbedder(),
        )
    except ValueError as exc:
        if "Unsupported vector DB provider" not in str(exc):
            raise
        return
    raise AssertionError("Only Pinecone should be supported.")


def test_retrieval_evaluation_metrics() -> None:
    store = DummyVectorStore().get_or_create([_doc()], FakeEmbedder())
    metrics = evaluate_retrieval(
        store,
        [{"query": "than thu sau rieng", "relevant_ids": ["chunk-001"]}],
        k_values=(1, 5),
    )
    expected = {
        "recall@1": 1.0,
        "precision@1": 1.0,
        "recall@5": 1.0,
        "precision@5": 0.2,
        "hit_rate": 1.0,
        "mrr": 1.0,
        "query_count": 1,
    }
    if metrics != expected:
        raise AssertionError(f"Unexpected retrieval metrics: {metrics}")


if __name__ == "__main__":
    test_prepare_documents_uses_chunk_id_as_stable_vector_id()
    test_prepare_documents_rejects_duplicate_chunk_ids()
    test_common_ops_output_delete_filter_and_upsert()
    test_factory_defaults_to_pinecone_config()
    test_factory_rejects_non_pinecone_provider()
    test_retrieval_evaluation_metrics()
    print("Vector DB smoke test passed.")
