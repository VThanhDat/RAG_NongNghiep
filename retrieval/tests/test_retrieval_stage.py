"""
Smoke tests for the simplified retrieval stage.

Run from project root:
    python -B retrieval/tests/test_retrieval_stage.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document

from pre_retrieval.core.types import TransformResult
from retrieval.core.utils import add_rank_scores, deduplicate, is_low_value_document, reciprocal_rank_fusion
from retrieval.factory import build_retriever_from_config, get_retriever
from retrieval.strategies.hybrid import HybridRetriever


class MockVectorStore:
    def similarity_search(self, query: str, k: int, filter=None, **kwargs):
        docs = [
            Document(
                page_content=f"Dense result về bệnh thán thư trên sầu riêng cho query {query}",
                metadata={"source": "dense_1.pdf", "relevance_score": 0.9, "knowledge_group": "sau-benh"},
            ),
            Document(
                page_content="Dense result về bón phân kali cho cây cà phê",
                metadata={"source": "dense_2.pdf", "relevance_score": 0.8, "knowledge_group": "ca-phe"},
            ),
            Document(
                page_content="Dense result về quản lý nước cho lúa",
                metadata={"source": "dense_3.pdf", "relevance_score": 0.7, "knowledge_group": "lua"},
            ),
        ]
        if filter:
            docs = [
                doc
                for doc in docs
                if all(doc.metadata.get(key) == value for key, value in filter.items())
            ]
        return docs[:k]

    def similarity_search_with_relevance_scores(self, query: str, k: int, filter=None, **kwargs):
        return [(doc, doc.metadata.get("relevance_score", 0.0)) for doc in self.similarity_search(query, k, filter)]


def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="Bệnh thán thư trên sầu riêng thường phát triển khi ẩm độ cao.",
            metadata={"source": "sau-rieng.pdf", "page_number": 3, "knowledge_group": "sau-benh"},
        ),
        Document(
            page_content="Thiếu kali làm mép lá cà phê cháy vàng và năng suất giảm.",
            metadata={"source": "ca-phe.pdf", "page_number": 8, "knowledge_group": "ca-phe"},
        ),
        Document(
            page_content="Lúa giai đoạn đẻ nhánh cần quản lý nước và bón phân hợp lý.",
            metadata={"source": "lua.pdf", "page_number": 12, "knowledge_group": "lua"},
        ),
    ]


def test_factory_builds_hybrid_retriever() -> None:
    retriever = get_retriever(
        "hybrid",
        MockVectorStore(),
        documents=sample_documents(),
        top_k=5,
        candidate_k=6,
    )
    if not isinstance(retriever, HybridRetriever):
        raise AssertionError("Factory should build HybridRetriever.")
    if retriever.top_k != 5 or retriever.candidate_k != 6:
        raise AssertionError("Retriever config was not applied.")


def test_factory_rejects_removed_strategies() -> None:
    removed = ["dense", "sparse", "multi_query", "parent_document", "sentence_window", "multi_hop", "contextual"]
    for strategy in removed:
        try:
            get_retriever(strategy, MockVectorStore(), documents=sample_documents())
        except ValueError:
            continue
        raise AssertionError(f"Removed retrieval strategy should be rejected: {strategy}")


def test_factory_requires_documents_for_bm25() -> None:
    try:
        get_retriever("hybrid", MockVectorStore(), documents=None)
    except ValueError as exc:
        if "BM25" not in str(exc):
            raise AssertionError(f"Unexpected error: {exc}")
        return
    raise AssertionError("Hybrid retrieval should require documents.")


def test_build_retriever_from_config() -> None:
    cfg = {
        "query_pipeline": {
            "retrieval": {
                "strategy": "hybrid",
                "top_k": 4,
                "candidate_k": 9,
                "rrf_k": 42,
                "score_threshold": 0.2,
            }
        }
    }
    retriever = build_retriever_from_config(cfg, MockVectorStore(), documents=sample_documents())
    if retriever.top_k != 4 or retriever.candidate_k != 9 or retriever.rrf_k != 42:
        raise AssertionError("Config builder did not apply retrieval options.")


def test_hybrid_retrieval_returns_ranked_documents() -> None:
    retriever = HybridRetriever(
        MockVectorStore(),
        documents=sample_documents(),
        top_k=5,
        candidate_k=3,
    )
    result = TransformResult(
        original_query="Cách phòng bệnh thán thư trên sầu riêng?",
        clean_query="Cách phòng bệnh thán thư trên sầu riêng?",
        queries=["Cách phòng bệnh thán thư trên sầu riêng?"],
    )

    docs = retriever.retrieve(result)
    if not docs:
        raise AssertionError("Hybrid retrieval should return documents.")
    if docs[0].metadata.get("rank") != 1 or "score" not in docs[0].metadata:
        raise AssertionError(f"Retrieved docs must include rank/score metadata: {docs[0].metadata}")
    if not any("thán thư" in doc.page_content for doc in docs):
        raise AssertionError("BM25 agriculture keyword result should be present.")


def test_hybrid_retrieval_filters_by_knowledge_group() -> None:
    retriever = HybridRetriever(
        MockVectorStore(),
        documents=sample_documents(),
        top_k=5,
        candidate_k=3,
    )
    result = TransformResult(
        original_query="Cách quản lý nước cho lúa?",
        clean_query="Cách quản lý nước cho lúa?",
        queries=["Cách quản lý nước cho lúa?"],
        metadata_filter={"knowledge_group": "lua"},
    )

    docs = retriever.retrieve(result)
    if not docs:
        raise AssertionError("Filtered retrieval should return lua documents.")
    if any(doc.metadata.get("knowledge_group") != "lua" for doc in docs):
        raise AssertionError(f"Retrieval leaked documents outside group: {[doc.metadata for doc in docs]}")


def test_reciprocal_rank_fusion_and_deduplication() -> None:
    a = Document(page_content="Same content", metadata={"source": "a"})
    b = Document(page_content="Same content ", metadata={"source": "b"})
    c = Document(page_content="Different content", metadata={"source": "c"})
    fused = reciprocal_rank_fusion([[a, c], [b]], k=60)
    unique = deduplicate(fused)
    if len(unique) != 2:
        raise AssertionError(f"Expected 2 unique docs, got {len(unique)}")
    ranked = add_rank_scores(unique)
    if ranked[0].metadata["rank"] != 1 or "score" not in ranked[0].metadata:
        raise AssertionError("Rank score helper should normalize metadata.")


def test_navigation_chunks_are_not_evidence() -> None:
    toc = Document(
        page_content="Mục lục nội dung 1. Đất 2. Nước 3. Dinh dưỡng 4. Sâu bệnh",
        metadata={},
    )
    evidence = Document(
        page_content="Cây cần dinh dưỡng đa lượng và vi lượng để sinh trưởng; đạm hỗ trợ thân lá.",
        metadata={},
    )
    if not is_low_value_document(toc):
        raise AssertionError("Table-of-contents chunks should be filtered.")
    if is_low_value_document(evidence):
        raise AssertionError("Explanatory chunks should remain eligible.")


def main() -> None:
    test_factory_builds_hybrid_retriever()
    test_factory_rejects_removed_strategies()
    test_factory_requires_documents_for_bm25()
    test_build_retriever_from_config()
    test_hybrid_retrieval_returns_ranked_documents()
    test_hybrid_retrieval_filters_by_knowledge_group()
    test_reciprocal_rank_fusion_and_deduplication()
    test_navigation_chunks_are_not_evidence()
    print("Retrieval smoke test passed.")


if __name__ == "__main__":
    main()
