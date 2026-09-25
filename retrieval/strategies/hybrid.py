"""
Hybrid retrieval for agriculture RAG.

The retriever combines:
- dense search from the managed Pinecone vector store
- sparse keyword search with BM25 over indexed chunks
- Reciprocal Rank Fusion (RRF)
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from retrieval.core.base import BaseRetriever
from retrieval.core.utils import (
    add_rank_scores,
    deduplicate,
    is_low_value_document,
    reciprocal_rank_fusion,
)


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: VectorStore,
        documents: list[Document],
        top_k: int = 10,
        candidate_k: int = 30,
        rrf_k: int = 60,
        score_threshold: float = 0.0,
    ):
        super().__init__(vector_store, top_k)
        self.documents = documents
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.score_threshold = score_threshold

        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([self._tokenize(doc.page_content) for doc in documents])

    def retrieve(self, result) -> list[Document]:
        queries = result.all_queries() if hasattr(result, "all_queries") else [result.original_query]
        metadata_filter = getattr(result, "metadata_filter", None)

        ranked_lists: list[list[Document]] = []
        for query in queries:
            dense_docs = self._dense_search(query, metadata_filter)
            sparse_docs = self._sparse_search(query, metadata_filter)
            if dense_docs:
                ranked_lists.append(dense_docs)
            if sparse_docs:
                ranked_lists.append(sparse_docs)

        if not ranked_lists:
            return []

        fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
        return add_rank_scores(deduplicate(fused))[: self.top_k]

    def _dense_search(self, query: str, metadata_filter: dict | None) -> list[Document]:
        """Dense vector search via Pinecone.

        Pinecone supports native metadata filtering, so LangChain passes the
        filter *after* fetching top-K vectors.  When a group is small relative
        to the total corpus, most of the top-K hits belong to other groups and
        get discarded, leaving very few (or zero) results.

        When a ``metadata_filter`` is active, Pinecone applies the filter in
        much larger candidate pool (``fetch_k``), apply the filter ourselves,
        and fall back to a manual scan of the known ``self.documents`` list if
        the managed vector index.
        """
        fetch_k = self.candidate_k

        if self.score_threshold > 0.0 and hasattr(
            self.vector_store, "similarity_search_with_relevance_scores"
        ):
            pairs = self.vector_store.similarity_search_with_relevance_scores(
                query,
                k=fetch_k,
                filter=metadata_filter,
            )
            docs: list[Document] = []
            for doc, score in pairs:
                if score >= self.score_threshold:
                    docs.append(
                        Document(
                            page_content=doc.page_content,
                            metadata={**doc.metadata, "relevance_score": float(score)},
                        )
                    )
            return docs[: self.candidate_k]

        results = self.vector_store.similarity_search(
            query,
            k=fetch_k,
            filter=metadata_filter,
        )

        return [doc for doc in results if not is_low_value_document(doc)][: self.candidate_k]

    def _sparse_search(self, query: str, metadata_filter: dict | None) -> list[Document]:
        """BM25 keyword search with optional pre-filtering by metadata.

        When a ``metadata_filter`` is provided we first narrow the corpus to
        only matching documents, then run BM25 scoring on that subset.  This
        ensures that all ``candidate_k`` slots are filled from the target
        group rather than being dominated by chunks from other groups that
        happen to rank higher in the global corpus.
        """
        tokens = self._tokenize(query)

        if metadata_filter:
            # Build a filtered index on-the-fly so BM25 scores are computed
            # only over documents that satisfy the filter.
            filtered_indices = [
                i for i, doc in enumerate(self.documents)
                if self._matches_filter(doc, metadata_filter)
            ]
            if not filtered_indices:
                return []

            sub_docs = [self.documents[i] for i in filtered_indices]
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi([self._tokenize(d.page_content) for d in sub_docs])
            scores = bm25.get_scores(tokens)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        else:
            sub_docs = self.documents
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        docs: list[Document] = []
        for idx in ranked:
            score = float(scores[idx])
            if score <= 0:
                break  # BM25 scores are monotonically non-increasing at this point
            doc = sub_docs[idx]
            if is_low_value_document(doc):
                continue
            docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "bm25_score": score},
                )
            )
            if len(docs) >= self.candidate_k:
                break
        return docs

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return (text or "").casefold().split()

    @staticmethod
    def _matches_filter(doc: Document, metadata_filter: dict) -> bool:
        metadata = doc.metadata or {}
        for key, expected in metadata_filter.items():
            value = metadata.get(key)
            if isinstance(expected, (list, tuple, set)):
                if value not in expected:
                    return False
            elif value != expected:
                return False
        return True
