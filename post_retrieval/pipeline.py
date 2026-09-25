"""
Simple post-retrieval pipeline for agriculture RAG.

This stage keeps retrieval output small and clean before generation:
exact deduplication -> score filtering -> near-duplicate filtering -> ordering.
"""
from __future__ import annotations

import logging
import time

from langchain_core.documents import Document

from post_retrieval.core.types import PostRetrievalResult
from post_retrieval.core.utils import deduplicate
from post_retrieval.processors.context_orderer import ContextOrderer
from post_retrieval.processors.redundancy_filter import RedundancyFilter
from post_retrieval.processors.score_filter import ScoreFilter
from post_retrieval.processors.toc_filter import TocFilter

logger = logging.getLogger(__name__)


class PostRetrievalPipeline:
    def __init__(
        self,
        top_n: int = 5,
        score_threshold: float = 0.0,
        apply_redundancy: bool = True,
        redundancy_threshold: float = 0.9,
        context_ordering: str = "relevance",
        apply_toc_filter: bool = True,
        toc_dot_ratio: float = 0.15,
        toc_short_line_ratio: float = 0.70,
        toc_min_chars: int = 200,
        toc_early_page_limit: int = 3,
    ):
        self.top_n = top_n
        self.score_threshold = score_threshold
        self.apply_redundancy = apply_redundancy
        self.redundancy_threshold = redundancy_threshold
        self.context_ordering = context_ordering
        self.apply_toc_filter = apply_toc_filter

        self._score_filter = ScoreFilter(threshold=score_threshold, keep_min=min(1, top_n))
        self._toc_filter = TocFilter(
            dot_ratio=toc_dot_ratio,
            short_line_ratio=toc_short_line_ratio,
            min_chars=toc_min_chars,
            early_page_limit=toc_early_page_limit,
            keep_min=min(1, top_n),
        )
        self._redundancy_filter = RedundancyFilter(top_n=top_n, threshold=redundancy_threshold)
        self._orderer = ContextOrderer(ordering=context_ordering)

    def process(self, query: str, docs: list[Document]) -> PostRetrievalResult:
        t0 = time.perf_counter()
        input_count = len(docs)
        steps_run: list[str] = []

        current = deduplicate(docs)
        steps_run.append("Deduplicate")

        if current and self.score_threshold > 0:
            current = self._score_filter.process(query, current)
            steps_run.append("ScoreFilter")

        if current and self.apply_toc_filter:
            current = self._toc_filter.process(query, current)
            steps_run.append("TocFilter")

        if current and self.apply_redundancy:
            current = self._redundancy_filter.process(query, current)
            steps_run.append("RedundancyFilter")

        if current:
            current = self._orderer.process(query, current)
            steps_run.append("ContextOrderer")

        result_docs = current[: self.top_n]
        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "PostRetrievalPipeline: %d -> %d docs (%.1f ms)",
            input_count,
            len(result_docs),
            latency_ms,
        )
        return PostRetrievalResult(
            query=query,
            documents=result_docs,
            steps_run=steps_run,
            input_count=input_count,
            output_count=len(result_docs),
            latency_ms=latency_ms,
        )
