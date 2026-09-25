"""
Simple pre-retrieval pipeline for agriculture RAG queries.
"""
from __future__ import annotations

import logging

from pre_retrieval.core.base import BaseTransformer
from pre_retrieval.core.config import PreRetrievalConfig
from pre_retrieval.core.registry import get_transformer_class
from pre_retrieval.core.types import TransformResult

logger = logging.getLogger(__name__)


class PreRetrievalPipeline:
    """Run deterministic query preparation before retrieval."""

    def __init__(self, config: PreRetrievalConfig):
        self.config = config
        self._transformers: list[BaseTransformer] = [
            self._build(name) for name in (config.transformations or ["none"])
        ]

    def transform(self, query: str, chat_history: list | None = None) -> TransformResult:
        merged = TransformResult(original_query=query, queries=[])
        current_query = query

        for transformer in self._transformers:
            result = transformer.transform(
                current_query,
                chat_history=chat_history or [],
                language=merged.language,
            )

            for candidate in result.queries:
                if candidate and self._query_key(candidate) not in {
                    self._query_key(existing) for existing in merged.queries
                }:
                    merged.queries.append(candidate)

            if result.clean_query is not None:
                merged.clean_query = result.clean_query
            if result.rewritten_query is not None:
                merged.rewritten_query = result.rewritten_query
            if result.language is not None:
                merged.language = result.language
            if result.intent is not None:
                merged.intent = result.intent
            if result.metadata_filter is not None:
                merged.metadata_filter = result.metadata_filter
            if result.retrieval_path is not None:
                merged.retrieval_path = result.retrieval_path

            merged.metadata.update(result.metadata)
            merged.extra.update(result.extra)
            merged.expanded_terms = self._dedupe(merged.expanded_terms + result.expanded_terms)
            merged.warnings.extend(result.warnings)

            if not result.is_valid:
                merged.is_valid = False
                merged.rejection_reason = result.rejection_reason
                merged.queries = []
                break

            replacement_query = result.rewritten_query or result.clean_query
            if replacement_query and replacement_query != current_query:
                current_query = replacement_query

        if merged.is_valid and current_query and self._query_key(current_query) not in {
            self._query_key(existing) for existing in merged.queries
        }:
            merged.queries.insert(0, current_query)

        logger.info(
            "PreRetrievalPipeline: %r -> %d queries, language=%s, intent=%s",
            query[:60],
            len(merged.queries),
            merged.language,
            merged.intent,
        )
        return merged

    def _build(self, name: str) -> BaseTransformer:
        transformer_class = get_transformer_class(name)
        cfg = self.config
        base_args = {"language": cfg.language}
        if name == "clean":
            return transformer_class(lowercase=cfg.clean_lowercase, **base_args)
        if name == "validate":
            return transformer_class(
                max_chars=cfg.max_query_chars,
                max_repeated_char=cfg.max_repeated_char,
                **base_args,
            )
        return transformer_class(**base_args)

    @staticmethod
    def _query_key(query: str) -> str:
        return " ".join((query or "").split()).casefold()

    @classmethod
    def _dedupe(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = cls._query_key(value)
            if key and key not in seen:
                seen.add(key)
                result.append(value)
        return result
