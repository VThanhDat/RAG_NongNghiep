"""
Registry for the simplified pre-retrieval transformers.
"""
from __future__ import annotations

from typing import Type

from pre_retrieval.core.base import BaseTransformer


def get_transformer_class(name: str) -> Type[BaseTransformer]:
    if name == "none":
        from pre_retrieval.core.passthrough import PassthroughTransformer

        return PassthroughTransformer
    if name == "clean":
        from pre_retrieval.transforms.clean import QueryCleaner

        return QueryCleaner
    if name == "validate":
        from pre_retrieval.transforms.validate import QueryValidator

        return QueryValidator
    if name == "context":
        from pre_retrieval.transforms.context import ContextResolver

        return ContextResolver
    if name == "language":
        from pre_retrieval.transforms.language import LanguageDetector

        return LanguageDetector
    if name == "intent":
        from pre_retrieval.transforms.intent import IntentClassifier

        return IntentClassifier

    raise ValueError(
        f"Unknown pre-retrieval transformation: {name!r}. "
        "Supported transformations: none, clean, validate, context, language, intent."
    )
