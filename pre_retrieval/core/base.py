"""
Base class for pre-retrieval query transformers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pre_retrieval.core.types import TransformResult


class BaseTransformer(ABC):
    def __init__(self, language: str = "both"):
        self.language = language

    @abstractmethod
    def transform(self, query: str, **kwargs) -> TransformResult:
        """Transform a raw user query."""
