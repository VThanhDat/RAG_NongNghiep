"""
Validate user queries before retrieval.
"""
from __future__ import annotations

import re

from pre_retrieval.core.base import BaseTransformer
from pre_retrieval.core.types import TransformResult


class QueryValidator(BaseTransformer):
    _INJECTION_PATTERNS = (
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"system\s*prompt",
        r"developer\s*message",
        r"jailbreak",
        r"act\s+as\s+(dan|developer|system|admin)",
        r"bỏ\s+qua\s+(tất\s+cả\s+)?(hướng\s+dẫn|chỉ\s+thị)\s+(trước|trước\s+đó)",
        r"hãy\s+đóng\s+vai\s+(system|developer|admin)",
        r"tiết\s+lộ\s+(system\s+prompt|prompt\s+hệ\s+thống)",
    )

    def __init__(self, max_chars: int = 500, max_repeated_char: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.max_chars = max_chars
        self.max_repeated_char = max_repeated_char

    def transform(self, query: str, **kwargs) -> TransformResult:
        clean = (query or "").strip()
        reason = self._rejection_reason(clean)
        is_valid = reason is None
        return TransformResult(
            original_query=query,
            queries=[clean] if clean and is_valid else [],
            clean_query=clean,
            intent="prompt_injection" if reason == "prompt_injection" else None,
            is_valid=is_valid,
            rejection_reason=reason,
            warnings=[] if is_valid else [f"Rejected query: {reason}"],
            metadata={"validation": {"max_chars": self.max_chars}},
        )

    def _rejection_reason(self, query: str) -> str | None:
        if not query:
            return "empty_query"
        if len(query) > self.max_chars:
            return "query_too_long"
        if re.search(r"(.)\1{" + str(self.max_repeated_char) + r",}", query):
            return "spam_repeated_characters"
        lowered = query.casefold()
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in self._INJECTION_PATTERNS):
            return "prompt_injection"
        return None
