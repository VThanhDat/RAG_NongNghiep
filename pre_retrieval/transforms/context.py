"""
Resolve short follow-up queries using recent chat history.
"""
from __future__ import annotations

import re
from typing import Any

from pre_retrieval.core.base import BaseTransformer
from pre_retrieval.core.types import TransformResult


class ContextResolver(BaseTransformer):
    _FOLLOW_UP_RE = re.compile(
        r"^(còn|tiếp tục|vậy|thì sao|ra sao|nó|bệnh đó|cây đó|loại đó)\b",
        re.IGNORECASE,
    )
    _PRONOUN_RE = re.compile(r"\b(nó|bệnh đó|cây đó|loại đó)\b", re.IGNORECASE)

    def transform(self, query: str, **kwargs) -> TransformResult:
        history = kwargs.get("chat_history") or []
        resolved = self.resolve(query, history)
        clean = (query or "").strip()
        extra = {}
        if resolved != clean:
            extra["context_resolved_query"] = resolved
        return TransformResult(
            original_query=query,
            queries=[resolved] if resolved else [],
            rewritten_query=resolved if resolved != clean else None,
            extra=extra,
        )

    def resolve(self, query: str, chat_history: list[Any]) -> str:
        clean = (query or "").strip()
        if not clean or not self._FOLLOW_UP_RE.search(clean):
            return clean

        topic = self._last_topic(chat_history)
        if not topic:
            return clean

        lowered = clean.casefold()
        if lowered.startswith(("còn", "tiếp tục", "vậy", "thì sao", "ra sao")):
            return f"{clean} về {topic}"
        return self._PRONOUN_RE.sub(topic, clean)

    def _last_topic(self, chat_history: list[Any]) -> str | None:
        for item in reversed(chat_history):
            text = self._history_text(item).strip()
            if not text:
                continue
            words = text.split()
            if not words:
                continue
            return " ".join(words[:12])
        return None

    @staticmethod
    def _history_text(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return str(item.get("content") or item.get("query") or item.get("text") or "")
        return str(item)
