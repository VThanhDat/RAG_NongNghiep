"""
Rule-based intent classification for agriculture RAG queries.
"""
from __future__ import annotations

import re
import unicodedata

from pre_retrieval.core.base import BaseTransformer
from pre_retrieval.core.types import TransformResult


class IntentClassifier(BaseTransformer):
    _CHAT_RE = re.compile(r"^(xin chào|chào|hello|hi|cảm ơn|thanks|thank you)\b", re.IGNORECASE)
    _QUESTION_RE = re.compile(
        r"(\?| là gì\b| thế nào\b| như thế nào\b| ra sao\b| vì sao\b| tại sao\b| khi nào\b|"
        r" bao nhiêu\b| cách\b| làm sao\b|\bwhat\b|\bwhy\b|\bhow\b|\bwhen\b|\bwhere\b|\bwho\b)",
        re.IGNORECASE,
    )
    _COMMAND_RE = re.compile(
        r"\b(hãy|vui lòng|cho tôi|liệt kê|tóm tắt|so sánh|tìm|xem|phân tích|"
        r"summarize|compare|list|find|show|explain)\b",
        re.IGNORECASE,
    )
    _AGRI_TERMS = {
        "nong nghiep", "cay", "cay trong", "lua", "ngo", "ca phe", "sau rieng",
        "rau", "dat", "phan bon", "thuoc bao ve thuc vat", "sau", "benh",
        "sau benh", "nam", "vi khuan", "co dai", "tuoi", "nuoc", "mua vu",
        "canh tac", "giong", "nang suat", "thu hoach", "sau cuon la",
        "than thu", "kali", "dam", "lan",
    }

    def transform(self, query: str, **kwargs) -> TransformResult:
        clean = (query or "").strip()
        intent = self.classify(clean)
        return TransformResult(
            original_query=query,
            queries=[clean] if clean else [],
            intent=intent,
            metadata={"intent_classifier": "rule_based_agriculture"},
        )

    def classify(self, query: str) -> str:
        if not query:
            return "empty"
        if self._CHAT_RE.search(query):
            return "chat"

        is_agriculture = self._is_agriculture_query(query)
        if self._COMMAND_RE.search(query):
            return "agriculture_command" if is_agriculture else "command"
        if self._QUESTION_RE.search(query):
            return "agriculture_question" if is_agriculture else "question"
        if is_agriculture:
            return "agriculture_lookup"
        return "out_of_domain"

    def _is_agriculture_query(self, query: str) -> bool:
        normalized = self._strip_accents(query.casefold())
        return any(term in normalized for term in self._AGRI_TERMS)

    @staticmethod
    def _strip_accents(text: str) -> str:
        decomposed = unicodedata.normalize("NFD", text)
        stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        return stripped.replace("đ", "d").replace("Đ", "D")
