"""
Rule-based query cleaner for lightweight agriculture RAG.
"""
from __future__ import annotations

import re
import unicodedata

from pre_retrieval.core.base import BaseTransformer
from pre_retrieval.core.types import TransformResult


class QueryCleaner(BaseTransformer):
    _SPACE_RE = re.compile(r"\s+")
    _REPEATED_PUNCT_RE = re.compile(r"([!?.,;:])\1{2,}")
    _REPEATED_LETTER_RE = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)
    _LEADING_FILLER_RE = re.compile(
        r"^\s*(?:(?:dạa*|ạa*|vâng|xin chào|chào|alo|hello|hi|"
        r"ad\s+ơi|admin\s+ơi|bạn\s+ơi|anh\s+chị\s+ơi|"
        r"cho\s+(?:em|tôi|mình)\s+hỏi|(?:em|tôi|mình)\s+hỏi|xin\s+hỏi)"
        r"[\s,.;:!?]*)+",
        re.IGNORECASE | re.UNICODE,
    )
    _TRAILING_POLITE_RE = re.compile(
        r"[\s,.;:!?]*(?:ạ+|nhé|nha|với|giúp\s+(?:em|tôi|mình))\s*$",
        re.IGNORECASE | re.UNICODE,
    )
    _TOKEN_RE = re.compile(r"(?<!\w)[A-Za-zÀ-ỹĐđ]+(?!\w)", re.UNICODE)
    _TOKEN_NORMALIZATIONS = {
        "ko": "không",
        "k": "không",
        "khong": "không",
        "hok": "không",
        "hong": "không",
        "hông": "không",
        "kg": "không",
        "dc": "được",
        "đc": "được",
        "duoc": "được",
        "vs": "với",
        "voi": "với",
        "j": "gì",
        "gi": "gì",
        "benh": "bệnh",
        "bi": "bị",
        "lua": "lúa",
        "cay": "cây",
        "phan": "phân",
        "sau": "sâu",
        "thuoc": "thuốc",
        "dao": "đạo",
        "on": "ôn",
        "vang": "vàng",
    }

    def __init__(self, lowercase: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.lowercase = lowercase

    def transform(self, query: str, **kwargs) -> TransformResult:
        raw_query = query or ""
        normalized = unicodedata.normalize("NFC", raw_query)
        normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
        normalized = self._SPACE_RE.sub(" ", normalized).strip()
        normalized = self._REPEATED_PUNCT_RE.sub(r"\1", normalized)
        normalized = self._normalize_user_text(normalized)

        casefold_query = normalized.casefold()
        clean_query = casefold_query if self.lowercase else normalized
        changed = clean_query != raw_query.strip()

        return TransformResult(
            original_query=raw_query,
            queries=[clean_query] if clean_query else [],
            clean_query=clean_query,
            metadata={
                "raw_query": raw_query,
                "casefold_query": casefold_query,
                "unicode_normalization": "NFC",
                "lowercase_applied": self.lowercase,
                "user_text_normalization": "rule_based_v2",
                "user_text_normalized": changed,
            },
        )

    def _normalize_user_text(self, query: str) -> str:
        if not query:
            return ""

        normalized = query
        if not self._is_single_character_spam(normalized):
            normalized = self._REPEATED_LETTER_RE.sub(r"\1\1", normalized)

        normalized = self._LEADING_FILLER_RE.sub("", normalized)
        normalized = self._TRAILING_POLITE_RE.sub("", normalized)
        normalized = self._normalize_tokens(normalized)
        normalized = self._SPACE_RE.sub(" ", normalized).strip(" ,.;:")
        return normalized

    def _normalize_tokens(self, query: str) -> str:
        def replace(match: re.Match) -> str:
            token = match.group(0)
            replacement = self._TOKEN_NORMALIZATIONS.get(token.casefold())
            if replacement is None:
                return token
            if token == "K":
                return token
            return replacement

        return self._TOKEN_RE.sub(replace, query)

    @staticmethod
    def _is_single_character_spam(query: str) -> bool:
        compact = "".join(ch for ch in query.strip() if not ch.isspace())
        return len(compact) >= 8 and len(set(compact.casefold())) == 1
