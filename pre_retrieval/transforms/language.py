"""
Lightweight Vietnamese/English language detection for retrieval queries.
"""
from __future__ import annotations

import re
import unicodedata

from pre_retrieval.core.base import BaseTransformer
from pre_retrieval.core.types import TransformResult


class LanguageDetector(BaseTransformer):
    _VI_DIACRITIC_RE = re.compile(
        r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩị"
        r"óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
        re.IGNORECASE,
    )
    _VI_WORDS = {
        "benh", "cay", "canh", "ca", "cach", "dat", "dau", "dau hieu", "giong",
        "lua", "mua", "nong", "nong nghiep", "phan", "phong", "sau", "sau benh",
        "thuoc", "trong", "tuoi", "vu", "xu ly", "la", "gi", "nhu", "the", "nao",
        "khi", "nao", "tai", "sao", "vi", "sao", "co", "khong",
    }
    _EN_WORDS = {
        "what", "why", "how", "when", "where", "who", "crop", "soil", "fertilizer",
        "disease", "pest", "irrigation", "agriculture", "plant", "rice",
    }

    def transform(self, query: str, **kwargs) -> TransformResult:
        language = self.detect(query)
        clean = (query or "").strip()
        return TransformResult(
            original_query=query,
            queries=[clean] if clean else [],
            language=language,
            metadata={"language_detection": "heuristic_vi_en_agriculture"},
        )

    def detect(self, query: str) -> str:
        text = (query or "").strip()
        if not text:
            return "unknown"

        lowered = text.casefold()
        ascii_text = self._strip_accents(lowered)
        tokens = set(re.findall(r"[a-z]+", ascii_text))
        has_vi_marks = bool(self._VI_DIACRITIC_RE.search(text))
        vi_score = len(tokens & self._VI_WORDS) + (2 if has_vi_marks else 0)
        en_score = len(tokens & self._EN_WORDS)

        if vi_score and en_score:
            return "both"
        if vi_score:
            return "vi"
        if en_score:
            return "en"
        if re.fullmatch(r"[\x00-\x7f]+", text):
            return "unknown"
        return "unknown"

    @staticmethod
    def _strip_accents(text: str) -> str:
        decomposed = unicodedata.normalize("NFD", text)
        stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        return stripped.replace("đ", "d").replace("Đ", "D")
