"""
Text preparation helpers for embedding.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_embedding_text(text: str) -> str:
    """
    Normalize text before embedding.

    Keep paragraph boundaries, but remove control/null bytes, normalize Unicode
    accents, trim whitespace, and collapse excessive blank lines.
    """
    text = unicodedata.normalize("NFC", str(text or ""))
    text = text.replace("\x00", "")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]

    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 1:
                cleaned.append("")
            continue
        blank_run = 0
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def is_embedding_text_valid(text: str, *, min_chars: int = 20) -> bool:
    """Return True when a chunk has enough signal to embed."""
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < min_chars:
        return False

    alnum = sum(1 for char in compact if char.isalnum())
    return alnum >= max(8, min_chars // 2)


def embedding_text_quality_flags(text: str) -> list[str]:
    """Return non-fatal quality flags for noisy text."""
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return ["empty"]

    flags: list[str] = []
    symbol_count = sum(1 for char in compact if not char.isalnum())
    if symbol_count / max(1, len(compact)) > 0.35:
        flags.append("high_symbol_ratio")
    if re.search(r"(.)\1{8,}", compact):
        flags.append("repeated_character_run")
    if len(set(compact.lower())) <= 3 and len(compact) >= 20:
        flags.append("low_character_diversity")
    return flags
