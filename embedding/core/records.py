"""
Record helpers for embedding output.
"""

from __future__ import annotations

import hashlib


def stable_text_id(text: str, *, prefix: str = "chunk") -> str:
    """Build a deterministic id when chunk metadata does not already have one."""
    digest = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{prefix}:{digest}"
