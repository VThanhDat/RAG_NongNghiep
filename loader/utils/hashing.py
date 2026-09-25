"""
loader/utils/hashing.py
=======================
Content hashing utilities for deduplication.
"""

from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    """MD5 fingerprint của chuỗi text — dùng cho deduplication chính xác."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
