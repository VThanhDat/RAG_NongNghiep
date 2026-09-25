"""
Persistent cache for dense embedding vectors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class EmbeddingCache:
    """Small JSON-backed cache keyed by model settings and normalized text."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.available = True
        self._data: dict[str, list[float]] = {}
        self._load()

    def get(self, key: str) -> list[float] | None:
        vector = self._data.get(key)
        if vector is None:
            return None
        return [float(value) for value in vector]

    def set(self, key: str, vector: list[float]) -> None:
        if not self.available:
            return
        self._data[key] = [float(value) for value in vector]
        self._save()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.available = False
            return
        if isinstance(raw, dict):
            self._data = {
                str(key): [float(value) for value in vector]
                for key, vector in raw.items()
                if isinstance(vector, list)
            }

    def _save(self) -> None:
        try:
            self.path.write_text(
                json.dumps(self._data, separators=(",", ":")),
                encoding="utf-8",
            )
        except OSError:
            self.available = False


def make_embedding_cache_key(
    *,
    provider: str,
    model: str,
    expected_dimension: int | None,
    normalize_dense: bool,
    text: str,
) -> str:
    """Build a stable cache key for model settings plus normalized text."""
    payload = "\n".join([
        provider,
        model,
        str(expected_dimension or ""),
        str(bool(normalize_dense)),
        text,
    ])
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()
