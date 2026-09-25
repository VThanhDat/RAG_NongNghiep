"""
Configuration for the simplified pre-retrieval stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_TRANSFORMATIONS = ["clean", "validate", "context", "language", "intent"]


@dataclass
class PreRetrievalConfig:
    transformations: list[str] = field(default_factory=lambda: list(DEFAULT_TRANSFORMATIONS))
    language: str = "both"
    clean_lowercase: bool = False
    max_query_chars: int = 500
    max_repeated_char: int = 8

    @classmethod
    def from_dict(
        cls,
        pre_cfg: dict | None,
        gen_cfg: dict | None = None,
        global_cfg: dict | None = None,
    ) -> "PreRetrievalConfig":
        pre_cfg = pre_cfg or {}
        global_cfg = global_cfg or {}
        return cls(
            transformations=pre_cfg.get("transformations", list(DEFAULT_TRANSFORMATIONS)),
            language=global_cfg.get("language", pre_cfg.get("language", "both")),
            clean_lowercase=pre_cfg.get("clean_lowercase", False),
            max_query_chars=pre_cfg.get("max_query_chars", 500),
            max_repeated_char=pre_cfg.get("max_repeated_char", 8),
        )
