from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class RAGState:
    lock: Lock = field(default_factory=Lock)
    cfg: dict[str, Any] | None = None
    source_path: str | None = None
    indexing_result: Any | None = None
    vdb_result: dict[str, Any] | None = None


rag_state = RAGState()
