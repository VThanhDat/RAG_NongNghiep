from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FailedFileRecorder:
    def __init__(self, failed_log_path: str | None = None):
        self.failed_log_path = Path(failed_log_path or "loader_failed_files.jsonl")
        self.failed_files: list[dict] = []

    def record(self, path: Path, exc: Exception) -> None:
        item = {
            "source": str(path),
            "file_name": path.name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.failed_files.append(item)
        try:
            with self.failed_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Could not write failed file log: %s", self.failed_log_path)
