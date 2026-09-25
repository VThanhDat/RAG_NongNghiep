from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from loader.core.base import Language
from loader.core.errors import FailedFileRecorder
from loader.orchestrators.pdf_document_loader import PDFDocumentLoader
from loader.core.types import ErrorMode, OCRMode, PDFStrategy, SUPPORTED_EXTENSIONS
from loader.utils.hashing import content_hash

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Compatibility wrapper for the simple PDF-only loader.

    Returns LangChain Document objects. To export to the JSON-like shape:
    {"text": "...", "metadata": {...}}, use `loader.core.records.to_text_records`.
    """

    def __init__(
        self,
        language: Language = "both",
        pdf_strategy: PDFStrategy = "pypdf",
        extract_tables: bool = True,
        deduplicate: bool = True,
        ocr: OCRMode = "never",
        remove_boilerplate: bool = True,
        failed_log_path: str | None = None,
        on_error: ErrorMode = "skip",
        max_file_size_mb: int | None = None,
    ):
        self.language = language
        self.deduplicate = deduplicate
        self.on_error = on_error
        self.max_file_size_mb = max_file_size_mb
        self.recorder = FailedFileRecorder(failed_log_path)
        self.failed_files = self.recorder.failed_files
        self._seen: set[str] = set()
        self._pdf_loader = PDFDocumentLoader(
            language=language,
            pdf_strategy=pdf_strategy,
            extract_tables=extract_tables,
            deduplicate=False,
            ocr=ocr,
            remove_boilerplate=remove_boilerplate,
            failed_log_path=str(self.recorder.failed_log_path),
            on_error=on_error,
            max_file_size_mb=max_file_size_mb,
        )

    def load(self, path: str) -> list[Document]:
        p = Path(path)
        if p.is_file():
            return self._load_file(p)
        if p.is_dir():
            return self._load_directory(p)
        raise ValueError(f"'{path}' is not a valid PDF file or directory.")

    def _load_directory(self, root: Path) -> list[Document]:
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        logger.info("DocumentLoader: found %d supported files in '%s'.", len(files), root)
        all_docs: list[Document] = []
        for file_path in files:
            all_docs.extend(self._load_file(file_path))
        return all_docs

    def _load_file(self, path: Path) -> list[Document]:
        try:
            docs = self._load_file_strict(path)
        except Exception as exc:
            self.recorder.record(path, exc)
            if self.on_error == "raise":
                raise
            logger.error("Skipping failed file: %s (%s: %s)", path, type(exc).__name__, exc)
            return []
        return self._dedup(docs) if self.deduplicate else docs

    def _load_file_strict(self, path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            logger.warning("Skip '%s': unsupported extension '%s'.", path.name, suffix)
            return []
        self._validate_size(path)
        if suffix == ".pdf":
            before = len(self._pdf_loader.failed_files)
            docs = self._pdf_loader._load_file(path)
            self.failed_files.extend(self._pdf_loader.failed_files[before:])
            return docs
        return []

    def _validate_size(self, path: Path) -> None:
        if self.max_file_size_mb is None:
            return
        max_bytes = self.max_file_size_mb * 1024 * 1024
        if path.stat().st_size > max_bytes:
            raise ValueError(f"File too large: {path} > {self.max_file_size_mb} MB")

    def _dedup(self, docs: list[Document]) -> list[Document]:
        unique: list[Document] = []
        for doc in docs:
            h = content_hash(doc.page_content)
            if h not in self._seen:
                self._seen.add(h)
                unique.append(doc)
        return unique
