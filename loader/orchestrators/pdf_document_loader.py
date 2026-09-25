from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from loader.core.base import BaseLoader, Language
from loader.core.errors import FailedFileRecorder
from loader.core.types import ErrorMode, OCRMode, PDFStrategy
from loader.utils.hashing import content_hash
from loader.utils.boilerplate import strip_repeated_boilerplate
from loader.pdf.factory import build_pdf_loader

logger = logging.getLogger(__name__)


class PDFDocumentLoader:
    """
    Simple PDF-only loader for the agriculture RAG pipeline.

    Supported strategies:
      - "pypdf": fast text-layer extraction.
      - "pdfplumber": text-layer extraction with table support.

    OCR is intentionally disabled. Use ocr="never".
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
        self.pdf_strategy = pdf_strategy
        self.extract_tables = extract_tables
        self.ocr = ocr
        self.remove_boilerplate = remove_boilerplate
        self.on_error = on_error
        self.max_file_size_mb = max_file_size_mb
        self.deduplicate = deduplicate
        self.recorder = FailedFileRecorder(failed_log_path)
        self.failed_files = self.recorder.failed_files
        self._seen: set[str] = set()
        if self.ocr != "never":
            raise ValueError("OCR is disabled in the simple PDF loader. Use ocr='never'.")
        self._loader: BaseLoader = self._make_pdf_loader(ocr="never")

    def load(self, path: str) -> list[Document]:
        p = Path(path)
        if p.is_file():
            return self._load_file(p)
        if p.is_dir():
            return self._load_directory(p)
        raise ValueError(f"'{path}' is not a valid PDF file or directory.")

    def _make_pdf_loader(self, *, ocr: OCRMode | None = None) -> BaseLoader:
        return build_pdf_loader(
            language=self.language,
            pdf_strategy=self.pdf_strategy,
            extract_tables=self.extract_tables,
            ocr=ocr or self.ocr,
        )

    def _load_file(self, path: Path) -> list[Document]:
        try:
            return self._load_file_strict(path)
        except Exception as exc:
            self.recorder.record(path, exc)
            if self.on_error == "raise":
                raise
            logger.error("Skipping failed PDF: %s (%s: %s)", path, type(exc).__name__, exc)
            return []

    def _load_file_strict(self, path: Path) -> list[Document]:
        if path.suffix.lower() != ".pdf":
            logger.warning("Skip '%s': not a PDF file.", path.name)
            return []
        self._validate_size(path)

        loader = self._loader

        logger.info("Loading: %s", path.name)
        docs = loader.load(str(path))
        for doc in docs:
            doc.metadata.setdefault("is_scanned_pdf", False)
        if self.remove_boilerplate:
            docs = strip_repeated_boilerplate(docs)
        return self._dedup(docs) if self.deduplicate else docs

    def _load_directory(self, root: Path) -> list[Document]:
        pdf_files = sorted(root.rglob("*.pdf"))
        logger.info("PDFDocumentLoader: found %d PDF files in '%s'.", len(pdf_files), root)
        all_docs: list[Document] = []
        for pdf in pdf_files:
            all_docs.extend(self._load_file(pdf))
        logger.info("PDFDocumentLoader: total %d documents.", len(all_docs))
        return all_docs

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
