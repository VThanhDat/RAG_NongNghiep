"""
loader/pdf/factory.py
=====================
Factory function for the simple PDF-only loader setup.

Only two text-layer strategies are supported:
  - pypdf: fast default for normal digital PDFs.
  - pdfplumber: slower, better when tables matter.

OCR is intentionally disabled for the current simple agriculture RAG scope.
"""

from __future__ import annotations

from loader.core.base import BaseLoader, Language
from loader.core.types import OCRMode, PDFStrategy


def build_pdf_loader(
    language: Language,
    pdf_strategy: PDFStrategy,
    extract_tables: bool,
    ocr: OCRMode = "never",
) -> BaseLoader:
    if ocr != "never":
        raise ValueError("OCR is disabled in the simple loader. Use ocr='never'.")

    if pdf_strategy == "pypdf":
        from loader.pdf.strategies.pypdf import PyPDFLoader

        return PyPDFLoader(language)

    if pdf_strategy == "pdfplumber":
        from loader.pdf.strategies.pdfplumber import PDFPlumberLoader

        return PDFPlumberLoader(language, extract_tables=extract_tables)

    raise ValueError("Unsupported PDF strategy. Use 'pypdf' or 'pdfplumber'.")
