from __future__ import annotations

from typing import Literal

PDFStrategy = Literal["pypdf", "pdfplumber"]
OCRMode = Literal["never"]
ErrorMode = Literal["skip", "raise"]

SUPPORTED_EXTENSIONS = {".pdf"}
