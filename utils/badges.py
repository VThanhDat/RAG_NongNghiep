"""
Simple HTML badges for the current PDF-only agriculture RAG pipeline.
"""
from __future__ import annotations


def file_type_badge(file_type: str) -> str:
    color = "#e74c3c" if file_type.lower() == "pdf" else "#95a5a6"
    return (
        f'<span style="background:{color}; color:white; padding:2px 8px; '
        f'border-radius:8px; font-size:0.75rem; font-weight:bold;">'
        f'{file_type.upper()}</span>'
    )


def chunk_type_badge(index: int, level: str = "") -> str:
    label = f"Chunk {index + 1}" + (f" [{level}]" if level else "")
    return (
        '<span style="background:#2c7a7b; color:white; padding:2px 8px; '
        f'border-radius:8px; font-size:0.75rem; font-weight:bold;">{label}</span>'
    )


def friendly_import_error(exc: ModuleNotFoundError) -> str:
    missing = exc.name or str(exc)
    hints = {
        "pypdf": "pip install -r requirements.txt",
        "pdfplumber": "pip install -r requirements.txt",
        "rank_bm25": "pip install -r requirements.txt",
        "cohere": "pip install -r requirements.txt",
    }
    hint = hints.get(missing)
    if hint:
        return f"Missing package `{missing}`. Install with: `{hint}`"
    return f"Missing module `{missing}`."
