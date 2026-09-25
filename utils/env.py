"""
Small environment helpers for the simplified RAG Nong Nghiep pipeline.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

try:
    from dotenv import dotenv_values, load_dotenv

    load_dotenv(override=True)
    _ENV = dotenv_values()
except ImportError:
    _ENV = {}


SUPPORTED_PACKAGES = {
    "pypdf": "pypdf",
    "pdfplumber": "pdfplumber",
    "langchain_core": "langchain_core",
    "langchain_text_splitters": "langchain_text_splitters",
    "rank_bm25": "rank_bm25",
    "langchain_community": "langchain_community",
    "google_genai": "google.genai",
    "cohere": "cohere",
    "pinecone": "pinecone",
    "yaml": "yaml",
}


def is_installed(package: str) -> bool:
    import_name = SUPPORTED_PACKAGES.get(package, package)
    return importlib.util.find_spec(import_name) is not None


def get_env(key: str) -> str | None:
    value = _ENV.get(key) or os.environ.get(key)
    if not value:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in {"none", "null", "placeholder", "changeme"}:
        return None
    if stripped.lower().startswith("your_"):
        return None
    return stripped


def uploads_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "rag_nong_nghiep_uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_uploaded_files(uploaded_files) -> str:
    target_dir = uploads_dir()
    for old_file in target_dir.glob("*"):
        if old_file.is_file():
            old_file.unlink()

    for uploaded in uploaded_files:
        destination = target_dir / uploaded.name
        destination.write_bytes(uploaded.read())
    return str(target_dir)


def file_profile(source_path: str) -> dict:
    path = Path(source_path)
    files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    pdf_files = [item for item in files if item.suffix.lower() == ".pdf"]
    return {
        "total": len(files),
        "pdf": len(pdf_files),
        "unsupported": len(files) - len(pdf_files),
        "pdf_only": len(files) == len(pdf_files),
    }
