"""
Smoke test for the Loader stage against the local ./data folder.

Run from project root:
    python loader/tests/test_loader_data.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loader import DocumentLoader
from loader.core.records import to_text_records


REQUIRED_METADATA = {
    "source",
    "file_name",
    "page_number",
    "title",
    "section",
    "created_at",
    "document_type",
}


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    loader = DocumentLoader(
        pdf_strategy="pypdf",
        ocr="never",  # Important: this test checks normal text-layer loading only.
        failed_log_path=str(PROJECT_ROOT / "loader_failed_files.jsonl"),
    )
    docs = loader.load(str(data_dir))

    if not docs:
        raise AssertionError("Loader returned no documents.")

    missing = sorted({
        key
        for doc in docs
        for key in REQUIRED_METADATA
        if key not in doc.metadata
    })
    if missing:
        raise AssertionError(f"Missing required metadata keys: {missing}")

    records = to_text_records(docs[:1])
    if not records or set(records[0]) != {"text", "metadata"}:
        raise AssertionError("to_text_records output is not {'text', 'metadata'}.")

    type_counts = Counter(doc.metadata.get("document_type") for doc in docs)
    scanned_count = sum(1 for doc in docs if doc.metadata.get("is_scanned_pdf"))

    print("Loader smoke test passed.")
    print(f"Data folder: {data_dir}")
    print(f"Documents loaded: {len(docs)}")
    print(f"Document types: {dict(type_counts)}")
    print(f"Failed files: {len(loader.failed_files)}")
    print(f"Scanned/OCR documents: {scanned_count}")
    print(f"First document metadata: {docs[0].metadata}")
    print(f"First text preview: {docs[0].page_content[:200].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
