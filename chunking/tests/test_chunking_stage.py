"""
Smoke tests for the simplified Chunking stage.

Run from project root:
    python chunking/tests/test_chunking_stage.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from chunking import chunk_documents_from_config, deduplicate_chunks, get_chunker, to_text_records
from chunking.core.metadata import enrich_chunk_metadata
from chunking.postprocess.merge import merge_tiny_chunks


REQUIRED_CHUNK_METADATA = {
    "source",
    "file_name",
    "page",
    "page_number",
    "title",
    "section",
    "chunk_id",
    "chunk_index",
    "char_count",
    "token_count",
    "token_count_method",
}


def _doc(text: str, *, source: str = "sample.pdf") -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "file_name": Path(source).name,
            "page_number": 1,
            "title": "Sample",
            "section": None,
            "file_type": "pdf",
            "document_type": "pdf",
        },
    )


def test_only_recursive_strategy_is_supported() -> None:
    get_chunker("recursive")
    try:
        get_chunker("unsupported")
    except ValueError as exc:
        if "Valid: recursive" not in str(exc):
            raise
        return
    raise AssertionError("Only recursive strategy should be supported.")


def test_recursive_metadata() -> None:
    text = (
        "Benh than thu tren sau rieng thuong phat sinh khi am do cao. "
        "Can tia canh tao thong thoang va theo doi vet benh tren la.\n\n"
        "Neu benh xuat hien som, can thu gom la benh va xu ly theo khuyen cao."
    ) * 4
    chunker = get_chunker("recursive", chunk_size=180, chunk_overlap=30)
    chunks = chunker.split([
        _doc(text, source="a.pdf"),
        _doc(text, source="b.pdf"),
    ])

    if not chunks:
        raise AssertionError("RecursiveChunker returned no chunks.")

    missing = sorted({
        key
        for chunk in chunks
        for key in REQUIRED_CHUNK_METADATA
        if key not in chunk.metadata
    })
    if missing:
        raise AssertionError(f"Missing required chunk metadata: {missing}")

    first_by_source: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        first_by_source.setdefault(source, chunk.metadata["chunk_index"])
    if first_by_source != {"a.pdf": 0, "b.pdf": 0}:
        raise AssertionError(f"chunk_index should reset per source: {first_by_source}")
    if any(chunk.metadata["token_count"] < 1 for chunk in chunks):
        raise AssertionError("token_count must be present and positive.")

    records = to_text_records(chunks[:1])
    if not records or set(records[0]) != {"text", "metadata"}:
        raise AssertionError("Chunk records must be {'text', 'metadata'}.")


def test_deduplicate_exact() -> None:
    chunks = [
        _doc("Noi dung trung lap.", source="dup.pdf"),
        _doc("Noi dung trung lap.", source="dup.pdf"),
        _doc("Noi dung khac.", source="dup.pdf"),
    ]
    unique = deduplicate_chunks(chunks, method="exact")
    if len(unique) != 2:
        raise AssertionError(f"Expected 2 unique chunks, got {len(unique)}.")
    try:
        deduplicate_chunks(chunks, method="unsupported")
    except ValueError:
        return
    raise AssertionError("Only exact deduplication should be supported.")


def test_merge_tiny_chunks() -> None:
    chunks = enrich_chunk_metadata([
        _doc("1. Phong tru benh"),
        _doc("Can cat tia canh benh, ve sinh vuon va theo doi do am."),
    ])
    chunks[0].metadata["section"] = "1. Phong tru benh"
    chunks[1].metadata["section"] = "1. Phong tru benh"
    merged = merge_tiny_chunks(chunks, min_tokens=80, max_tokens=1000)
    if len(merged) >= len(chunks):
        raise AssertionError("Tiny chunks should be merged with nearby content.")
    if not any(chunk.metadata.get("merged_tiny_chunks") for chunk in merged):
        raise AssertionError("Merged chunks should record merged_tiny_chunks metadata.")


def test_config_recursive_pipeline() -> None:
    text = "Sau rieng can thoat nuoc tot trong mua mua. " * 50
    cfg = {
        "indexing": {
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 220,
                "chunk_overlap": 40,
                "token_encoding": "cl100k_base",
                "merge_tiny_chunks": True,
                "min_chunk_tokens": 20,
                "max_chunk_tokens": 1000,
                "deduplicate_chunks": True,
                "deduplicate_method": "exact",
            }
        }
    }
    chunks = chunk_documents_from_config([_doc(text, source="config.pdf")], cfg)
    if not chunks:
        raise AssertionError("Config chunking returned no chunks.")
    if any(chunk.metadata.get("chunk_id") is None for chunk in chunks):
        raise AssertionError("Config chunking must enrich chunk metadata.")


def main() -> None:
    test_only_recursive_strategy_is_supported()
    test_recursive_metadata()
    test_deduplicate_exact()
    test_merge_tiny_chunks()
    test_config_recursive_pipeline()
    print("Chunking smoke test passed.")


if __name__ == "__main__":
    main()
