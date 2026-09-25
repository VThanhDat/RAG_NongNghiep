"""
Smoke tests for the Embedding stage.

These tests avoid downloading real embedding models by replacing the dense
embedder with a tiny deterministic fake.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedding import EmbeddingPipeline, get_embedder_from_config


class FakeDenseEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return [float(len(query)), 1.0, 0.0]


def test_config_defaults_to_cohere_embedding() -> None:
    pipeline = get_embedder_from_config({"indexing": {"embedding": {}}})
    if pipeline.dense_provider != "cohere":
        raise AssertionError("Default provider must be Cohere.")
    if pipeline.dense_model != "embed-v4.0":
        raise AssertionError("Default model must be embed-v4.0.")
    if pipeline.expected_dimension != 1024:
        raise AssertionError("Cohere embedding default dimension must be 1024.")


def test_unknown_providers_are_rejected() -> None:
    try:
        get_embedder_from_config({
            "indexing": {
                "embedding": {
                    "provider": "unsupported_provider",
                    "model_name": "unsupported-model",
                }
            }
        })
    except ValueError as exc:
        if "Unsupported embedding provider" not in str(exc):
            raise
        return
    raise AssertionError("Paid API embedding providers should be rejected.")


def test_embedding_records_output_format() -> None:
    pipeline = EmbeddingPipeline(
        dense_provider="cohere",
        dense_model="embed-v4.0",
        expected_dimension=3,
        enable_sparse=True,
        min_text_chars=10,
    )
    fake = FakeDenseEmbedder()
    pipeline._dense = fake

    records = [
        {
            "text": "  Day la chunk tieng Viet ve phong tru benh than thu tren sau rieng.  ",
            "metadata": {
                "source": "book.pdf",
                "file_name": "book.pdf",
                "page_number": 1,
                "title": "Book",
                "section": "Chuong 1",
                "chunk_id": "chunk-001",
                "chunk_index": 0,
            },
        },
        {"text": "ngan", "metadata": {"chunk_id": "too-short"}},
    ]

    output = pipeline.embed_records(records)
    if len(output) != 1:
        raise AssertionError(f"Expected one valid embedded record, got {len(output)}.")

    item = output[0]
    if set(item) != {"id", "embedding", "text", "metadata", "sparse_embedding"}:
        raise AssertionError(f"Unexpected output keys: {set(item)}")
    if item["id"] != "chunk-001":
        raise AssertionError("Output id must come from metadata.chunk_id.")
    if len(item["embedding"]) != 3:
        raise AssertionError("Dense vector dimension mismatch.")
    if item["metadata"]["embedding_model"] != "embed-v4.0":
        raise AssertionError("Embedding metadata must record the dense model.")
    if item["metadata"]["embedding_metric"] != "cosine":
        raise AssertionError("Embedding metadata must record the vector metric.")
    if "sparse_embedding" not in item or not isinstance(item["sparse_embedding"], dict):
        raise AssertionError("Sparse BM25 vector field should be present when enabled.")

    pipeline.embed_records(records[:1])
    if fake.calls != 1:
        raise AssertionError("Embedding cache should avoid recomputing identical text.")


def test_embedding_records_handle_table_code_long_and_noise() -> None:
    pipeline = EmbeddingPipeline(
        dense_provider="cohere",
        dense_model="embed-v4.0",
        expected_dimension=3,
        enable_sparse=False,
        min_text_chars=10,
    )
    pipeline._dense = FakeDenseEmbedder()

    records = [
        {
            "text": "| Cay trong | Benh |\n| --- | --- |\n| Sau rieng | Than thu |",
            "metadata": _metadata("table-001", section="Bang du lieu"),
        },
        {
            "text": "def search(query):\n    return vector_store.similarity_search(query, k=5)",
            "metadata": _metadata("code-001", section="Code"),
        },
        {
            "text": "Noi dung dai ve ky thuat canh tac va phong tru sau benh. " * 80,
            "metadata": _metadata("long-001", section="Doan dai"),
        },
        {
            "text": "!!!!!! ||||||| ////// ?????? ######",
            "metadata": _metadata("noise-001", section="OCR"),
        },
    ]

    output = pipeline.embed_records(records)
    ids = {item["id"] for item in output}
    if ids != {"table-001", "code-001", "long-001"}:
        raise AssertionError(f"Unexpected embedded ids: {ids}")


def test_strict_metadata_rejects_missing_source_fields() -> None:
    pipeline = EmbeddingPipeline(
        dense_provider="cohere",
        dense_model="embed-v4.0",
        expected_dimension=3,
        strict_metadata=True,
        min_text_chars=10,
    )
    pipeline._dense = FakeDenseEmbedder()

    try:
        pipeline.embed_records([
            {"text": "Day la chunk du dai nhung thieu metadata nguon.", "metadata": {}}
        ])
    except ValueError as exc:
        if "metadata missing" not in str(exc):
            raise
        return

    raise AssertionError("Expected strict metadata validation to raise ValueError.")


def test_persistent_cache_reuses_vectors_across_pipeline_instances() -> None:
    cache_path = ROOT / ".cache" / "test_embedding_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.unlink(missing_ok=True)
    first = EmbeddingPipeline(
        dense_provider="cohere",
        dense_model="embed-v4.0",
        expected_dimension=3,
        cache_path=str(cache_path),
        min_text_chars=10,
    )
    first_fake = FakeDenseEmbedder()
    first._dense = first_fake
    first.embed_records([
        {"text": "Day la chunk cache persistent.", "metadata": _metadata("cache-001")}
    ])
    if first_fake.calls != 1:
        raise AssertionError("First pipeline should compute the vector once.")

    second = EmbeddingPipeline(
        dense_provider="cohere",
        dense_model="embed-v4.0",
        expected_dimension=3,
        cache_path=str(cache_path),
        min_text_chars=10,
    )
    second_fake = FakeDenseEmbedder()
    second._dense = second_fake
    second.embed_records([
        {"text": "Day la chunk cache persistent.", "metadata": _metadata("cache-001")}
    ])
    if second_fake.calls != 0:
        raise AssertionError("Second pipeline should read vector from persistent cache.")


def test_dimension_mismatch_is_rejected() -> None:
    pipeline = EmbeddingPipeline(
        dense_provider="cohere",
        dense_model="embed-v4.0",
        expected_dimension=4,
        min_text_chars=10,
    )
    pipeline._dense = FakeDenseEmbedder()

    try:
        pipeline.embed_records([
            {"text": "Day la chunk du dai de tao embedding.", "metadata": {}}
        ])
    except ValueError as exc:
        if "dimension mismatch" not in str(exc):
            raise
        return

    raise AssertionError("Expected dimension mismatch to raise ValueError.")


def _metadata(chunk_id: str, *, section: str = "Chuong 1") -> dict:
    return {
        "source": "book.pdf",
        "file_name": "book.pdf",
        "page_number": 1,
        "title": "Book",
        "section": section,
        "chunk_id": chunk_id,
        "chunk_index": 0,
    }


if __name__ == "__main__":
    test_config_defaults_to_cohere_embedding()
    test_paid_or_extra_providers_are_rejected()
    test_embedding_records_output_format()
    test_embedding_records_handle_table_code_long_and_noise()
    test_strict_metadata_rejects_missing_source_fields()
    test_persistent_cache_reuses_vectors_across_pipeline_instances()
    test_dimension_mismatch_is_rejected()
    print("Embedding smoke test passed.")
