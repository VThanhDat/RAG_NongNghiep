"""
Smoke tests for the simplified post-retrieval stage.

Run from project root:
    python -B post_retrieval/tests/test_post_retrieval_stage.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from langchain_core.documents import Document

from post_retrieval import build_pipeline, build_pipeline_from_config
from post_retrieval.core.types import PostRetrievalResult
from post_retrieval.core.utils import deduplicate
from post_retrieval.pipeline import PostRetrievalPipeline
from post_retrieval.processors.context_orderer import ContextOrderer
from post_retrieval.processors.redundancy_filter import RedundancyFilter
from post_retrieval.processors.score_filter import ScoreFilter


def make_doc(content: str, **metadata) -> Document:
    return Document(page_content=content, metadata=metadata)


def sample_docs() -> list[Document]:
    return [
        make_doc(
            "Bệnh thán thư trên sầu riêng thường phát triển khi ẩm độ cao.",
            source="sau-rieng.pdf",
            page_number=3,
            chunk_id="c1",
            score=0.90,
            rank=1,
        ),
        make_doc(
            "Phòng trừ thán thư bằng vệ sinh vườn và dùng thuốc phù hợp.",
            source="sau-rieng.pdf",
            page_number=4,
            chunk_id="c2",
            score=0.82,
            rank=2,
        ),
        make_doc(
            "Bệnh thán thư trên sầu riêng thường phát triển khi ẩm độ cao. ",
            source="sau-rieng.pdf",
            page_number=3,
            chunk_id="c1_dup",
            score=0.80,
            rank=3,
        ),
        make_doc(
            "Thiếu kali làm mép lá cà phê cháy vàng.",
            source="ca-phe.pdf",
            page_number=8,
            chunk_id="c3",
            score=0.20,
            rank=4,
        ),
    ]


def test_deduplicate_removes_exact_duplicates() -> None:
    docs = sample_docs()
    unique = deduplicate(docs)
    if len(unique) != 3:
        raise AssertionError(f"Expected 3 unique docs, got {len(unique)}")


def test_score_filter_keeps_minimum_when_all_below_threshold() -> None:
    docs = sample_docs()
    filtered = ScoreFilter(threshold=0.99, keep_min=1).process("thán thư", docs)
    if len(filtered) != 1 or filtered[0].metadata["rank"] != 1:
        raise AssertionError("ScoreFilter should keep the top document as fail-safe.")


def test_score_filter_drops_low_scores() -> None:
    docs = sample_docs()
    filtered = ScoreFilter(threshold=0.5, keep_min=1).process("thán thư", docs)
    if any(doc.metadata["score"] < 0.5 for doc in filtered):
        raise AssertionError("ScoreFilter kept a document below threshold.")


def test_redundancy_filter_removes_near_duplicates() -> None:
    docs = [
        make_doc("bệnh thán thư do nấm gây ra trên sầu riêng", score=0.9),
        make_doc("bệnh thán thư do nấm gây ra trên cây sầu riêng", score=0.8),
        make_doc("lúa giai đoạn đẻ nhánh cần nước", score=0.7),
    ]
    filtered = RedundancyFilter(top_n=5, threshold=0.7).process("thán thư", docs)
    if len(filtered) != 2:
        raise AssertionError(f"Expected near duplicate removal, got {len(filtered)} docs.")


def test_context_orderer_relevance() -> None:
    docs = [
        make_doc("low", score=0.1),
        make_doc("high", score=0.9),
        make_doc("mid", score=0.5),
    ]
    ordered = ContextOrderer(ordering="relevance").process("query", docs)
    if [doc.page_content for doc in ordered] != ["high", "mid", "low"]:
        raise AssertionError("ContextOrderer should sort by descending score.")


def test_pipeline_processes_context() -> None:
    pipeline = PostRetrievalPipeline(
        top_n=2,
        score_threshold=0.3,
        apply_redundancy=True,
        redundancy_threshold=0.9,
        context_ordering="relevance",
    )
    result = pipeline.process("Cách phòng bệnh thán thư trên sầu riêng?", sample_docs())
    if not isinstance(result, PostRetrievalResult):
        raise AssertionError("Pipeline should return PostRetrievalResult.")
    if result.output_count != 2:
        raise AssertionError(f"Expected top_n=2 output, got {result.output_count}")
    if result.documents[0].metadata["score"] < result.documents[1].metadata["score"]:
        raise AssertionError("Output should be relevance ordered.")
    if "ScoreFilter" not in result.steps_run or "ContextOrderer" not in result.steps_run:
        raise AssertionError(f"Missing expected steps: {result.steps_run}")


def test_pipeline_empty_input() -> None:
    result = build_pipeline(top_n=5).process("query", [])
    if not result.is_empty or result.output_count != 0:
        raise AssertionError("Empty input should return empty PostRetrievalResult.")


def test_build_pipeline_from_config() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pipeline = build_pipeline_from_config(cfg)
    if pipeline.top_n != 5 or pipeline.context_ordering != "relevance":
        raise AssertionError("Config builder did not apply post_retrieval settings.")


def test_result_context_string_and_dict() -> None:
    result = PostRetrievalResult(
        query="thán thư",
        documents=sample_docs()[:2],
        steps_run=["Deduplicate", "ContextOrderer"],
        input_count=4,
        output_count=2,
        latency_ms=1.5,
    )
    context = result.to_context_string()
    if "[1]" not in context or "sau-rieng.pdf" not in context:
        raise AssertionError("Context string should include rank and source.")
    payload = result.as_dict()
    if payload["documents"][0]["page"] != 3:
        raise AssertionError("as_dict should expose page_number as page.")


def main() -> None:
    test_deduplicate_removes_exact_duplicates()
    test_score_filter_keeps_minimum_when_all_below_threshold()
    test_score_filter_drops_low_scores()
    test_redundancy_filter_removes_near_duplicates()
    test_context_orderer_relevance()
    test_pipeline_processes_context()
    test_pipeline_empty_input()
    test_build_pipeline_from_config()
    test_result_context_string_and_dict()
    print("Post-retrieval smoke test passed.")


if __name__ == "__main__":
    main()
