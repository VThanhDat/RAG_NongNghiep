"""
Smoke tests for the simplified top-level pipelines.

Run from project root:
    python -B pipeline/tests/test_pipeline_stage.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from langchain_core.documents import Document

from pipeline.generation_pipeline import GenerationPipeline


class MockVectorStore:
    def similarity_search(self, query: str, k: int, filter=None, **kwargs):
        return [
            Document(
                page_content="Bệnh thán thư trên sầu riêng phát triển khi ẩm độ cao.",
                metadata={"source": "sau-rieng.pdf", "score": 0.9, "rank": 1},
            ),
            Document(
                page_content="Cần vệ sinh vườn để hạn chế nguồn bệnh.",
                metadata={"source": "phong-tru.pdf", "score": 0.8, "rank": 2},
            ),
        ][:k]

    def similarity_search_with_relevance_scores(self, query: str, k: int, filter=None, **kwargs):
        return [(doc, doc.metadata["score"]) for doc in self.similarity_search(query, k, filter)]


def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="Bệnh thán thư trên sầu riêng phát triển khi ẩm độ cao.",
            metadata={"source": "sau-rieng.pdf", "page_number": 3, "score": 0.9, "rank": 1},
        ),
        Document(
            page_content="Vệ sinh vườn giúp hạn chế nguồn bệnh thán thư.",
            metadata={"source": "phong-tru.pdf", "page_number": 5, "score": 0.8, "rank": 2},
        ),
    ]


def test_generation_pipeline_until_prompt() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    vdb_result = {"vector_store": MockVectorStore(), "documents": sample_documents()}
    result = GenerationPipeline().run(
        "Cách phòng bệnh thán thư trên sầu riêng?",
        vdb_result=vdb_result,
        cfg=cfg,
        skip_generation=True,
    )
    if not result.success:
        raise AssertionError(f"Pipeline failed: {result.error}")
    if not result.final_docs:
        raise AssertionError("Pipeline should produce final context docs.")
    if result.prompt_result is None or result.prompt_result.template_name != "citation":
        raise AssertionError("Pipeline should build citation prompt.")
    if [step.step for step in result.steps] != ["pre_retrieval", "retrieval", "post_retrieval", "prompt"]:
        raise AssertionError(f"Unexpected steps: {result.steps}")


def main() -> None:
    test_generation_pipeline_until_prompt()
    print("Pipeline smoke test passed.")


if __name__ == "__main__":
    main()
