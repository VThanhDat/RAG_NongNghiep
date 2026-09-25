"""
Smoke tests for the simplified prompt stage.

Run from project root:
    python -B prompt/tests/test_prompt_stage.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from langchain_core.documents import Document

from prompt import CitationPromptBuilder, build_prompt_builder_from_config, get_prompt_builder


def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="Bệnh thán thư trên sầu riêng thường phát triển khi ẩm độ cao.",
            metadata={"source": "sau-rieng.pdf", "page_number": 3, "title": "Bệnh thán thư"},
        ),
        Document(
            page_content="Cần vệ sinh vườn và dùng thuốc phù hợp theo khuyến cáo kỹ thuật.",
            metadata={"source": "phong-tru.pdf", "page_number": 5},
        ),
    ]


def test_citation_prompt_contains_sources_and_question() -> None:
    builder = CitationPromptBuilder(max_context_chars=6000)
    result = builder.build("Cách phòng bệnh thán thư trên sầu riêng?", sample_docs())

    if result.template_name != "citation":
        raise AssertionError("Prompt template should be citation.")
    if result.n_sources != 2:
        raise AssertionError("Prompt should track source count.")
    user_message = result.messages[1]["content"]
    if "[NGUON 1]" not in user_message or "sau-rieng.pdf" not in user_message:
        raise AssertionError("Prompt should include numbered source context.")
    if "Cách phòng bệnh thán thư" not in user_message:
        raise AssertionError("Prompt should include the user query.")
    if "Tóm tắt và gộp các ý trùng nhau" not in user_message:
        raise AssertionError("Prompt should ask Gemini to synthesize overlapping source content.")
    if "Không chỉ đưa ra tiêu đề" not in user_message:
        raise AssertionError("Prompt should require substantive content, not headings only.")


def test_citation_extraction() -> None:
    cited = CitationPromptBuilder.extract_cited_indices("Nội dung trả lời [NGUON 2], thêm ý khác [NGUỒN 1].")
    if cited != [1, 2]:
        raise AssertionError(f"Unexpected cited indices: {cited}")


def test_factory_rejects_removed_templates() -> None:
    for template in ["basic", "conversational", "structured"]:
        try:
            get_prompt_builder(template)
        except ValueError:
            continue
        raise AssertionError(f"Removed prompt template should be rejected: {template}")


def test_build_prompt_builder_from_config() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    builder = build_prompt_builder_from_config(cfg)
    if not isinstance(builder, CitationPromptBuilder):
        raise AssertionError("Config should build CitationPromptBuilder.")
    if builder.max_context_chars != 6000:
        raise AssertionError("Config max_context_chars was not applied.")


def main() -> None:
    test_citation_prompt_contains_sources_and_question()
    test_citation_extraction()
    test_factory_rejects_removed_templates()
    test_build_prompt_builder_from_config()
    print("Prompt smoke test passed.")


if __name__ == "__main__":
    main()
