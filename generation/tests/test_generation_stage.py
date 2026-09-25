"""
Smoke tests for the simplified generation stage.

Run from project root:
    python -B generation/tests/test_generation_stage.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from generation import GeminiGenerator, GenerationResult, build_generator_from_config, get_generator
from prompt.core.base import PromptResult


def prompt_result() -> PromptResult:
    return PromptResult(
        messages=[
            {"role": "system", "content": "Bạn là trợ lý RAG."},
            {"role": "user", "content": "NGUON..."},
        ],
        n_sources=2,
        template_name="citation",
    )


def test_factory_builds_gemini_generator() -> None:
    generator = get_generator("gemini", "gemini-3.6-flash", api_key="test-key")
    if not isinstance(generator, GeminiGenerator):
        raise AssertionError("Factory should build GeminiGenerator.")
    if generator.model_name != "gemini-3.6-flash":
        raise AssertionError("Model name was not applied.")


def test_factory_rejects_remote_providers() -> None:
    for provider in ["openai", "anthropic", "google", "cohere"]:
        try:
            get_generator(provider, "some-model")
        except ValueError:
            continue
        raise AssertionError(f"Removed provider should be rejected: {provider}")


def test_build_generator_from_config() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    generator = build_generator_from_config(cfg)
    if not isinstance(generator, GeminiGenerator):
        raise AssertionError("Config should build GeminiGenerator.")
    if generator.model_name != "gemini-3.6-flash":
        raise AssertionError("Config model_name was not applied.")
    if generator.max_tokens != 1024:
        raise AssertionError("Config max_tokens was not applied.")


def test_post_process_extracts_valid_citations() -> None:
    generator = GeminiGenerator(model_name="gemini-3.6-flash", api_key="test-key")
    result = generator._post_process(
        answer="Có thể phòng bệnh bằng vệ sinh vườn [NGUON 1]. Nguồn ngoài phạm vi [NGUON 9].",
        prompt_result=prompt_result(),
        input_tokens=10,
        output_tokens=20,
    )
    if not isinstance(result, GenerationResult):
        raise AssertionError("Post-process should return GenerationResult.")
    if result.cited_sources != [1]:
        raise AssertionError(f"Invalid citation extraction: {result.cited_sources}")
    if result.provider != "gemini":
        raise AssertionError("Provider should be gemini.")


def main() -> None:
    test_factory_builds_gemini_generator()
    test_factory_rejects_remote_providers()
    test_build_generator_from_config()
    test_post_process_extracts_valid_citations()
    print("Generation smoke test passed.")


if __name__ == "__main__":
    main()
