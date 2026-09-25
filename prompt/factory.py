"""
Factory for the simplified prompt stage.
"""
from __future__ import annotations

from prompt.core.base import BasePromptBuilder
from prompt.templates.citation import CitationPromptBuilder

DEFAULT_TEMPLATE = "citation"


def get_prompt_builder(template: str = DEFAULT_TEMPLATE, **kwargs) -> BasePromptBuilder:
    if template != DEFAULT_TEMPLATE:
        raise ValueError("Only the citation prompt template is supported.")
    return CitationPromptBuilder(**kwargs)


def build_prompt_builder_from_config(cfg: dict) -> BasePromptBuilder:
    prompt_cfg = cfg.get("query_pipeline", {}).get("prompt", {})
    template = prompt_cfg.get("template", DEFAULT_TEMPLATE)
    return get_prompt_builder(
        template,
        language=prompt_cfg.get("language", cfg.get("data", {}).get("language", "vi")),
        max_context_chars=prompt_cfg.get("max_context_chars", 6000),
        system_instruction=prompt_cfg.get("system_instruction", ""),
        validate_citations=prompt_cfg.get("validate_citations", True),
    )
