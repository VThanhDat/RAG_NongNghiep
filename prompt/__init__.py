"""
Prompt stage for RAG Nong Nghiep.

Supported template:
    citation - grounded Vietnamese answer with inline source citations
"""

from prompt.core.base import BasePromptBuilder, PromptResult
from prompt.templates.citation import CitationPromptBuilder
from prompt.factory import build_prompt_builder_from_config, get_prompt_builder

__all__ = [
    "BasePromptBuilder",
    "PromptResult",
    "CitationPromptBuilder",
    "get_prompt_builder",
    "build_prompt_builder_from_config",
]
