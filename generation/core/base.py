"""
Base objects for the API generation stage.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

from prompt.core.base import PromptResult

# Khớp vị trí kết thúc câu hợp lệ:
#   "]."  "]!"  "]?"  (citation rồi dấu câu)  hoặc dấu câu đơn  ".  !  ?"
_SENTENCE_END_RE = re.compile(r"(?:\]\s*[.!?]|[.!?])")


@dataclass
class GenerationResult:
    answer: str = ""
    provider: str = ""
    model_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    cited_sources: list[int] = field(default_factory=list)


class BaseGenerator(ABC):
    def __init__(
        self,
        model_name: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        streaming: bool = False,
        provider: str = "gemini",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        self.provider = provider

    @abstractmethod
    def generate(self, prompt_result: PromptResult) -> GenerationResult:
        """Generate a complete answer from PromptResult."""

    @abstractmethod
    def stream(self, prompt_result: PromptResult) -> Iterator[str]:
        """Stream answer chunks from PromptResult."""

    def _post_process(
        self,
        answer: str,
        prompt_result: PromptResult,
        input_tokens: int = 0,
        output_tokens: int = 0,
        finish_reason: str = "stop",
    ) -> GenerationResult:
        cited_sources: list[int] = []
        if prompt_result.template_name == "citation":
            from prompt.templates.citation import CitationPromptBuilder

            cited_sources = CitationPromptBuilder.extract_cited_indices(answer)
            cited_sources = [index for index in cited_sources if 1 <= index <= prompt_result.n_sources]

        # Nếu LLM bị dừng do hết token (không phải dừng tự nhiên),
        # cắt tại dấu câu cuối cùng để tránh câu dở.
        cleaned = answer.strip()
        if finish_reason in ("max_tokens", "length", "max_output_tokens"):
            cleaned = _trim_to_sentence(cleaned)

        return GenerationResult(
            answer=cleaned,
            provider=self.provider,
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            cited_sources=cited_sources,
        )


def _trim_to_sentence(text: str) -> str:
    """Cắt text tại dấu câu cuối cùng (. ! ?) để tránh câu bị bỏ dở.

    Ưu tiên pattern [NGUON N]. để giữ citation liền dấu câu nguyên vẹn.
    Nếu không tìm thấy dấu câu nào, trả về text nguyên gốc.
    """
    if not text:
        return text
    last_pos = -1
    for m in _SENTENCE_END_RE.finditer(text):
        last_pos = m.end()
    if last_pos > 0:
        return text[:last_pos].rstrip()
    return text
