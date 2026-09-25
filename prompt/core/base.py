"""
Base prompt objects for RAG Nong Nghiep.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document


@dataclass
class PromptResult:
    messages: list[dict] = field(default_factory=list)
    full_prompt: str = ""
    context_docs: list[Document] = field(default_factory=list)
    n_sources: int = 0
    template_name: str = ""


class BasePromptBuilder(ABC):
    def __init__(
        self,
        system_instruction: str = "",
        language: str = "vi",
        max_context_chars: int = 6000,
    ):
        self.system_instruction = system_instruction
        self.language = language
        self.max_context_chars = max_context_chars

    @abstractmethod
    def build(
        self,
        query: str,
        docs: list[Document],
        history: list[dict] | None = None,
    ) -> PromptResult:
        """Build chat messages and a string fallback prompt."""

    def _format_context(self, docs: list[Document]) -> str:
        parts: list[str] = []
        total_chars = 0

        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source") or doc.metadata.get("file_name") or "unknown"
            page = doc.metadata.get("page_number", doc.metadata.get("page"))
            title = doc.metadata.get("title") or doc.metadata.get("section") or ""

            ref_parts = [Path(str(source)).name]
            if page:
                ref_parts.append(f"trang {page}")
            if title:
                ref_parts.append(str(title))
            ref = " | ".join(ref_parts)

            text = doc.page_content.strip()
            block = f"[NGUON {index}] ({ref})\n{text}"

            if self.max_context_chars > 0:
                remaining = self.max_context_chars - total_chars
                if remaining <= 0:
                    break
                if len(block) > remaining:
                    block = block[:remaining].rstrip() + "..."

            parts.append(block)
            total_chars += len(block)

        return "\n\n".join(parts)

    @staticmethod
    def _messages_to_string(messages: list[dict]) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(parts)
