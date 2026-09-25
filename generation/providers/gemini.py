"""Google Gemini text generation provider."""

from __future__ import annotations

import os
from typing import Iterator

from generation.core.base import BaseGenerator, GenerationResult
from prompt.core.base import PromptResult


class GeminiGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str = "gemini-3.6-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        streaming: bool = False,
    ):
        super().__init__(model_name, temperature, max_tokens, streaming, provider="gemini")
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def _get_client(self):
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Add it to .env or the deployment secret store."
            )
        from google import genai

        return genai.Client(api_key=self._api_key)

    @staticmethod
    def _message_parts(prompt_result: PromptResult) -> tuple[str, list[dict]]:
        system_parts: list[str] = []
        contents: list[dict] = []
        for message in prompt_result.messages:
            role = message.get("role", "user")
            content = str(message.get("content", ""))
            if role == "system":
                system_parts.append(content)
            else:
                contents.append({"role": "model" if role == "assistant" else "user", "parts": [{"text": content}]})
        if not contents:
            contents = [{"role": "user", "parts": [{"text": prompt_result.full_prompt}]}]
        return "\n\n".join(system_parts), contents

    def _config(self):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=self._system_instruction,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

    def _prepare(self, prompt_result: PromptResult):
        self._system_instruction, contents = self._message_parts(prompt_result)
        return contents

    @staticmethod
    def _usage(response) -> tuple[int, int, str]:
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        finish_reason = "stop"
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = getattr(candidates[0], "finish_reason", None)
            if finish:
                finish_reason = str(finish).lower().split(".")[-1]
        return int(input_tokens or 0), int(output_tokens or 0), finish_reason

    def generate(self, prompt_result: PromptResult) -> GenerationResult:
        client = self._get_client()
        response = client.models.generate_content(
            model=self.model_name,
            contents=self._prepare(prompt_result),
            config=self._config(),
        )
        answer = getattr(response, "text", None) or ""
        input_tokens, output_tokens, finish_reason = self._usage(response)
        return self._post_process(
            answer=answer,
            prompt_result=prompt_result,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

    def stream(self, prompt_result: PromptResult) -> Iterator[str]:
        client = self._get_client()
        for chunk in client.models.generate_content_stream(
            model=self.model_name,
            contents=self._prepare(prompt_result),
            config=self._config(),
        ):
            text = getattr(chunk, "text", None) or ""
            if text:
                yield text
