"""
Factory for the simplified generation stage.
"""
from __future__ import annotations

from generation.core.base import BaseGenerator
from generation.providers.gemini import GeminiGenerator

DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-3.6-flash"


def get_generator(provider: str = DEFAULT_PROVIDER, model_name: str = DEFAULT_MODEL, **kwargs) -> BaseGenerator:
    if provider == "gemini":
        return GeminiGenerator(model_name=model_name, **kwargs)
    raise ValueError(f"Unsupported generation provider: {provider}.")


def build_generator_from_config(cfg: dict) -> BaseGenerator:
    gen_cfg = cfg.get("query_pipeline", {}).get("generation", {})
    provider = gen_cfg.get("provider", DEFAULT_PROVIDER)
    kwargs = {
        "temperature": gen_cfg.get("temperature", 0.0),
        "max_tokens": gen_cfg.get("max_tokens", 1024),
        "streaming": gen_cfg.get("streaming", False),
    }
    if provider == "gemini":
        kwargs["api_key"] = gen_cfg.get("api_key")
    return get_generator(
        provider=provider,
        model_name=gen_cfg.get("model_name", DEFAULT_MODEL),
        **kwargs,
    )
