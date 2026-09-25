"""
Generation stage for RAG Nong Nghiep.

Supported providers:
    gemini - Gemini Flash API
"""

from generation.core.base import BaseGenerator, GenerationResult
from generation.factory import build_generator_from_config, get_generator
from generation.providers.gemini import GeminiGenerator

__all__ = [
    "BaseGenerator",
    "GenerationResult",
    "GeminiGenerator",
    "get_generator",
    "build_generator_from_config",
]
