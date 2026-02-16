from __future__ import annotations

from autodl.config import LLMConfig
from autodl.llm.base import LLMProvider
from autodl.llm.ollama_provider import OllamaProvider
from autodl.llm.openai_provider import OpenAIProvider


class NullProvider(LLMProvider):
    def suggest(self, system_prompt: str, user_prompt: str) -> str:
        return '{"version":"1.0","node_recommendations":[],"risk_flags":[],"questions_for_user":[]}'


def make_llm_provider(config: LLMConfig) -> LLMProvider:
    if config.provider == "ollama":
        return OllamaProvider(config)
    if config.provider == "openai":
        return OpenAIProvider(config)
    return NullProvider()
