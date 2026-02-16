from __future__ import annotations

import os

from autodl.config import LLMConfig
from autodl.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def suggest(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI dependency is not installed. Install with: pip install 'dnn-automation[llm]'") from exc

        api_key = None
        if self.config.api_key_env:
            api_key = os.getenv(self.config.api_key_env)

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=self.config.model,
            temperature=self.config.temperature,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text
