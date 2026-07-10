from __future__ import annotations

from autodl.config import LLMConfig
from autodl.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def suggest(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("Ollama dependency is not installed. Install with: pip install 'dnn-automation[llm]'") from exc

        client = ollama.Client(host=self.config.base_url)
        response = client.chat(
            model=self.config.model,
            options={"temperature": self.config.temperature},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"]
