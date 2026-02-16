from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def suggest(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError
