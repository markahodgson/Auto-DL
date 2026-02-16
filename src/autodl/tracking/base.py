from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Tracker(ABC):
    @abstractmethod
    def start_run(self, run_id: str, run_dir: Path, params: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish_run(self) -> None:
        raise NotImplementedError
