from __future__ import annotations

from pathlib import Path
from typing import Any

from autodl.config import TrackingConfig
from autodl.tracking.base import Tracker


class WandbTracker(Tracker):
    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self._run = None

    def start_run(self, run_id: str, run_dir: Path, params: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb is not installed. Install with: pip install 'dnn-automation[tracking]'") from exc

        self._run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            mode=self.config.mode,
            id=run_id,
            config=params,
            dir=str(run_dir),
            reinit=True,
        )

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.log(metrics)

    def finish_run(self) -> None:
        if self._run is not None:
            self._run.finish()
