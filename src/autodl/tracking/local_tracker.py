from __future__ import annotations

from pathlib import Path
from typing import Any

from autodl.tracking.base import Tracker
from autodl.utils import write_json


class LocalTracker(Tracker):
    def __init__(self) -> None:
        self.run_dir: Path | None = None

    def start_run(self, run_id: str, run_dir: Path, params: dict[str, Any]) -> None:
        self.run_dir = run_dir
        write_json(run_dir / "tracking_local_start.json", {"run_id": run_id, "params": params})

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        if self.run_dir is None:
            return
        write_json(self.run_dir / "tracking_local_metrics.json", metrics)

    def finish_run(self) -> None:
        if self.run_dir is None:
            return
        write_json(self.run_dir / "tracking_local_finish.json", {"status": "completed"})
