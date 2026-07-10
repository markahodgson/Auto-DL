from __future__ import annotations

from autodl.config import TrackingConfig
from autodl.tracking.base import Tracker
from autodl.tracking.local_tracker import LocalTracker
from autodl.tracking.wandb_tracker import WandbTracker


def make_tracker(config: TrackingConfig) -> Tracker:
    if not config.enabled:
        return LocalTracker()
    if config.backend == "wandb":
        return WandbTracker(config)
    return LocalTracker()
