from __future__ import annotations

import numpy as np

from autodl.policy.train_director import TrainingPolicyDecision
from autodl.train import _fit_class_weight


def test_fit_class_weight_returns_none_when_disabled() -> None:
    policy = TrainingPolicyDecision(
        loss="binary_crossentropy",
        use_class_weights=False,
        class_weight_by_label={},
        confidence=0.9,
        rationale="x",
    )
    y = np.array([0, 1, 0, 1])
    assert _fit_class_weight(policy, y) is None


def test_fit_class_weight_filters_to_present_labels() -> None:
    policy = TrainingPolicyDecision(
        loss="binary_crossentropy",
        use_class_weights=True,
        class_weight_by_label={"0": 0.5, "1": 1.5, "99": 3.0},
        confidence=0.9,
        rationale="x",
    )
    y = np.array([0, 1, 0, 1])
    weights = _fit_class_weight(policy, y)
    assert weights == {0: 0.5, 1: 1.5}
