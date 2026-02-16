from __future__ import annotations

import numpy as np

from autodl.config import AppConfig
from autodl.policy.train_director import build_training_policy


def test_regression_policy_defaults_to_mse() -> None:
    config = AppConfig()
    y = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    decision = build_training_policy(y_encoded=y, task_type="regression", n_classes=1, config=config)

    assert decision.loss == "mse"
    assert decision.use_class_weights is False


def test_binary_severe_imbalance_uses_focal_and_weights() -> None:
    config = AppConfig()
    y = np.array([0] * 99 + [1])

    decision = build_training_policy(y_encoded=y, task_type="binary", n_classes=2, config=config)

    assert decision.loss == "binary_focal_crossentropy"
    assert decision.use_class_weights is True
    assert "0" in decision.class_weight_by_label
    assert "1" in decision.class_weight_by_label


def test_binary_balanced_disables_class_weights() -> None:
    config = AppConfig()
    y = np.array([0, 1] * 20)

    decision = build_training_policy(y_encoded=y, task_type="binary", n_classes=2, config=config)

    assert decision.loss == "binary_crossentropy"
    assert decision.use_class_weights is False


def test_multiclass_imbalance_enables_weights() -> None:
    config = AppConfig()
    y = np.array([0] * 60 + [1] * 20 + [2] * 10)

    decision = build_training_policy(y_encoded=y, task_type="multiclass", n_classes=3, config=config)

    assert decision.loss == "sparse_categorical_crossentropy"
    assert decision.use_class_weights is True
