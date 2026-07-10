from __future__ import annotations

import numpy as np

from autodl.config import AppConfig
from autodl.policy import metric_director


def test_binary_rare_outcome_prefers_roc_auc() -> None:
    config = AppConfig()
    y = np.array([0] * 95 + [1] * 5)
    metrics = {"accuracy": 0.9, "roc_auc": 0.84, "f1": 0.4}

    decision = metric_director.decide_primary_metric(
        task_type="binary",
        evaluation_metrics=metrics,
        y_encoded=y,
        preprocess_summary={},
        config=config,
    )

    assert decision.metric == "roc_auc"
    assert decision.source == "deterministic"


def test_binary_business_goal_recall_prefers_recall() -> None:
    config = AppConfig()
    y = np.array([0] * 70 + [1] * 30)
    metrics = {"accuracy": 0.8, "recall": 0.9, "f1": 0.75, "roc_auc": 0.8}

    decision = metric_director.decide_primary_metric(
        task_type="binary",
        evaluation_metrics=metrics,
        y_encoded=y,
        preprocess_summary={"narrative_business_goal": "maximize recall for positives"},
        config=config,
    )

    assert decision.metric == "recall"


def test_multiclass_defaults_to_f1_macro() -> None:
    config = AppConfig()
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    metrics = {"accuracy": 0.8, "f1_macro": 0.79, "precision_macro": 0.81}

    decision = metric_director.decide_primary_metric(
        task_type="multiclass",
        evaluation_metrics=metrics,
        y_encoded=y,
        preprocess_summary={},
        config=config,
    )

    assert decision.metric == "f1_macro"


def test_llm_unavailable_metric_falls_back(monkeypatch) -> None:
    class MockProvider:
        def suggest(self, system_prompt: str, user_prompt: str) -> str:
            return '{"version":"1.0","metric":"not_a_metric","confidence":0.9,"rationale":"x"}'

    monkeypatch.setattr(metric_director, "make_llm_provider", lambda _config: MockProvider())

    config = AppConfig.model_validate(
        {
            "llm": {"provider": "openai", "model": "dummy", "api_key_env": "MISSING_KEY"},
            "policy": {"llm_fallback_threshold": 1.0, "user_confirmation_threshold": 0.5},
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1])
    metrics = {"accuracy": 0.75, "f1": 0.76}

    decision = metric_director.decide_primary_metric(
        task_type="binary",
        evaluation_metrics=metrics,
        y_encoded=y,
        preprocess_summary={},
        config=config,
    )

    assert decision.source == "deterministic_fallback"
    assert decision.metric in metrics
    assert "LLM fallback failed" in decision.rationale
