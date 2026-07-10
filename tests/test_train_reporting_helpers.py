from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from autodl.config import AppConfig

from autodl.train import (
    _build_data_characteristics,
    _deterministic_metric_interpretation,
    _maybe_llm_metric_interpretation,
    _maybe_llm_summary,
    _safe_read_json,
    _score_band,
    _write_markdown_report,
)


def test_score_band_binary_and_brier() -> None:
    assert _score_band("roc_auc", 0.91) == "excellent"
    assert _score_band("accuracy", 0.8) == "fair"
    assert _score_band("brier_score", 0.1) == "good"
    assert _score_band("brier_score", 0.25) == "weak"


def test_build_data_characteristics_binary_marks_rare() -> None:
    X = pd.DataFrame({"a": [1, 1, 1, 1, 1, 1], "b": [1, 2, 3, 4, 5, 6]})
    y = np.array([0, 0, 0, 0, 0, 1])

    out = _build_data_characteristics(X_df=X, y=y, task_type="binary")

    assert out["n_rows"] == 6
    assert out["n_features"] == 2
    assert out["rare_outcome"] is True
    assert "imbalance_ratio" in out


def test_build_data_characteristics_multiclass_entropy() -> None:
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [1, 1, 2, 2, 3, 3]})
    y = np.array([0, 1, 2, 0, 1, 2])

    out = _build_data_characteristics(X_df=X, y=y, task_type="multiclass")

    assert out["class_diversity"] in {"high", "medium", "low"}
    assert 0.0 <= out["class_entropy_normalized"] <= 1.0


def test_deterministic_metric_interpretation_includes_calibration_note() -> None:
    interp = _deterministic_metric_interpretation(
        task_type="binary",
        metrics={"accuracy": 0.72, "brier_score": 0.18, "roc_auc": 0.83},
        primary_metric={"metric": "accuracy"},
        data_characteristics={
            "rare_outcome": False,
            "positive_rate": 0.4,
            "imbalance_ratio": 0.6,
            "mean_feature_unique_ratio": 0.2,
            "low_variance_columns": 2,
        },
    )

    assert interp["source"] == "deterministic"
    assert "brier_score" in interp["text"]
    assert interp["overall_assessment"] in {"good", "fair", "needs_attention"}


def test_deterministic_metric_interpretation_rare_outcome_guidance() -> None:
    interp = _deterministic_metric_interpretation(
        task_type="binary",
        metrics={"accuracy": 0.8, "roc_auc": 0.9},
        primary_metric={"metric": "accuracy"},
        data_characteristics={
            "rare_outcome": True,
            "positive_rate": 0.08,
            "imbalance_ratio": 0.09,
            "mean_feature_unique_ratio": 0.3,
            "low_variance_columns": 0,
        },
    )

    assert "consider using `roc_auc`" in interp["text"]


def test_deterministic_metric_interpretation_multiclass_branch() -> None:
    interp = _deterministic_metric_interpretation(
        task_type="multiclass",
        metrics={"f1_macro": 0.72},
        primary_metric={"metric": "f1_macro"},
        data_characteristics={
            "class_diversity": "high",
            "mean_feature_unique_ratio": 0.4,
            "low_variance_columns": 1,
        },
    )

    assert "Class diversity is `high`" in interp["text"]


def test_maybe_llm_metric_interpretation_success_and_failure(monkeypatch) -> None:
    config = AppConfig.model_validate({"llm": {"provider": "openai", "model": "dummy", "api_key_env": "MISSING_KEY"}})
    payload = {"deterministic": {"overall_assessment": "fair"}}

    class OkProvider:
        def suggest(self, system_prompt: str, user_prompt: str) -> str:
            return "- LLM metric interpretation"

    monkeypatch.setattr("autodl.train.make_llm_provider", lambda _cfg: OkProvider())
    out = _maybe_llm_metric_interpretation(config=config, payload=payload)
    assert out is not None
    assert out["source"] == "llm"
    assert out["overall_assessment"] == "fair"

    class BadProvider:
        def suggest(self, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("boom")

    monkeypatch.setattr("autodl.train.make_llm_provider", lambda _cfg: BadProvider())
    assert _maybe_llm_metric_interpretation(config=config, payload=payload) is None


def test_maybe_llm_summary_none_when_disabled_and_on_errors(monkeypatch) -> None:
    disabled = AppConfig.model_validate({"llm": {"provider": "none"}})
    assert _maybe_llm_summary(disabled, {"a": 1}) is None

    enabled = AppConfig.model_validate({"llm": {"provider": "openai", "model": "dummy", "api_key_env": "MISSING_KEY"}})

    class EmptyProvider:
        def suggest(self, system_prompt: str, user_prompt: str) -> str:
            return ""

    monkeypatch.setattr("autodl.train.make_llm_provider", lambda _cfg: EmptyProvider())
    assert _maybe_llm_summary(enabled, {"a": 1}) is None

    class RaiseProvider:
        def suggest(self, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("x")

    monkeypatch.setattr("autodl.train.make_llm_provider", lambda _cfg: RaiseProvider())
    assert _maybe_llm_summary(enabled, {"a": 1}) is None


def test_write_markdown_report_includes_key_sections(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "completed",
        "target": "target",
        "task_type": "binary",
        "problem_type": "classification",
        "full_rows": 100,
        "sample_rows": 20,
        "n_features": 10,
        "optuna_trials": 3,
        "best_stage1_trial": 1,
        "best_stage1_value": 0.7,
        "best_final_trial": 2,
        "best_final_score": 0.8,
        "model_backend": "tensorflow",
        "model_family": "dense_neural_network",
        "search_engine": "optuna",
        "preprocessing_summary": {"available": False},
        "training_progress": {
            "stage1_sample_fraction": 0.2,
            "epochs_tuning": 10,
            "epochs_final": 20,
            "patience": 3,
            "completed_trials": 3,
            "total_trials": 3,
            "finalists_retrained": 1,
        },
        "primary_metric": {
            "metric": "accuracy",
            "source": "deterministic",
            "confidence": 0.9,
            "rationale": "balanced",
        },
        "metric_interpretation": {
            "source": "deterministic",
            "overall_assessment": "good",
            "text": "- Looks good",
        },
    }
    best_params = {"learning_rate": 0.001}
    evaluation = {
        "task_type": "binary",
        "selected_threshold": 0.45,
        "test_metrics": {"accuracy": 0.8},
        "confusion_matrix": [[40, 10], [8, 42]],
    }
    training_policy = {
        "source": "deterministic",
        "loss": "binary_crossentropy",
        "use_class_weights": True,
        "confidence": 0.8,
        "rationale": "imbalance",
        "class_weight_by_label": {"0": 0.7, "1": 1.3},
    }

    report_path = _write_markdown_report(
        run_dir=run_dir,
        summary=summary,
        best_params=best_params,
        evaluation=evaluation,
        plot_paths=[],
        llm_summary="- concise llm summary",
        training_policy=training_policy,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "## Executive Summary" in text
    assert "Preprocess metadata unavailable" in text
    assert "## Training Policy" in text
    assert "## Metric Interpretation" in text
    assert "### Confusion Matrix" in text
    assert "selected_threshold" in text


def test_safe_read_json_handles_missing_and_valid(tmp_path: Path) -> None:
    missing = _safe_read_json(tmp_path / "missing.json")
    assert missing is None

    path = tmp_path / "ok.json"
    path.write_text('{"x": 1}', encoding="utf-8")
    parsed = _safe_read_json(path)
    assert parsed == {"x": 1}


def test_safe_read_json_invalid_returns_none(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not-json", encoding="utf-8")
    assert _safe_read_json(broken) is None
