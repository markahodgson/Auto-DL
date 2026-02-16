from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from autodl.config import AppConfig
from autodl.llm.factory import make_llm_provider
from autodl.policy.engine import should_call_llm


class PrimaryMetricDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "1.0"
    metric: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    source: str = "deterministic"


class _PrimaryMetricLLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    metric: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


def decide_primary_metric(
    task_type: str,
    evaluation_metrics: dict[str, float],
    y_encoded: np.ndarray,
    preprocess_summary: dict[str, Any],
    config: AppConfig,
) -> PrimaryMetricDecision:
    deterministic = _deterministic_metric(task_type, evaluation_metrics, y_encoded, preprocess_summary)

    if config.llm.provider == "none":
        return deterministic

    if not should_call_llm(deterministic.confidence, config.policy.llm_fallback_threshold):
        return deterministic

    payload = {
        "request_type": "primary_metric_selection_v1",
        "task_type": task_type,
        "available_metrics": list(evaluation_metrics.keys()),
        "class_counts": _class_counts(y_encoded) if task_type in {"binary", "multiclass"} else {},
        "preprocess_summary": preprocess_summary,
        "deterministic_candidate": deterministic.model_dump(mode="json"),
        "constraints": {
            "must_select_from_available_metrics": True,
        },
    }

    system_prompt = (
        "You are an AutoDL metric director. Return strict JSON only. "
        "Select exactly one primary metric from available_metrics. "
        "Favor ROC-AUC for rare binary outcomes when available."
    )

    try:
        provider = make_llm_provider(config.llm)
        response_text = provider.suggest(system_prompt, json.dumps(payload))
        parsed = _PrimaryMetricLLMResponse.model_validate_json(response_text)
        if parsed.metric not in evaluation_metrics:
            raise ValueError(f"LLM selected unavailable metric: {parsed.metric}")
        return PrimaryMetricDecision(
            version=parsed.version,
            metric=parsed.metric,
            confidence=parsed.confidence,
            rationale=parsed.rationale,
            source="llm",
        )
    except (ValidationError, ValueError, RuntimeError, Exception) as exc:
        fallback = deterministic.model_copy(
            update={
                "source": "deterministic_fallback",
                "rationale": f"{deterministic.rationale} LLM fallback failed: {exc}",
            }
        )
        return fallback


def _deterministic_metric(
    task_type: str,
    evaluation_metrics: dict[str, float],
    y_encoded: np.ndarray,
    preprocess_summary: dict[str, Any],
) -> PrimaryMetricDecision:
    available = set(evaluation_metrics.keys())
    business_goal = str((preprocess_summary.get("narrative_business_goal") or "")).lower()

    def choose(candidates: list[str]) -> str:
        for candidate in candidates:
            if candidate in available:
                return candidate
        return next(iter(evaluation_metrics.keys())) if evaluation_metrics else "accuracy"

    if task_type == "regression":
        metric = choose(["rmse", "mae", "r2"])
        return PrimaryMetricDecision(
            metric=metric,
            confidence=0.95,
            rationale="Regression objective defaults to error-based metric priority.",
            source="deterministic",
        )

    if task_type == "binary":
        counts = _class_counts(y_encoded)
        imbalance_ratio = _imbalance_ratio(counts)

        if imbalance_ratio < 0.20 and "roc_auc" in available:
            return PrimaryMetricDecision(
                metric="roc_auc",
                confidence=0.93,
                rationale="Rare/imbalanced binary outcome detected; ROC-AUC is robust to thresholding.",
                source="deterministic",
            )

        if "recall" in business_goal:
            metric = choose(["recall", "f1", "roc_auc", "accuracy"])
            return PrimaryMetricDecision(
                metric=metric,
                confidence=0.85,
                rationale="Business goal emphasizes recall.",
                source="deterministic",
            )

        if "precision" in business_goal:
            metric = choose(["precision", "f1", "roc_auc", "accuracy"])
            return PrimaryMetricDecision(
                metric=metric,
                confidence=0.85,
                rationale="Business goal emphasizes precision.",
                source="deterministic",
            )

        if "f1" in business_goal:
            metric = choose(["f1", "roc_auc", "accuracy"])
            return PrimaryMetricDecision(
                metric=metric,
                confidence=0.84,
                rationale="Business goal emphasizes F1.",
                source="deterministic",
            )

        metric = choose(["accuracy", "f1", "roc_auc", "precision", "recall"])
        return PrimaryMetricDecision(
            metric=metric,
            confidence=0.8,
            rationale="Binary objective with no explicit preference; using general-purpose metric order.",
            source="deterministic",
        )

    metric = choose(["f1_macro", "accuracy", "recall_macro", "precision_macro"])
    return PrimaryMetricDecision(
        metric=metric,
        confidence=0.9,
        rationale="Multiclass objective defaults to macro-F1 for balanced class emphasis.",
        source="deterministic",
    )


def _class_counts(y_encoded: np.ndarray) -> dict[str, int]:
    if y_encoded.size == 0:
        return {}
    values, counts = np.unique(y_encoded, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts, strict=False)}


def _imbalance_ratio(counts: dict[str, int]) -> float:
    if not counts:
        return 1.0
    values = np.asarray(list(counts.values()), dtype=np.float64)
    return float(np.min(values) / max(np.max(values), 1.0))
