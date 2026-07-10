from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from autodl.config import AppConfig
from autodl.llm.factory import make_llm_provider
from autodl.policy.engine import should_call_llm


class TrainingPolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "1.0"
    loss: str
    use_class_weights: bool
    class_weight_by_label: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    source: str = "deterministic"


class _TrainingPolicyLLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    loss: str
    use_class_weights: bool
    class_weight_by_label: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


_SYSTEM_PROMPT = (
    "You are an AutoDL training policy director. "
    "Return strict JSON only. "
    "You must choose a valid loss and optional class weights."
)


def build_training_policy(
    y_encoded: np.ndarray,
    task_type: str,
    n_classes: int,
    config: AppConfig,
) -> TrainingPolicyDecision:
    deterministic = _deterministic_policy(y_encoded=y_encoded, task_type=task_type, n_classes=n_classes)

    if config.llm.provider == "none":
        return deterministic

    if not should_call_llm(deterministic.confidence, config.policy.llm_fallback_threshold):
        return deterministic

    payload = {
        "request_type": "training_policy_v1",
        "task_type": task_type,
        "n_classes": int(n_classes),
        "class_counts": _class_counts(y_encoded),
        "deterministic_candidate": deterministic.model_dump(mode="json"),
        "allowed_losses": _allowed_losses(task_type),
    }

    try:
        provider = make_llm_provider(config.llm)
        response_text = provider.suggest(_SYSTEM_PROMPT, json.dumps(payload))
        parsed = _TrainingPolicyLLMResponse.model_validate_json(response_text)
        return TrainingPolicyDecision(
            version=parsed.version,
            loss=parsed.loss,
            use_class_weights=parsed.use_class_weights,
            class_weight_by_label=parsed.class_weight_by_label,
            confidence=parsed.confidence,
            rationale=parsed.rationale,
            source="llm",
        )
    except (ValidationError, RuntimeError, ValueError, Exception) as exc:
        fallback = deterministic.model_copy(
            update={
                "source": "deterministic_fallback",
                "rationale": f"{deterministic.rationale} LLM fallback failed: {exc}",
            }
        )
        return fallback


def _deterministic_policy(y_encoded: np.ndarray, task_type: str, n_classes: int) -> TrainingPolicyDecision:
    if task_type == "regression":
        return TrainingPolicyDecision(
            loss="mse",
            use_class_weights=False,
            class_weight_by_label={},
            confidence=0.95,
            rationale="Regression task uses MSE baseline.",
            source="deterministic",
        )

    counts = _class_counts(y_encoded)
    if len(counts) <= 1:
        return TrainingPolicyDecision(
            loss="binary_crossentropy" if task_type == "binary" else "sparse_categorical_crossentropy",
            use_class_weights=False,
            class_weight_by_label={},
            confidence=0.6,
            rationale="Single observed class in training labels; class weighting disabled.",
            source="deterministic",
        )

    values = np.array(list(counts.values()), dtype=np.float64)
    imbalance_ratio = float(np.min(values) / np.max(values))
    weights = _balanced_weights(counts)

    if task_type == "binary":
        if imbalance_ratio < 0.10:
            return TrainingPolicyDecision(
                loss="binary_focal_crossentropy",
                use_class_weights=True,
                class_weight_by_label=weights,
                confidence=0.86,
                rationale="Severe binary imbalance detected; use focal loss and class weights.",
                source="deterministic",
            )
        if imbalance_ratio < 0.25:
            return TrainingPolicyDecision(
                loss="binary_crossentropy",
                use_class_weights=True,
                class_weight_by_label=weights,
                confidence=0.9,
                rationale="Binary imbalance detected; use class-weighted binary crossentropy.",
                source="deterministic",
            )
        return TrainingPolicyDecision(
            loss="binary_crossentropy",
            use_class_weights=False,
            class_weight_by_label={},
            confidence=0.9,
            rationale="Balanced binary labels; no class weighting needed.",
            source="deterministic",
        )

    if imbalance_ratio < 0.25:
        return TrainingPolicyDecision(
            loss="sparse_categorical_crossentropy",
            use_class_weights=True,
            class_weight_by_label=weights,
            confidence=0.86,
            rationale="Multiclass imbalance detected; use class weights.",
            source="deterministic",
        )

    return TrainingPolicyDecision(
        loss="sparse_categorical_crossentropy",
        use_class_weights=False,
        class_weight_by_label={},
        confidence=0.9,
        rationale="Multiclass labels are reasonably balanced.",
        source="deterministic",
    )


def _class_counts(y_encoded: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(y_encoded, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts, strict=False)}


def _balanced_weights(counts: dict[str, int]) -> dict[str, float]:
    n_classes = max(len(counts), 1)
    total = float(sum(counts.values()))
    return {
        cls: float(total / (n_classes * max(count, 1)))
        for cls, count in counts.items()
    }


def _allowed_losses(task_type: str) -> list[str]:
    if task_type == "binary":
        return ["binary_crossentropy", "binary_focal_crossentropy"]
    if task_type == "multiclass":
        return ["sparse_categorical_crossentropy"]
    return ["mse"]
