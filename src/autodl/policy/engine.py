from __future__ import annotations

from typing import Any


def select_text_backend(n_rows: int, time_budget_minutes: int, deployment_target: str = "tensorflow_saved_model") -> dict[str, Any]:
    if n_rows > 1_000_000 or time_budget_minutes <= 60:
        return {
            "action": "hashing",
            "confidence": 0.9,
            "reason": "Large data or tight budget favors fast hashing for tuning stage.",
        }
    if deployment_target == "tensorflow_saved_model":
        return {
            "action": "tfidf",
            "confidence": 0.8,
            "reason": "Balanced speed/quality for local CPU-friendly preprocessing.",
        }
    return {
        "action": "tfidf",
        "confidence": 0.75,
        "reason": "Default text backend.",
    }


def should_call_llm(confidence: float, threshold: float) -> bool:
    return confidence < threshold
