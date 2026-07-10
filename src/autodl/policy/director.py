from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError
import yaml

from autodl.config import AppConfig
from autodl.llm.factory import make_llm_provider
from autodl.policy.engine import should_call_llm
from autodl.profiling import ProfileResult, profile_dataframe


class NarrativeInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dataset_summary: str = ""
    target_definition: str = ""
    business_goal: str = ""
    leakage_warnings: list[str] = Field(default_factory=list)
    column_hints: dict[str, str] = Field(default_factory=dict)


class DirectorTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    reason: str


class DirectorColumn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    semantic_type: str
    actions: list[str]
    mapping: dict[str, Any] | None = None
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class DirectorPreprocessing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_backend: str
    normalize_numeric: bool
    one_hot_categorical: bool
    passthrough_columns: list[str] = Field(default_factory=list)


class DirectorPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    task: str
    target: DirectorTarget
    confidence: float = Field(ge=0.0, le=1.0)
    narrative_summary: str = ""
    columns: list[DirectorColumn]
    preprocessing: DirectorPreprocessing
    review_questions: list[str] = Field(default_factory=list)


@dataclass
class DirectorBuildResult:
    plan: DirectorPlan
    source: str
    llm_error: str | None


@dataclass
class DirectorApplyResult:
    dataframe: pd.DataFrame
    target: str
    decision_log: list[dict[str, Any]]


_SYSTEM_PROMPT = (
    "You are the AutoDL Director for tabular ML pipelines. "
    "Return strict JSON only. Do not return markdown. "
    "Use only allowed actions and allowed text backend values. "
    "If unsure, lower confidence and add review_questions."
)


def load_narrative_input(
    narrative_file: Path | None,
    narrative_text: str | None,
    config_narrative: NarrativeInput | None = None,
) -> NarrativeInput:
    merged = NarrativeInput()
    if config_narrative is not None:
        merged = config_narrative

    if narrative_file is not None:
        raw = yaml.safe_load(narrative_file.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict) and "narrative" in raw and isinstance(raw["narrative"], dict):
            raw = raw["narrative"]
        file_narrative = NarrativeInput.model_validate(raw)
        merged = merged.model_copy(update=file_narrative.model_dump(exclude_unset=True))

    if narrative_text:
        merged = merged.model_copy(update={"dataset_summary": narrative_text.strip()})

    return merged


def build_director_plan(
    df: pd.DataFrame,
    target: str,
    profile: ProfileResult,
    config: AppConfig,
    narrative: NarrativeInput,
) -> DirectorBuildResult:
    deterministic = _build_deterministic_plan(df=df, target=target, profile=profile, config=config, narrative=narrative)

    if config.llm.provider == "none":
        return DirectorBuildResult(plan=deterministic, source="deterministic", llm_error=None)

    if not should_call_llm(deterministic.confidence, config.policy.llm_fallback_threshold):
        return DirectorBuildResult(plan=deterministic, source="deterministic", llm_error=None)

    user_payload = _build_llm_user_payload(df=df, target=target, profile=profile, config=config, narrative=narrative)

    try:
        provider = make_llm_provider(config.llm)
        response_text = provider.suggest(_SYSTEM_PROMPT, json.dumps(user_payload, ensure_ascii=False))
        plan = DirectorPlan.model_validate_json(response_text)
        return DirectorBuildResult(plan=plan, source="llm", llm_error=None)
    except (ValidationError, ValueError, RuntimeError, Exception) as exc:
        fallback = deterministic.model_copy(
            update={
                "review_questions": deterministic.review_questions
                + ["LLM output invalid; using deterministic defaults. Review narrative hints if needed."],
            }
        )
        return DirectorBuildResult(plan=fallback, source="deterministic_fallback", llm_error=str(exc))


def apply_director_plan(df: pd.DataFrame, target: str, plan: DirectorPlan) -> DirectorApplyResult:
    updated = df.copy()
    decision_log: list[dict[str, Any]] = []

    chosen_target = plan.target.column.strip() if plan.target.column else target
    if chosen_target in updated.columns:
        target = chosen_target

    columns_by_name = {c.name: c for c in plan.columns}

    for column_name, column_plan in columns_by_name.items():
        if column_name not in updated.columns:
            continue

        if column_name == target and column_plan.role in {"drop", "id", "leakage_risk"}:
            continue

        if column_plan.role in {"drop", "id", "leakage_risk"} or "drop" in column_plan.actions:
            updated = updated.drop(columns=[column_name])
            decision_log.append(
                {
                    "column": column_name,
                    "action": "drop",
                    "reason": column_plan.reason,
                    "confidence": column_plan.confidence,
                }
            )
            continue

        for action in column_plan.actions:
            if action == "binary_map" or action == "ordinal_map":
                if column_plan.mapping:
                    before_null = int(updated[column_name].isna().sum())
                    updated[column_name] = _map_series(updated[column_name], column_plan.mapping)
                    decision_log.append(
                        {
                            "column": column_name,
                            "action": action,
                            "mapping_keys": list(column_plan.mapping.keys()),
                            "before_null": before_null,
                            "after_null": int(updated[column_name].isna().sum()),
                            "confidence": column_plan.confidence,
                        }
                    )
            elif action == "parse_datetime":
                updated[column_name] = pd.to_datetime(updated[column_name], errors="coerce", utc=True)
                decision_log.append(
                    {
                        "column": column_name,
                        "action": action,
                        "confidence": column_plan.confidence,
                    }
                )

    if target not in updated.columns:
        raise ValueError(f"Target column '{target}' is missing after director plan application.")

    return DirectorApplyResult(dataframe=updated, target=target, decision_log=decision_log)


def _build_llm_user_payload(
    df: pd.DataFrame,
    target: str,
    profile: ProfileResult,
    config: AppConfig,
    narrative: NarrativeInput,
) -> dict[str, Any]:
    sample_values = _sample_values(df=df, max_values=8)
    return {
        "request_type": "director_plan_v1",
        "hard_constraints": {
            "must_be_reproducible": True,
            "cannot_use_target_leakage": True,
            "allowed_actions": [
                "impute_median",
                "impute_mode",
                "fill_missing_token",
                "one_hot",
                "binary_map",
                "ordinal_map",
                "scale_standard",
                "vectorize_hashing",
                "vectorize_tfidf",
                "parse_datetime",
                "drop",
            ],
            "allowed_text_backends": ["none", "hashing", "tfidf", "tf_text_vectorization"],
        },
        "run_context": {
            "task_hint": "auto",
            "target_hint": target,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "dataset_profile": {
                "numeric_cols": profile.numeric_cols,
                "categorical_cols": profile.categorical_cols,
                "text_cols": profile.text_cols,
                "missing_rates": profile.missing_rates,
                "sample_values": sample_values,
            },
            "user_narrative": narrative.model_dump(mode="json"),
            "config_defaults": {
                "normalize_numeric": config.preprocess.normalize_numeric,
                "one_hot_categorical": config.preprocess.one_hot_categorical,
                "text_backend": config.preprocess.text_backend,
            },
        },
    }


def _build_deterministic_plan(
    df: pd.DataFrame,
    target: str,
    profile: ProfileResult,
    config: AppConfig,
    narrative: NarrativeInput,
) -> DirectorPlan:
    id_like = _detect_id_like_columns(df=df, target=target)
    narrative_map_hints = _extract_bool_mapping_hints(narrative.column_hints)

    columns: list[DirectorColumn] = []
    feature_cols = [c for c in df.columns if c != target]

    for col in feature_cols:
        if col in id_like:
            columns.append(
                DirectorColumn(
                    name=col,
                    role="id",
                    semantic_type="unknown",
                    actions=["drop"],
                    reason="Likely identifier column based on name or high uniqueness.",
                    confidence=0.9,
                )
            )
            continue

        if col in profile.numeric_cols:
            columns.append(
                DirectorColumn(
                    name=col,
                    role="feature",
                    semantic_type="numeric",
                    actions=["impute_median", "scale_standard"],
                    reason="Numeric feature with deterministic preprocessing defaults.",
                    confidence=0.95,
                )
            )
            continue

        if col in profile.text_cols:
            vector_action = "vectorize_hashing" if config.preprocess.text_backend == "hashing" else "vectorize_tfidf"
            columns.append(
                DirectorColumn(
                    name=col,
                    role="feature",
                    semantic_type="text",
                    actions=["fill_missing_token", vector_action],
                    reason="Text feature routed to configured text backend.",
                    confidence=0.9,
                )
            )
            continue

        hint = narrative.column_hints.get(col, "")
        bool_map = narrative_map_hints.get(col)
        if bool_map is None:
            bool_map = _infer_boolean_mapping_from_data(df[col])

        if bool_map is not None:
            columns.append(
                DirectorColumn(
                    name=col,
                    role="feature",
                    semantic_type="boolean",
                    actions=["binary_map", "impute_mode"],
                    mapping=bool_map,
                    reason=f"Boolean-like feature inferred from values or narrative hint: {hint or 'detected token pair'}",
                    confidence=0.85 if hint else 0.78,
                )
            )
        else:
            columns.append(
                DirectorColumn(
                    name=col,
                    role="feature",
                    semantic_type="categorical",
                    actions=["fill_missing_token", "one_hot"],
                    reason="Categorical feature with default one-hot policy.",
                    confidence=0.88,
                )
            )

    narrative_signal = int(bool(narrative.dataset_summary.strip())) + int(bool(narrative.target_definition.strip())) + int(bool(narrative.column_hints))
    confidence = min(0.95, 0.65 + 0.1 * narrative_signal)
    review_questions: list[str] = []
    if confidence < config.policy.user_confirmation_threshold:
        review_questions.append("Narrative is sparse. Confirm target definition and any leakage columns.")

    target_kind = "classification" if df[target].nunique(dropna=True) <= 20 else "regression"

    return DirectorPlan(
        version="1.0",
        task=target_kind,
        target=DirectorTarget(column=target, reason="User-provided CLI target."),
        confidence=float(confidence),
        narrative_summary=narrative.dataset_summary[:400],
        columns=columns,
        preprocessing=DirectorPreprocessing(
            text_backend=config.preprocess.text_backend,
            normalize_numeric=config.preprocess.normalize_numeric,
            one_hot_categorical=config.preprocess.one_hot_categorical,
            passthrough_columns=config.preprocess.passthrough_columns,
        ),
        review_questions=review_questions,
    )


def _sample_values(df: pd.DataFrame, max_values: int = 8) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for col in df.columns:
        values = df[col].dropna().head(200)
        uniques: list[Any] = []
        seen: set[str] = set()
        for value in values.tolist():
            key = str(value)
            if key in seen:
                continue
            seen.add(key)
            uniques.append(value)
            if len(uniques) >= max_values:
                break
        out[col] = uniques
    return out


def _detect_id_like_columns(df: pd.DataFrame, target: str) -> set[str]:
    names = {c for c in df.columns if c != target and re.search(r"(^id$|_id$|uuid|identifier|record)", c, flags=re.IGNORECASE)}
    for col in df.columns:
        if col == target:
            continue
        unique_ratio = float(df[col].nunique(dropna=True) / max(len(df[col]), 1))
        if unique_ratio >= 0.98 and df[col].nunique(dropna=True) > 50:
            names.add(col)
    return names


def _extract_bool_mapping_hints(hints: dict[str, str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for column, hint in hints.items():
        lower = hint.lower()
        if "0/1" in lower or "binary" in lower:
            out[column] = {"0": 0, "1": 1}
        elif "m/f" in lower or "male/female" in lower or "gender" in lower or "sex" in lower:
            out[column] = {"m": 1, "male": 1, "f": 0, "female": 0}
    return out


def _infer_boolean_mapping_from_data(series: pd.Series) -> dict[str, int] | None:
    values = series.dropna().astype(str).str.strip().str.lower()
    uniques = sorted(set(values.tolist()))
    if len(uniques) != 2:
        return None

    value_set = set(uniques)
    if value_set == {"0", "1"}:
        return {"0": 0, "1": 1}
    if value_set == {"true", "false"}:
        return {"false": 0, "true": 1}
    if value_set == {"yes", "no"}:
        return {"no": 0, "yes": 1}
    if value_set == {"m", "f"}:
        return {"f": 0, "m": 1}
    if value_set == {"male", "female"}:
        return {"female": 0, "male": 1}
    return None


def _map_series(series: pd.Series, mapping: dict[str, Any]) -> pd.Series:
    lowered_mapping = {str(k).strip().lower(): v for k, v in mapping.items()}

    def _convert(value: Any) -> Any:
        if pd.isna(value):
            return value
        key = str(value).strip().lower()
        return lowered_mapping.get(key, value)

    return series.map(_convert)
