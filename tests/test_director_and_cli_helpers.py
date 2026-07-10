from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import pytest

from autodl.cli import _resolve_target_column
from autodl.cli import _resolve_train_output_context
from autodl.config import AppConfig
from autodl.policy.director import (
    DirectorColumn,
    DirectorPlan,
    DirectorPreprocessing,
    DirectorTarget,
    NarrativeInput,
    apply_director_plan,
    build_director_plan,
    load_narrative_input,
)
from autodl.profiling import profile_dataframe


def test_resolve_target_column_case_insensitive() -> None:
    resolved = _resolve_target_column(["Age", "Target"], "target")
    assert resolved == "Target"


def test_resolve_target_column_raises_when_missing() -> None:
    with pytest.raises(Exception):
        _resolve_target_column(["a", "b"], "target")


def test_resolve_train_output_context_uses_explicit_run_id(tmp_path: Path) -> None:
    run_id, run_dir, source = _resolve_train_output_context(
        parquet=tmp_path / "x.parquet",
        runs_dir=str(tmp_path / "runs"),
        run_id="shared_run",
    )

    assert run_id == "shared_run"
    assert run_dir.name == "shared_run"
    assert run_dir.exists()
    assert source == "explicit_run_id"


def test_resolve_train_output_context_uses_parquet_parent_for_preprocessed_file(tmp_path: Path) -> None:
    parquet = tmp_path / "runs" / "run_abc" / "preprocessed.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_text("", encoding="utf-8")

    run_id, run_dir, source = _resolve_train_output_context(
        parquet=parquet,
        runs_dir=str(tmp_path / "runs"),
        run_id=None,
    )

    assert run_id == "run_abc"
    assert run_dir == parquet.parent
    assert source == "inferred_from_parquet_parent"


def test_load_narrative_input_merges_file_and_text(tmp_path: Path) -> None:
    narrative_file = tmp_path / "narrative.yaml"
    narrative_file.write_text(
        """
        narrative:
          dataset_summary: file summary
          business_goal: improve recall
          column_hints:
            sex: binary sex column
        """,
        encoding="utf-8",
    )

    config_narr = NarrativeInput(dataset_summary="config summary", target_definition="label is target")

    merged = load_narrative_input(
        narrative_file=narrative_file,
        narrative_text="cli summary",
        config_narrative=config_narr,
    )

    assert merged.dataset_summary == "cli summary"
    assert merged.target_definition == "label is target"
    assert merged.business_goal == "improve recall"
    assert "sex" in merged.column_hints


def test_apply_director_plan_maps_and_drops_columns() -> None:
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "sex": ["M", "F"],
            "target": [1, 0],
        }
    )
    plan = DirectorPlan(
        version="1.0",
        task="classification",
        target=DirectorTarget(column="target", reason="provided"),
        confidence=0.9,
        columns=[
            DirectorColumn(
                name="id",
                role="id",
                semantic_type="unknown",
                actions=["drop"],
                reason="identifier",
                confidence=0.9,
            ),
            DirectorColumn(
                name="sex",
                role="feature",
                semantic_type="boolean",
                actions=["binary_map"],
                mapping={"m": 1, "f": 0},
                reason="binary",
                confidence=0.9,
            ),
        ],
        preprocessing=DirectorPreprocessing(
            text_backend="hashing",
            normalize_numeric=True,
            one_hot_categorical=True,
            passthrough_columns=[],
        ),
        review_questions=[],
    )

    result = apply_director_plan(df=df, target="target", plan=plan)

    assert "id" not in result.dataframe.columns
    assert result.dataframe["sex"].tolist() == [1, 0]
    assert result.target == "target"
    assert any(item["action"] == "drop" for item in result.decision_log)


def test_build_director_plan_uses_llm_when_threshold_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"age": [20, 30], "target": [0, 1]})
    profile = profile_dataframe(df, target="target")

    config = AppConfig.model_validate(
        {
            "llm": {"provider": "openai", "model": "dummy", "api_key_env": "MISSING_KEY"},
            "policy": {"llm_fallback_threshold": 0.8, "user_confirmation_threshold": 0.5},
        }
    )

    class MockProvider:
        def suggest(self, system_prompt: str, user_prompt: str) -> str:
            payload = {
                "version": "1.0",
                "task": "classification",
                "target": {"column": "target", "reason": "llm"},
                "confidence": 0.92,
                "narrative_summary": "",
                "columns": [
                    {
                        "name": "age",
                        "role": "feature",
                        "semantic_type": "numeric",
                        "actions": ["impute_median", "scale_standard"],
                        "reason": "numeric",
                        "confidence": 0.9,
                    }
                ],
                "preprocessing": {
                    "text_backend": "hashing",
                    "normalize_numeric": True,
                    "one_hot_categorical": True,
                    "passthrough_columns": [],
                },
                "review_questions": [],
            }
            return json.dumps(payload)

    monkeypatch.setattr("autodl.policy.director.make_llm_provider", lambda _cfg: MockProvider())

    out = build_director_plan(
        df=df,
        target="target",
        profile=profile,
        config=config,
        narrative=NarrativeInput(),
    )

    assert out.source == "llm"
    assert out.plan.target.column == "target"
    assert out.llm_error is None


def test_build_director_plan_invalid_llm_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"age": [20, 30], "target": [0, 1]})
    profile = profile_dataframe(df, target="target")
    config = AppConfig.model_validate(
        {
            "llm": {"provider": "openai", "model": "dummy", "api_key_env": "MISSING_KEY"},
            "policy": {"llm_fallback_threshold": 0.8, "user_confirmation_threshold": 0.5},
        }
    )

    class BadProvider:
        def suggest(self, system_prompt: str, user_prompt: str) -> str:
            return "{not valid json"

    monkeypatch.setattr("autodl.policy.director.make_llm_provider", lambda _cfg: BadProvider())

    out = build_director_plan(
        df=df,
        target="target",
        profile=profile,
        config=config,
        narrative=NarrativeInput(),
    )

    assert out.source == "deterministic_fallback"
    assert out.llm_error
    assert any("LLM output invalid" in question for question in out.plan.review_questions)


def test_build_director_plan_skips_llm_when_confident(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"age": [20, 30], "target": [0, 1]})
    profile = profile_dataframe(df, target="target")
    config = AppConfig.model_validate(
        {
            "llm": {"provider": "openai", "model": "dummy", "api_key_env": "MISSING_KEY"},
            "policy": {"llm_fallback_threshold": 0.5, "user_confirmation_threshold": 0.5},
        }
    )

    class ShouldNotBeCalledProvider:
        def suggest(self, system_prompt: str, user_prompt: str) -> str:
            raise AssertionError("LLM should not be called when confidence exceeds threshold")

    monkeypatch.setattr("autodl.policy.director.make_llm_provider", lambda _cfg: ShouldNotBeCalledProvider())

    out = build_director_plan(
        df=df,
        target="target",
        profile=profile,
        config=config,
        narrative=NarrativeInput(dataset_summary="has context"),
    )

    assert out.source == "deterministic"
    assert out.llm_error is None


def test_apply_director_plan_parses_datetime_and_preserves_target() -> None:
    df = pd.DataFrame(
        {
            "event_time": ["2025-01-01", "invalid"],
            "target": [1, 0],
        }
    )
    plan = DirectorPlan(
        version="1.0",
        task="classification",
        target=DirectorTarget(column="target", reason="provided"),
        confidence=0.9,
        columns=[
            DirectorColumn(
                name="event_time",
                role="feature",
                semantic_type="datetime",
                actions=["parse_datetime"],
                reason="timestamp",
                confidence=0.9,
            ),
            DirectorColumn(
                name="target",
                role="drop",
                semantic_type="unknown",
                actions=["drop"],
                reason="should be ignored for target",
                confidence=0.8,
            ),
        ],
        preprocessing=DirectorPreprocessing(
            text_backend="hashing",
            normalize_numeric=True,
            one_hot_categorical=True,
            passthrough_columns=[],
        ),
        review_questions=[],
    )

    out = apply_director_plan(df=df, target="target", plan=plan)

    assert out.target == "target"
    assert "target" in out.dataframe.columns
    assert str(out.dataframe["event_time"].dtype).startswith("datetime64")
    assert any(item["action"] == "parse_datetime" for item in out.decision_log)
