from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from autodl.config import load_config, save_default_config
from autodl.io import read_input, write_parquet
from autodl.llm.factory import make_llm_provider
from autodl.policy.director import NarrativeInput, apply_director_plan, build_director_plan, load_narrative_input
from autodl.policy.engine import select_text_backend, should_call_llm
from autodl.preprocess import preprocess_dataframe
from autodl.profiling import profile_dataframe
from autodl.tracking.factory import make_tracker
from autodl.train import train_with_optuna
from autodl.utils import ensure_dir, utc_run_id, write_json, write_jsonl

app = typer.Typer(help="Local-first AutoDL CLI scaffolding.")

def _resolve_target_column(df_columns: list[str], target: str) -> str:
    if target in df_columns:
        return target
    lookup = {c.lower(): c for c in df_columns}
    resolved = lookup.get(target.lower())
    if resolved is not None:
        return resolved
    raise typer.BadParameter(f"Target column '{target}' was not found. Available columns: {', '.join(df_columns)}")


def _resolve_train_output_context(parquet: Path, runs_dir: str, run_id: str | None) -> tuple[str, Path, str]:
    if run_id:
        return run_id, ensure_dir(Path(runs_dir) / run_id), "explicit_run_id"

    parquet_name = parquet.name.lower()
    parquet_parent = parquet.parent
    if parquet_name == "preprocessed.parquet" and parquet_parent.exists():
        return parquet_parent.name, ensure_dir(parquet_parent), "inferred_from_parquet_parent"

    generated = utc_run_id("train")
    return generated, ensure_dir(Path(runs_dir) / generated), "generated_train_run_id"


def _run_preprocess(
    data: Path,
    target: str,
    config_path: Path | None,
    narrative_file: Path | None,
    narrative: str | None,
    use_director: bool,
    approve_low_confidence: bool,
    run_id: str | None = None,
    run_dir: Path | None = None,
) -> dict[str, str]:
    config = load_config(config_path)
    df = read_input(data)

    resolved_target = _resolve_target_column(list(df.columns), target)
    effective_run_id = run_id or utc_run_id("prep")
    effective_run_dir = run_dir or ensure_dir(Path(config.runs_dir) / effective_run_id)

    narrative_input = load_narrative_input(
        narrative_file=narrative_file,
        narrative_text=narrative,
        config_narrative=NarrativeInput.model_validate(config.narrative.model_dump(mode="json")),
    )

    df_for_preprocess = df
    target_for_preprocess = resolved_target
    config_for_preprocess = config.model_copy(deep=True)
    director_build = None
    director_decision_log: list[dict] = []

    if use_director:
        profile = profile_dataframe(df, target=resolved_target, passthrough_cols=config.preprocess.passthrough_columns)
        director_build = build_director_plan(
            df=df,
            target=resolved_target,
            profile=profile,
            config=config,
            narrative=narrative_input,
        )

        if (
            director_build.plan.confidence < config.policy.user_confirmation_threshold
            and not approve_low_confidence
        ):
            review_summary = " ".join(director_build.plan.review_questions) or "Director flagged low confidence."
            should_apply = typer.confirm(
                f"Director confidence is {director_build.plan.confidence:.2f}, below threshold {config.policy.user_confirmation_threshold:.2f}. {review_summary} Apply plan?",
                default=False,
            )
            if not should_apply:
                raise typer.Abort()

        applied = apply_director_plan(df=df, target=resolved_target, plan=director_build.plan)
        df_for_preprocess = applied.dataframe
        target_for_preprocess = applied.target
        director_decision_log = applied.decision_log

        config_for_preprocess.preprocess = config.preprocess.model_copy(
            update={
                "text_backend": director_build.plan.preprocessing.text_backend,
                "normalize_numeric": director_build.plan.preprocessing.normalize_numeric,
                "one_hot_categorical": director_build.plan.preprocessing.one_hot_categorical,
                "passthrough_columns": director_build.plan.preprocessing.passthrough_columns,
            }
        )

    prep = preprocess_dataframe(df_for_preprocess, target=target_for_preprocess, config=config_for_preprocess)
    out_path = effective_run_dir / "preprocessed.parquet"
    write_parquet(prep.transformed, out_path)

    if director_build is not None:
        write_json(effective_run_dir / "director_plan.json", director_build.plan.model_dump(mode="json"))
        write_jsonl(effective_run_dir / "decision_log.jsonl", director_decision_log)
        prep.metadata["director"] = {
            "source": director_build.source,
            "confidence": director_build.plan.confidence,
            "review_questions": director_build.plan.review_questions,
            "llm_error": director_build.llm_error,
            "narrative_applied": narrative_input.model_dump(mode="json"),
        }

    write_json(effective_run_dir / "preprocess_metadata.json", prep.metadata)
    write_json(
        effective_run_dir / "preprocess_manifest.json",
        {
            "run_id": effective_run_id,
            "stage": "preprocess",
            "input": str(data),
            "target": target_for_preprocess,
            "requested_target": target,
            "resolved_target": resolved_target,
            "output_parquet": str(out_path),
            "config": config_for_preprocess.model_dump(mode="json"),
            "director_enabled": use_director,
        },
    )
    return {
        "run_id": effective_run_id,
        "run_dir": str(effective_run_dir),
        "output_parquet": str(out_path),
        "target": target_for_preprocess,
    }


def _run_train(
    parquet: Path,
    target: str,
    config_path: Path | None,
    primary_metric: str | None = None,
    run_id: str | None = None,
    run_dir: Path | None = None,
) -> dict[str, str]:
    config = load_config(config_path)
    parquet_df = pd.read_parquet(parquet)
    resolved_target = _resolve_target_column(list(parquet_df.columns), target)

    effective_run_id = run_id or utc_run_id("train")
    effective_run_dir = run_dir or ensure_dir(Path(config.runs_dir) / effective_run_id)

    tracker = make_tracker(config.tracking)
    tracker.start_run(effective_run_id, effective_run_dir, params=config.model_dump(mode="json"))

    summary = train_with_optuna(
        parquet_path=parquet,
        target=resolved_target,
        run_dir=effective_run_dir,
        config=config,
        primary_metric_override=primary_metric,
    )
    write_json(effective_run_dir / "metrics_summary.json", summary)
    tracker.log_metrics(
        {
            "sample_rows": summary["sample_rows"],
            "n_features": summary["n_features"],
            "best_stage1_value": summary["best_stage1_value"],
            "best_final_score": summary["best_final_score"],
        }
    )
    tracker.finish_run()

    write_json(
        effective_run_dir / "training_manifest.json",
        {
            "run_id": effective_run_id,
            "stage": "train",
            "input_parquet": str(parquet),
            "target": resolved_target,
            "requested_target": target,
            "config": config.model_dump(mode="json"),
            "notes": "Training completed with TensorFlow + Optuna.",
        },
    )
    return {
        "run_id": effective_run_id,
        "run_dir": str(effective_run_dir),
        "target": resolved_target,
        "report_path": str(summary.get("report_path", "")),
    }

@app.command("init-config")
def init_config(path: Path = typer.Option(Path("config.yaml"), help="Path to write default config.")) -> None:
    save_default_config(path)
    typer.echo(f"Wrote config template to: {path}")


@app.command("profile")
def profile(
    data: Path = typer.Option(..., help="CSV input path."),
    target: str = typer.Option(..., help="Target column name."),
    config_path: Path | None = typer.Option(None, "--config", help="Optional YAML config path."),
) -> None:
    config = load_config(config_path)
    df = read_input(data)

    resolved_target = _resolve_target_column(list(df.columns), target)
    run_id = utc_run_id("profile")
    run_dir = ensure_dir(Path(config.runs_dir) / run_id)

    result = profile_dataframe(df, target=resolved_target, passthrough_cols=config.preprocess.passthrough_columns)
    payload = {
        "run_id": run_id,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "numeric_cols": result.numeric_cols,
        "categorical_cols": result.categorical_cols,
        "text_cols": result.text_cols,
        "passthrough_cols": result.passthrough_cols,
        "missing_rates": result.missing_rates,
        "skew": result.skew,
        "sparse_ratio_estimate": result.sparse_ratio_estimate,
    }
    write_json(run_dir / "profile.json", payload)
    typer.echo(f"Profile written to {run_dir / 'profile.json'}")


@app.command("preprocess")
def preprocess(
    data: Path = typer.Option(..., help="CSV input path."),
    target: str = typer.Option(..., help="Target column name."),
    config_path: Path | None = typer.Option(None, "--config", help="Optional YAML config path."),
    narrative_file: Path | None = typer.Option(None, "--narrative-file", help="Optional narrative YAML/JSON path."),
    narrative: str | None = typer.Option(None, "--narrative", help="Optional narrative summary text."),
    use_director: bool = typer.Option(True, "--use-director/--no-director", help="Enable director plan generation and application."),
    approve_low_confidence: bool = typer.Option(False, "--approve-low-confidence", help="Apply director plan even when confidence is below the policy threshold."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run id. Use the same value in `train --run-id` to keep both stages in one directory."),
) -> None:
    resolved_run_dir = None
    if run_id:
        config = load_config(config_path)
        resolved_run_dir = ensure_dir(Path(config.runs_dir) / run_id)

    result = _run_preprocess(
        data=data,
        target=target,
        config_path=config_path,
        narrative_file=narrative_file,
        narrative=narrative,
        use_director=use_director,
        approve_low_confidence=approve_low_confidence,
        run_id=run_id,
        run_dir=resolved_run_dir,
    )
    typer.echo(f"Preprocessed parquet written to {result['output_parquet']}")


@app.command("policy-suggest")
def policy_suggest(
    rows: int = typer.Option(..., help="Estimated row count."),
    budget_minutes: int = typer.Option(120, help="Time budget in minutes."),
    deployment_target: str = typer.Option("tensorflow_saved_model", help="Deployment target."),
    config_path: Path | None = typer.Option(None, "--config", help="Optional YAML config path."),
) -> None:
    config = load_config(config_path)
    base = select_text_backend(rows, budget_minutes, deployment_target=deployment_target)

    if should_call_llm(base["confidence"], config.policy.llm_fallback_threshold):
        provider = make_llm_provider(config.llm)
        system_prompt = "You are an AutoDL policy advisor. Return strict JSON only."
        user_prompt = json.dumps(
            {
                "request_type": "policy_suggestion",
                "node": "N04",
                "deterministic_candidate": base,
                "constraints": {"must_be_cli_reproducible": True},
            }
        )
        response = provider.suggest(system_prompt, user_prompt)
        typer.echo(response)
        return

    typer.echo(json.dumps(base, indent=2))


@app.command("train")
def train(
    parquet: Path = typer.Option(..., help="Path to preprocessed parquet."),
    target: str = typer.Option(..., help="Target column name."),
    config_path: Path | None = typer.Option(None, "--config", help="Optional YAML config path."),
    primary_metric: str | None = typer.Option(None, "--primary-metric", help="Optional override for report primary metric (must match an available evaluation metric)."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run id. If omitted and parquet is runs/*/preprocessed.parquet, output stays in that same run directory."),
) -> None:
    config = load_config(config_path)
    resolved_run_id, resolved_run_dir, resolution_source = _resolve_train_output_context(
        parquet=parquet,
        runs_dir=config.runs_dir,
        run_id=run_id,
    )
    typer.echo(f"Training output context: run_id={resolved_run_id}, run_dir={resolved_run_dir}, source={resolution_source}")
    result = _run_train(
        parquet=parquet,
        target=target,
        config_path=config_path,
        primary_metric=primary_metric,
        run_id=resolved_run_id,
        run_dir=resolved_run_dir,
    )
    typer.echo(f"Training run complete. See {result['run_dir']}")


@app.command("run-full")
def run_full(
    data: Path = typer.Option(..., help="CSV input path."),
    target: str = typer.Option(..., help="Target column name."),
    config_path: Path | None = typer.Option(None, "--config", help="Optional YAML config path."),
    narrative_file: Path | None = typer.Option(None, "--narrative-file", help="Optional narrative YAML/JSON path."),
    narrative: str | None = typer.Option(None, "--narrative", help="Optional narrative summary text."),
    use_director: bool = typer.Option(True, "--use-director/--no-director", help="Enable director plan generation and application."),
    approve_low_confidence: bool = typer.Option(False, "--approve-low-confidence", help="Apply director plan even when confidence is below the policy threshold."),
    primary_metric: str | None = typer.Option(None, "--primary-metric", help="Optional override for report primary metric (must match an available evaluation metric)."),
) -> None:
    typer.echo("[N00] Triggering unified flow: preprocess -> train")
    config = load_config(config_path)
    run_id = utc_run_id("run")
    run_dir = ensure_dir(Path(config.runs_dir) / run_id)

    prep_result = _run_preprocess(
        data=data,
        target=target,
        config_path=config_path,
        narrative_file=narrative_file,
        narrative=narrative,
        use_director=use_director,
        approve_low_confidence=approve_low_confidence,
        run_id=run_id,
        run_dir=run_dir,
    )
    typer.echo(f"[N11] Preprocess complete: {prep_result['run_dir']}")

    train_result = _run_train(
        parquet=Path(prep_result["output_parquet"]),
        target=prep_result["target"],
        config_path=config_path,
        primary_metric=primary_metric,
        run_id=run_id,
        run_dir=run_dir,
    )
    typer.echo(f"[N18] Training complete: {train_result['run_dir']}")
    if train_result.get("report_path"):
        typer.echo(f"[N19] Report: {train_result['report_path']}")


if __name__ == "__main__":
    app()
