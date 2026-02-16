from __future__ import annotations

import json
from pathlib import Path

import typer

from autodl.config import load_config, save_default_config
from autodl.io import read_input, write_parquet
from autodl.llm.factory import make_llm_provider
from autodl.policy.engine import select_text_backend, should_call_llm
from autodl.preprocess import preprocess_dataframe
from autodl.profiling import profile_dataframe
from autodl.tracking.factory import make_tracker
from autodl.train import train_placeholder
from autodl.utils import ensure_dir, utc_run_id, write_json

app = typer.Typer(help="Local-first AutoDL CLI scaffolding.")


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

    run_id = utc_run_id("profile")
    run_dir = ensure_dir(Path(config.runs_dir) / run_id)

    result = profile_dataframe(df, target=target, passthrough_cols=config.preprocess.passthrough_columns)
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
) -> None:
    config = load_config(config_path)
    df = read_input(data)

    run_id = utc_run_id("prep")
    run_dir = ensure_dir(Path(config.runs_dir) / run_id)

    prep = preprocess_dataframe(df, target=target, config=config)
    out_path = run_dir / "preprocessed.parquet"
    write_parquet(prep.transformed, out_path)

    write_json(run_dir / "preprocess_metadata.json", prep.metadata)
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "stage": "preprocess",
            "input": str(data),
            "target": target,
            "output_parquet": str(out_path),
            "config": config.model_dump(mode="json"),
        },
    )
    typer.echo(f"Preprocessed parquet written to {out_path}")


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
) -> None:
    config = load_config(config_path)

    run_id = utc_run_id("train")
    run_dir = ensure_dir(Path(config.runs_dir) / run_id)

    tracker = make_tracker(config.tracking)
    tracker.start_run(run_id, run_dir, params=config.model_dump(mode="json"))

    summary = train_placeholder(parquet_path=parquet, target=target, run_dir=run_dir, config=config)
    write_json(run_dir / "metrics_summary.json", summary)
    tracker.log_metrics({"sample_rows": summary["sample_rows"], "n_features": summary["n_features"]})
    tracker.finish_run()

    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "stage": "train",
            "input_parquet": str(parquet),
            "target": target,
            "config": config.model_dump(mode="json"),
            "notes": "Training command is currently scaffold-only; integrate TensorFlow+Optuna next.",
        },
    )

    typer.echo(f"Training scaffold run complete. See {run_dir}")


if __name__ == "__main__":
    app()
