# DNN Automation (CLI-centric)

Local-first AutoDL scaffolding for wide/sparse tabular data with optional text features.

## Goals
- Keep data confidential and local by default
- Use CSV as primary input
- Store intermediate and model-ready datasets as Parquet
- Keep LLM and experiment tracking as configurable abstractions

## Current scaffold features
- Typer-based CLI with commands: `init-config`, `profile`, `preprocess`, `policy-suggest`, `train`
- Data profiling: numeric/categorical/text detection, missing rate, skew, sparse estimate
- Preprocessing: NaN handling, normalization, categorical one-hot, text backend routing, passthrough columns
- Intermediate output: `preprocessed.parquet`
- Training command scaffold with staged sampling metadata (ready for TensorFlow+Optuna integration)

## Local-first defaults
- Tracking defaults to local artifacts (`tracking.backend: local`)
- W&B remains optional via config toggle
- LLM defaults to `none`, with pluggable providers (`ollama`, `openai`)

## Quickstart
1. Install package in editable mode:

	`pip install -e .`

2. Write a default config:

	`autodl init-config --path config.yaml`

3. Profile a dataset:

	`autodl profile --data data/train.csv --target label --config config.yaml`

4. Preprocess and store Parquet:

	`autodl preprocess --data data/train.csv --target label --config config.yaml`

5. Run training scaffold on preprocessed data:

	`autodl train --parquet runs/<prep_run_id>/preprocessed.parquet --target label --config config.yaml`

## Run artifact conventions
- `runs/<run_id>/run_manifest.json`
- `runs/<run_id>/profile.json` (profile stage)
- `runs/<run_id>/preprocessed.parquet` (preprocess stage)
- `runs/<run_id>/preprocess_metadata.json`
- `runs/<run_id>/metrics_summary.json` (train stage scaffold)

## Optional dependencies
- LLM providers: `pip install -e .[llm]`
- W&B tracking: `pip install -e .[tracking]`
- Training stack (next step): `pip install -e .[train]`
