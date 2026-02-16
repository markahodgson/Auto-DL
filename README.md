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
- Training: staged TensorFlow + Optuna pipeline (sampled tuning, finalist retraining, best model export)

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

5. Run training on preprocessed data:

	`autodl train --parquet runs/<prep_run_id>/preprocessed.parquet --target label --config config.yaml`

## CLI preprocessing interaction (CSV -> profile -> preprocess)
Use this flow when you want to see how the CLI identifies numeric/categorical/text columns and applies preprocessing choices.

1. Run profiling first (column identification pass):

	`autodl profile --data data/your_file.csv --target label --config config.yaml`

2. Open the generated profile artifact to inspect detected column groups:

	`runs/<profile_run_id>/profile.json`

	Key fields to review:
	- `numeric_cols`
	- `categorical_cols`
	- `text_cols`
	- `missing_rates`
	- `skew`

3. Run preprocessing:

	`autodl preprocess --data data/your_file.csv --target label --config config.yaml`

4. Inspect preprocessing outputs:

	- transformed data: `runs/<prep_run_id>/preprocessed.parquet`
	- metadata summary: `runs/<prep_run_id>/preprocess_metadata.json`

### How preprocessing choices are controlled
Preprocessing behavior is configured in `config.yaml` under `preprocess:`.

Example:

```yaml
preprocess:
  normalize_numeric: true
  one_hot_categorical: true
  text_backend: hashing
  passthrough_columns: [record_id]
  max_text_vocab: 50000
  text_hash_features: 262144
```

What each choice does:
- `normalize_numeric`: standard-scales detected numeric columns.
- `one_hot_categorical`: one-hot encodes detected categorical columns.
- `text_backend`:
  - `none`: no text vectorization
  - `hashing`: fast sparse text features (default)
  - `tfidf`: vocabulary-based sparse text features
  - `tf_text_vectorization`: leaves raw text path for TF-graph-oriented flows
- `passthrough_columns`: keeps listed columns untransformed in output.

### Notes on automatic detection
- Numeric detection uses pandas numeric dtypes.
- Text detection is heuristic on string/object columns (length + uniqueness patterns).
- Non-numeric, non-text columns are treated as categorical by default.
- Missing values are imputed during preprocess (numeric median, categorical/text sentinel handling).

## Run artifact conventions
- `runs/<run_id>/run_manifest.json`
- `runs/<run_id>/profile.json` (profile stage)
- `runs/<run_id>/preprocessed.parquet` (preprocess stage)
- `runs/<run_id>/preprocess_metadata.json`
- `runs/<run_id>/metrics_summary.json` (train metrics)
- `runs/<run_id>/optuna_trials.parquet`
- `runs/<run_id>/best_params.json`
- `runs/<run_id>/REPORT.md` (human-readable training summary)
- `runs/<run_id>/model/` (TensorFlow SavedModel)

### Training report outputs
After `autodl train`, the run directory now includes:
- `REPORT.md` with model performance summary and hyperparameter findings
- optional `LLM Summary` section in `REPORT.md` when `llm.provider` is enabled
- confusion matrix and evaluation plots for classification tasks
- threshold analysis files for binary classification (`threshold_metrics.csv`, threshold plot)
- optimization progress plot (`optuna_progress.png`)

## Optional dependencies
- LLM providers: `pip install -e .[llm]`
- W&B tracking: `pip install -e .[tracking]`
- Training stack (next step): `pip install -e .[train]`
