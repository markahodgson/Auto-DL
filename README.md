# Auto-DL

Automatic Deep Learning Model Generator using TensorFlow

## Overview

Auto-DL is a Python library that automatically generates, optimizes, and trains high-performance deep learning models. It leverages TensorFlow and advanced machine learning techniques to handle the entire model development pipeline with minimal user input.

## Features

- **Automatic Model Generation**: Automatically creates neural network architectures based on your data characteristics
- **Hyperparameter Optimization**: Intelligently searches for optimal hyperparameters using random search
- **Data Preprocessing**: Handles data normalization, encoding, and splitting automatically
- **Multiple Task Types**: Supports both classification and regression tasks
- **Easy-to-Use API**: Simple, scikit-learn-like interface for quick model development
- **TensorFlow Backend**: Built on TensorFlow 2.x for high performance and flexibility

## Installation

```bash
# Clone the repository
git clone https://github.com/markahodgson/Auto-DL.git
cd Auto-DL

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Classification Example

```python
from autodl import AutoDL
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Create and train Auto-DL model
auto_dl = AutoDL(task_type="classification", optimize_hyperparameters=True)
model = auto_dl.fit(X_train, y_train)

# Evaluate
metrics = auto_dl.evaluate(X_test, y_test)
print(f"Test accuracy: {metrics['accuracy']:.4f}")

# Make predictions
predictions = auto_dl.predict(X_test)
```

### Regression Example

```python
from autodl import AutoDL
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train Auto-DL model
auto_dl = AutoDL(task_type="regression", optimize_hyperparameters=True)
model = auto_dl.fit(X_train, y_train)

# Evaluate
metrics = auto_dl.evaluate(X_test, y_test)
print(f"Test MAE: {metrics['mae']:.4f}")
```

## Usage

### Basic Usage

```python
from autodl import AutoDL

# Initialize with default settings
auto_dl = AutoDL(task_type="classification")

# Fit the model
model = auto_dl.fit(X_train, y_train)

# Make predictions
predictions = auto_dl.predict(X_test)
```

### Advanced Configuration

```python
auto_dl = AutoDL(
    task_type="classification",           # "classification" or "regression"
    optimize_hyperparameters=True,        # Enable hyperparameter optimization
    max_trials=10,                        # Number of optimization trials
    test_size=0.2,                        # Validation split size
    random_state=42,                      # Random seed for reproducibility
    verbose=1                             # Verbosity level (0, 1, or 2)
)
```

## Components

### AutoDL
Main interface for automatic model generation. Handles the entire pipeline from data preprocessing to model training.

### ModelGenerator
Generates neural network architectures automatically based on data characteristics. Supports both fully-connected networks and CNNs.

### HyperparameterOptimizer
Optimizes model hyperparameters using random search or grid search strategies.

### DataPreprocessor
Handles data preprocessing including normalization, encoding, and train-test splitting.

## Examples

Check the `examples/` directory for more detailed examples:
- `classification_example.py`: Iris dataset classification
- `regression_example.py`: Synthetic regression problem

Run examples with:
```bash
python examples/classification_example.py
python examples/regression_example.py
```

## Architecture

Auto-DL automatically determines the architecture based on:
- Input data dimensions
- Task type (classification/regression)
- Number of classes (for classification)

The architecture includes:
- Automatic layer size determination
- Dropout for regularization
- Appropriate activation functions
- Task-specific output layers

## Hyperparameter Optimization

Auto-DL searches over:
- Learning rate
- Batch size
- Number of epochs
- Hidden layer sizes
- Dropout rate
- Optimizer type (Adam, SGD, RMSprop)

## Requirements

- Python >= 3.8
- TensorFlow >= 2.13.0
- NumPy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

Built with TensorFlow and inspired by AutoML research.
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

	Optional shared run id for preprocess + train:

	`autodl preprocess --data data/train.csv --target label --config config.yaml --run-id run_20260216T120000Z`

5. Run training on preprocessed data:

	`autodl train --parquet runs/<prep_run_id>/preprocessed.parquet --target label --config config.yaml`

	By default, when parquet path is `runs/*/preprocessed.parquet`, training writes into that same run directory.

	You can also force a specific run directory with:

	`autodl train --parquet runs/<prep_run_id>/preprocessed.parquet --target label --config config.yaml --run-id run_20260216T120000Z`

	Optional metric override:

	`autodl train --parquet runs/<prep_run_id>/preprocessed.parquet --target label --config config.yaml --primary-metric f1_macro`

Or run the full flow in one command:

	`autodl run-full --data data/train.csv --target label --config config.yaml --narrative-file narrative.yaml`

	This writes preprocess + train artifacts into a single run directory: `runs/run_<timestamp>/`.

	Optional metric override in full flow:

	`autodl run-full --data data/train.csv --target label --config config.yaml --primary-metric roc_auc`

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

	Optional narrative-guided planning:

	`autodl preprocess --data data/your_file.csv --target label --config config.yaml --narrative-file narrative.yaml`

	For low-confidence director plans in non-interactive runs, add:

	`--approve-low-confidence`

4. Inspect preprocessing outputs:

	- transformed data: `runs/<prep_run_id>/preprocessed.parquet`
	- metadata summary: `runs/<prep_run_id>/preprocess_metadata.json`
	- director plan (when enabled): `runs/<prep_run_id>/director_plan.json`
	- decision log (when enabled): `runs/<prep_run_id>/decision_log.jsonl`

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
- `runs/<run_id>/preprocess_manifest.json` (preprocess stage)
- `runs/<run_id>/training_manifest.json` (train stage)
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
- `Problem + Model` section (classification/regression + task type + backend/framework)
- `Preprocessing Summary` section (director and preprocessing decisions)
- `Training Progress` section (stages, trials, and retrain progress)
- `Primary Metric Selection` section with objective-aware metric choice rationale (imbalance-aware, optional LLM refinement)
- `Training Policy` section in `REPORT.md` describing selected loss and class-weight strategy
- optional `LLM Summary` section in `REPORT.md` when `llm.provider` is enabled
- confusion matrix and evaluation plots for classification tasks
- threshold analysis files for binary classification (`threshold_metrics.csv`, threshold plot)
- binary evaluation includes calibration metric (`brier_score`) in summary/report
- optimization progress plot (`optuna_progress.png`)
- `training_summary.json` includes `training_policy` for programmatic audit

## Optional dependencies
- LLM providers: `pip install -e .[llm]`
- W&B tracking: `pip install -e .[tracking]`
- Training stack (next step): `pip install -e .[train]`

## LLM director planning docs
- Director MVP design and integration notes: `docs/llm_director_mvp.md`
- Director JSON output schema: `docs/schemas/llm_director_plan.schema.json`
- Prompt template (system + user payload): `docs/prompts/llm_director_prompt.md`
- Unified orchestration graph + node contracts: `docs/workflows/agent_orchestration_graph.md`
