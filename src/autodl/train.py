from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from autodl.config import AppConfig
from autodl.utils import write_json


def stage_sample(df: pd.DataFrame, config: AppConfig, target: str) -> pd.DataFrame:
    frac = config.train.stage1_sample_fraction
    if frac >= 1.0:
        return df
    try:
        if target in df.columns and df[target].nunique(dropna=True) > 1:
            return (
                df.groupby(target, group_keys=False)
                .apply(lambda g: g.sample(frac=frac, random_state=config.random_seed), include_groups=True)
            )
    except Exception:
        pass
    return df.sample(frac=frac, random_state=config.random_seed)


def _import_train_deps() -> tuple[Any, Any]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for training. Install with: pip install -e .[train]") from exc

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required for training. Install with: pip install -e .[train]") from exc

    return tf, optuna


def _to_dense_float32(X: pd.DataFrame) -> np.ndarray:
    dense = X.sparse.to_dense() if hasattr(X, "sparse") else X
    return np.asarray(dense, dtype=np.float32)


def _infer_task(y: pd.Series) -> tuple[str, int]:
    unique = y.nunique(dropna=True)
    if unique <= 2:
        return "binary", int(unique)
    if pd.api.types.is_integer_dtype(y) and unique <= 100:
        return "multiclass", int(unique)
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
        return "multiclass", int(unique)
    return "regression", int(unique)


def _build_model(tf: Any, trial: Any, input_dim: int, task_type: str, n_classes: int) -> Any:
    n_layers = trial.suggest_int("n_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for layer_index in range(n_layers):
        units = trial.suggest_int(f"units_{layer_index}", 32, 512, step=32)
        model.add(tf.keras.layers.Dense(units, activation="relu"))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))

    if task_type == "binary":
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
        metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")]
    elif task_type == "multiclass":
        model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    else:
        model.add(tf.keras.layers.Dense(1, activation="linear"))
        loss = "mse"
        metrics = [tf.keras.metrics.RootMeanSquaredError(name="rmse")]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model


def _score_history(task_type: str, history: Any) -> float:
    hist = history.history
    if task_type == "regression":
        values = hist.get("val_rmse") or hist.get("rmse") or hist.get("val_loss")
        return float(-min(values))
    if task_type == "binary":
        if hist.get("val_auc"):
            return float(max(hist["val_auc"]))
        if hist.get("val_accuracy"):
            return float(max(hist["val_accuracy"]))
        values = hist.get("val_loss") or hist.get("loss")
        return float(-min(values))
    values = hist.get("val_accuracy") or hist.get("val_loss")
    if hist.get("val_accuracy"):
        return float(max(values))
    return float(-min(values))


def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stratify = y if task_type in {"binary", "multiclass"} else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=random_seed,
        stratify=stratify,
    )
    val_relative = validation_fraction / max(1.0 - test_fraction, 1e-6)
    stratify_train = y_train_val if task_type in {"binary", "multiclass"} else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative,
        random_state=random_seed,
        stratify=stratify_train,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_with_optuna(
    parquet_path: Path,
    target: str,
    run_dir: Path,
    config: AppConfig,
) -> dict[str, Any]:
    tf, optuna = _import_train_deps()
    tf.keras.utils.set_random_seed(config.random_seed)

    df = pd.read_parquet(parquet_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in parquet.")

    y_series = df[target]
    X_df = df.drop(columns=[target])

    if X_df.shape[1] == 0:
        raise ValueError("No feature columns found after removing target.")

    if X_df.shape[1] > 20000:
        raise ValueError("Feature dimensionality is very high for dense TensorFlow baseline (>20k). Reduce features before training.")

    task_type, n_classes = _infer_task(y_series)
    if task_type in {"binary", "multiclass"} and not pd.api.types.is_numeric_dtype(y_series):
        y_codes = pd.Categorical(y_series).codes
        y = np.asarray(y_codes, dtype=np.int32)
    elif task_type in {"binary", "multiclass"}:
        y = np.asarray(y_series, dtype=np.int32)
    else:
        y = np.asarray(y_series, dtype=np.float32)

    sampled_df = stage_sample(df, config, target=target)
    sampled_X = _to_dense_float32(sampled_df.drop(columns=[target]))
    sampled_y_series = sampled_df[target]
    if task_type in {"binary", "multiclass"} and not pd.api.types.is_numeric_dtype(sampled_y_series):
        sampled_y = np.asarray(pd.Categorical(sampled_y_series).codes, dtype=np.int32)
    elif task_type in {"binary", "multiclass"}:
        sampled_y = np.asarray(sampled_y_series, dtype=np.int32)
    else:
        sampled_y = np.asarray(sampled_y_series, dtype=np.float32)

    X_train, X_val, _X_test, y_train, y_val, _y_test = _split_data(
        sampled_X,
        sampled_y,
        task_type=task_type,
        validation_fraction=config.train.validation_fraction,
        test_fraction=config.train.test_fraction,
        random_seed=config.random_seed,
    )

    def objective(trial: Any) -> float:
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        model = _build_model(
            tf=tf,
            trial=trial,
            input_dim=X_train.shape[1],
            task_type=task_type,
            n_classes=max(n_classes, 2),
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.train.patience,
                restore_best_weights=True,
            )
        ]
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=config.train.epochs_tuning,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        return _score_history(task_type, history)

    study = optuna.create_study(direction=config.train.direction)
    study.optimize(
        objective,
        n_trials=config.train.max_trials,
        timeout=config.train.timeout_minutes * 60,
    )

    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    trials_path = run_dir / "optuna_trials.parquet"
    trials_df.to_parquet(trials_path, index=False)

    full_X = _to_dense_float32(X_df)
    full_y = y

    X_train_f, X_val_f, X_test_f, y_train_f, y_val_f, y_test_f = _split_data(
        full_X,
        full_y,
        task_type=task_type,
        validation_fraction=config.train.validation_fraction,
        test_fraction=config.train.test_fraction,
        random_seed=config.random_seed,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ranked = sorted(completed, key=lambda t: t.value, reverse=(config.train.direction == "maximize"))
    finalists = ranked[: config.train.top_k_final]
    if not finalists:
        raise RuntimeError("No completed Optuna trials were available for final training.")

    best_final_score = None
    best_trial_number = None
    best_model = None
    final_scores: list[dict[str, Any]] = []

    for trial in finalists:
        fixed = optuna.trial.FixedTrial(trial.params)
        model = _build_model(
            tf=tf,
            trial=fixed,
            input_dim=X_train_f.shape[1],
            task_type=task_type,
            n_classes=max(n_classes, 2),
        )
        history = model.fit(
            X_train_f,
            y_train_f,
            validation_data=(X_val_f, y_val_f),
            epochs=config.train.epochs_final,
            batch_size=int(trial.params.get("batch_size", 32)),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=config.train.patience,
                    restore_best_weights=True,
                )
            ],
        )
        score = _score_history(task_type, history)
        final_scores.append({"trial_number": int(trial.number), "score": float(score), "params": trial.params})
        if best_final_score is None or (
            config.train.direction == "maximize" and score > best_final_score
        ) or (
            config.train.direction == "minimize" and score < best_final_score
        ):
            best_final_score = float(score)
            best_trial_number = int(trial.number)
            best_model = model

    if best_model is None:
        raise RuntimeError("Failed to select best final model.")

    eval_result = best_model.evaluate(X_test_f, y_test_f, verbose=0)
    metric_names = best_model.metrics_names
    eval_metrics = {name: float(value) for name, value in zip(metric_names, eval_result)}

    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model.export(str(model_dir))

    summary = {
        "status": "ok",
        "message": "Training completed with TensorFlow + Optuna",
        "target": target,
        "task_type": task_type,
        "full_rows": int(df.shape[0]),
        "sample_rows": int(sampled_df.shape[0]),
        "n_features": int(X_df.shape[1]),
        "optuna_trials": int(len(study.trials)),
        "best_stage1_trial": int(study.best_trial.number),
        "best_stage1_value": float(study.best_value),
        "best_final_trial": best_trial_number,
        "best_final_score": best_final_score,
        "final_candidates": final_scores,
        "test_metrics": eval_metrics,
        "model_dir": str(model_dir),
        "trials_path": str(trials_path),
    }

    write_json(run_dir / "training_summary.json", summary)
    (run_dir / "best_params.json").write_text(json.dumps(study.best_trial.params, indent=2), encoding="utf-8")
    return summary
