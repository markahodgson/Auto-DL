from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from autodl.config import AppConfig
from autodl.llm.factory import make_llm_provider
from autodl.policy.metric_director import decide_primary_metric
from autodl.policy.train_director import TrainingPolicyDecision, build_training_policy
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


def _build_model(
    tf: Any,
    trial: Any,
    input_dim: int,
    task_type: str,
    n_classes: int,
    training_policy: TrainingPolicyDecision,
) -> Any:
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
        if training_policy.loss == "binary_focal_crossentropy":
            loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)
        else:
            loss = "binary_crossentropy"
        metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")]
    elif task_type == "multiclass":
        model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
        loss = training_policy.loss
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    else:
        model.add(tf.keras.layers.Dense(1, activation="linear"))
        loss = training_policy.loss
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


def _fit_class_weight(training_policy: TrainingPolicyDecision, y_train: np.ndarray) -> dict[int, float] | None:
    if not training_policy.use_class_weights:
        return None
    if not training_policy.class_weight_by_label:
        return None
    present = {int(v) for v in np.unique(y_train).tolist()}
    out: dict[int, float] = {}
    for label_str, weight in training_policy.class_weight_by_label.items():
        try:
            label = int(label_str)
        except ValueError:
            continue
        if label in present:
            out[label] = float(weight)
    return out or None


def _compute_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _best_threshold_by_f1(y_val: np.ndarray, val_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.05, 0.95, 37)
    rows = []
    for threshold in thresholds:
        metrics = _compute_threshold_metrics(y_val, val_prob, float(threshold))
        rows.append({"threshold": float(threshold), **metrics})
    frame = pd.DataFrame(rows)
    best_row = frame.sort_values("f1", ascending=False).iloc[0]
    return float(best_row["threshold"]), frame


def _try_import_matplotlib() -> tuple[Any, Any] | tuple[None, None]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None
    return plt, np


def _save_optuna_progress_plot(trials_df: pd.DataFrame, run_dir: Path) -> Path | None:
    if trials_df.empty or "value" not in trials_df.columns:
        return None
    plt, _ = _try_import_matplotlib()
    if plt is None:
        return None

    values = trials_df["value"].astype(float).to_numpy()
    best = np.maximum.accumulate(values)
    out = run_dir / "optuna_progress.png"

    plt.figure(figsize=(8, 4))
    plt.plot(values, label="trial value", alpha=0.7)
    plt.plot(best, label="best so far", linewidth=2)
    plt.xlabel("Trial")
    plt.ylabel("Objective")
    plt.title("Optuna Optimization Progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    return out


def _plot_confusion_matrix(cm: np.ndarray, out: Path, title: str) -> Path | None:
    plt, _ = _try_import_matplotlib()
    if plt is None:
        return None

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    return out


def _evaluate_and_plot(
    task_type: str,
    y_val: np.ndarray,
    y_test: np.ndarray,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
    run_dir: Path,
    make_plots: bool,
) -> tuple[dict[str, Any], list[str]]:
    plot_paths: list[str] = []

    if task_type == "binary":
        val_prob = pred_val.reshape(-1)
        test_prob = pred_test.reshape(-1)
        best_threshold, threshold_frame = _best_threshold_by_f1(y_val, val_prob)
        threshold_path = run_dir / "threshold_metrics.csv"
        threshold_frame.to_csv(threshold_path, index=False)

        test_metrics = _compute_threshold_metrics(y_test, test_prob, best_threshold)
        test_metrics["roc_auc"] = float(roc_auc_score(y_test, test_prob))
        test_metrics["brier_score"] = float(brier_score_loss(y_test, test_prob))

        y_pred_test = (test_prob >= best_threshold).astype(np.int32)
        cm = confusion_matrix(y_test, y_pred_test)

        if make_plots:
            cm_plot = _plot_confusion_matrix(cm, run_dir / "confusion_matrix.png", "Binary Confusion Matrix")
            if cm_plot:
                plot_paths.append(str(cm_plot))

            plt, _ = _try_import_matplotlib()
            if plt is not None:
                curve_out = run_dir / "threshold_curve.png"
                plt.figure(figsize=(8, 4))
                plt.plot(threshold_frame["threshold"], threshold_frame["precision"], label="precision")
                plt.plot(threshold_frame["threshold"], threshold_frame["recall"], label="recall")
                plt.plot(threshold_frame["threshold"], threshold_frame["f1"], label="f1")
                plt.axvline(best_threshold, linestyle="--", color="black", alpha=0.6, label="best threshold")
                plt.xlabel("Threshold")
                plt.ylabel("Score")
                plt.title("Threshold Analysis (Validation)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(curve_out, dpi=140)
                plt.close()
                plot_paths.append(str(curve_out))

                fpr, tpr, _ = roc_curve(y_test, test_prob)
                roc_out = run_dir / "roc_curve.png"
                plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, label=f"AUC = {test_metrics['roc_auc']:.4f}")
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                plt.tight_layout()
                plt.savefig(roc_out, dpi=140)
                plt.close()
                plot_paths.append(str(roc_out))

        return (
            {
                "task_type": "binary",
                "selected_threshold": float(best_threshold),
                "test_metrics": test_metrics,
                "confusion_matrix": cm.tolist(),
                "threshold_metrics_path": str(threshold_path),
            },
            plot_paths,
        )

    if task_type == "multiclass":
        test_pred = np.argmax(pred_test, axis=1).astype(np.int32)
        metrics = {
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "precision_macro": float(precision_score(y_test, test_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, test_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
        }
        cm = confusion_matrix(y_test, test_pred)
        if make_plots:
            cm_plot = _plot_confusion_matrix(cm, run_dir / "confusion_matrix.png", "Multiclass Confusion Matrix")
            if cm_plot:
                plot_paths.append(str(cm_plot))
        return (
            {
                "task_type": "multiclass",
                "test_metrics": metrics,
                "confusion_matrix": cm.tolist(),
            },
            plot_paths,
        )

    y_pred = pred_test.reshape(-1)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    if make_plots:
        plt, _ = _try_import_matplotlib()
        if plt is not None:
            scatter = run_dir / "prediction_scatter.png"
            plt.figure(figsize=(6, 6))
            plt.scatter(y_test, y_pred, alpha=0.7)
            min_v = min(float(np.min(y_test)), float(np.min(y_pred)))
            max_v = max(float(np.max(y_test)), float(np.max(y_pred)))
            plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="gray")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Regression: Actual vs Predicted")
            plt.tight_layout()
            plt.savefig(scatter, dpi=140)
            plt.close()
            plot_paths.append(str(scatter))

    return ({"task_type": "regression", "test_metrics": metrics}, plot_paths)


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_preprocess_summary(parquet_path: Path) -> dict[str, Any]:
    prep_dir = parquet_path.parent
    summary: dict[str, Any] = {
        "available": False,
        "source_run_dir": str(prep_dir),
    }

    preprocess_metadata = _safe_read_json(prep_dir / "preprocess_metadata.json") or {}
    preprocess_manifest = _safe_read_json(prep_dir / "run_manifest.json") or {}
    director_plan = _safe_read_json(prep_dir / "director_plan.json") or {}

    if preprocess_metadata or preprocess_manifest:
        summary["available"] = True

    preprocess_config = preprocess_manifest.get("config", {}).get("preprocess", {})
    director_meta = preprocess_metadata.get("director", {})
    columns = director_plan.get("columns", []) if isinstance(director_plan, dict) else []

    dropped_columns = 0
    mapped_columns = 0
    for col in columns:
        actions = col.get("actions", []) if isinstance(col, dict) else []
        if "drop" in actions:
            dropped_columns += 1
        if "binary_map" in actions or "ordinal_map" in actions:
            mapped_columns += 1

    summary.update(
        {
            "text_backend": preprocess_metadata.get("text_backend") or preprocess_config.get("text_backend"),
            "normalize_numeric": preprocess_config.get("normalize_numeric"),
            "one_hot_categorical": preprocess_config.get("one_hot_categorical"),
            "director_source": director_meta.get("source"),
            "director_confidence": director_meta.get("confidence"),
            "review_questions": director_meta.get("review_questions", []),
            "narrative_business_goal": (director_meta.get("narrative_applied") or {}).get("business_goal", ""),
            "dropped_columns": dropped_columns,
            "mapped_columns": mapped_columns,
            "narrative_present": bool((director_meta.get("narrative_applied") or {}).get("dataset_summary")),
        }
    )
    return summary


def _build_data_characteristics(X_df: pd.DataFrame, y: np.ndarray, task_type: str) -> dict[str, Any]:
    n_rows = int(X_df.shape[0])
    unique_ratios: list[float] = []
    low_variance_cols = 0
    for col in X_df.columns:
        series = X_df[col]
        unique_ratio = float(series.nunique(dropna=True) / max(n_rows, 1))
        unique_ratios.append(unique_ratio)
        if unique_ratio <= 0.01:
            low_variance_cols += 1

    mean_unique_ratio = float(np.mean(unique_ratios)) if unique_ratios else 0.0

    result: dict[str, Any] = {
        "n_rows": n_rows,
        "n_features": int(X_df.shape[1]),
        "mean_feature_unique_ratio": mean_unique_ratio,
        "low_variance_columns": int(low_variance_cols),
    }

    if task_type == "binary":
        values, counts = np.unique(y, return_counts=True)
        class_counts = {str(int(v)): int(c) for v, c in zip(values, counts, strict=False)}
        total = float(sum(class_counts.values()))
        minority = min(class_counts.values()) if class_counts else 0
        majority = max(class_counts.values()) if class_counts else 1
        minority_rate = float(minority / max(total, 1.0))
        imbalance_ratio = float(minority / max(majority, 1))
        positive_rate = float(class_counts.get("1", 0) / max(total, 1.0))
        result.update(
            {
                "class_counts": class_counts,
                "positive_rate": positive_rate,
                "minority_rate": minority_rate,
                "imbalance_ratio": imbalance_ratio,
                "rare_outcome": bool(minority_rate < 0.2),
            }
        )
    elif task_type == "multiclass":
        values, counts = np.unique(y, return_counts=True)
        class_counts = {str(int(v)): int(c) for v, c in zip(values, counts, strict=False)}
        probs = np.asarray(counts, dtype=np.float64) / max(np.sum(counts), 1.0)
        entropy = float(-np.sum(np.where(probs > 0, probs * np.log2(probs), 0.0)))
        max_entropy = math.log2(max(len(counts), 1)) if len(counts) > 1 else 1.0
        normalized_entropy = float(entropy / max(max_entropy, 1e-6))
        result.update(
            {
                "class_counts": class_counts,
                "class_entropy_normalized": normalized_entropy,
                "class_diversity": "high" if normalized_entropy >= 0.8 else "medium" if normalized_entropy >= 0.5 else "low",
            }
        )

    return result


def _score_band(metric: str, value: float) -> str:
    if metric == "roc_auc":
        if value >= 0.9:
            return "excellent"
        if value >= 0.8:
            return "good"
        if value >= 0.7:
            return "fair"
        return "weak"
    if metric in {"f1", "f1_macro", "accuracy", "precision", "recall", "precision_macro", "recall_macro"}:
        if value >= 0.85:
            return "good"
        if value >= 0.7:
            return "fair"
        return "weak"
    if metric == "brier_score":
        if value <= 0.12:
            return "good"
        if value <= 0.2:
            return "fair"
        return "weak"
    if metric == "rmse":
        return "context-dependent"
    if metric == "r2":
        if value >= 0.8:
            return "good"
        if value >= 0.5:
            return "fair"
        return "weak"
    return "context-dependent"


def _deterministic_metric_interpretation(
    task_type: str,
    metrics: dict[str, float],
    primary_metric: dict[str, Any],
    data_characteristics: dict[str, Any],
) -> dict[str, Any]:
    metric_name = str(primary_metric.get("metric") or "")
    metric_value = metrics.get(metric_name)
    band = _score_band(metric_name, float(metric_value)) if metric_value is not None else "unknown"

    bullets: list[str] = []
    bullets.append(f"- Primary metric `{metric_name}` is `{band}` for a `{task_type}` task.")
    if metric_value is not None:
        bullets.append(f"- Observed `{metric_name}` value: `{float(metric_value):.6f}`.")

    if task_type == "binary":
        rare_outcome = bool(data_characteristics.get("rare_outcome", False))
        imbalance_ratio = data_characteristics.get("imbalance_ratio")
        positive_rate = data_characteristics.get("positive_rate")
        bullets.append(
            f"- Binary outcome profile: rare_outcome=`{rare_outcome}`, positive_rate=`{float(positive_rate):.3f}`"
            + (f", imbalance_ratio=`{float(imbalance_ratio):.3f}`." if imbalance_ratio is not None else ".")
        )
        if rare_outcome and "roc_auc" in metrics and metric_name != "roc_auc":
            bullets.append("- For rare outcomes, consider using `roc_auc` as primary metric to reduce threshold sensitivity.")
        if "brier_score" in metrics:
            brier_band = _score_band("brier_score", float(metrics["brier_score"]))
            bullets.append(f"- Calibration check via `brier_score` is `{brier_band}` (`{float(metrics['brier_score']):.6f}`).")
    elif task_type == "multiclass":
        diversity = data_characteristics.get("class_diversity", "unknown")
        bullets.append(f"- Class diversity is `{diversity}` based on normalized class entropy.")
    else:
        bullets.append("- Regression metric quality may depend on target scale and domain tolerance.")

    mean_unique_ratio = float(data_characteristics.get("mean_feature_unique_ratio", 0.0))
    low_var = int(data_characteristics.get("low_variance_columns", 0))
    bullets.append(
        f"- Feature diversity snapshot: mean_unique_ratio=`{mean_unique_ratio:.3f}`, low_variance_columns=`{low_var}`."
    )

    overall = "good" if band in {"good", "excellent"} else "fair" if band == "fair" else "needs_attention"
    return {
        "source": "deterministic",
        "overall_assessment": overall,
        "text": "\n".join(bullets),
    }


def _maybe_llm_metric_interpretation(config: AppConfig, payload: dict[str, Any]) -> dict[str, Any] | None:
    if config.llm.provider == "none":
        return None
    try:
        provider = make_llm_provider(config.llm)
        system_prompt = (
            "You are an ML evaluation analyst. Return concise markdown bullets only. "
            "Assess whether metrics are good for the problem type, using rarity/imbalance and diversity context."
        )
        user_prompt = json.dumps(payload)
        text = provider.suggest(system_prompt, user_prompt)
        if not text:
            return None
        return {
            "source": "llm",
            "overall_assessment": payload.get("deterministic", {}).get("overall_assessment", "unknown"),
            "text": text.strip(),
        }
    except Exception:
        return None


def _maybe_llm_summary(config: AppConfig, summary_payload: dict[str, Any]) -> str | None:
    if config.llm.provider == "none":
        return None
    try:
        provider = make_llm_provider(config.llm)
        system_prompt = "You are an ML experiment analyst. Return concise markdown only."
        user_prompt = json.dumps(
            {
                "request": "Write a concise summary with 3 parts: (1) problem + model type (binary/multiclass/regression and backend/framework), (2) preprocessing decisions, and (3) training progress.",
                "summary": summary_payload,
            }
        )
        text = provider.suggest(system_prompt, user_prompt)
        return text.strip() if text else None
    except Exception:
        return None


def _write_markdown_report(
    run_dir: Path,
    summary: dict[str, Any],
    best_params: dict[str, Any],
    evaluation: dict[str, Any],
    plot_paths: list[str],
    llm_summary: str | None,
    training_policy: dict[str, Any] | None = None,
) -> Path:
    lines: list[str] = []
    lines.append("# Training Report")
    lines.append("")

    evaluation_metrics = evaluation.get("test_metrics", {})
    primary_metric_decision = summary.get("primary_metric") or {}
    primary_metric = str(primary_metric_decision.get("metric") or "")
    if not primary_metric and evaluation_metrics:
        primary_metric = str(next(iter(evaluation_metrics.keys())))
    primary_value = evaluation_metrics.get(primary_metric)

    preprocessing_summary = summary.get("preprocessing_summary") or {}
    director_confidence = preprocessing_summary.get("director_confidence")

    lines.append("## Executive Summary")
    lines.append(f"- Problem/Task: {summary.get('problem_type', 'unknown')} / {summary.get('task_type', 'unknown')}")
    lines.append(f"- Model stack: {summary.get('model_backend', 'tensorflow')} + {summary.get('search_engine', 'optuna')}")
    if primary_value is not None:
        lines.append(f"- Primary result ({primary_metric}): {float(primary_value):.6f}")
    lines.append(f"- Best final trial/score: {summary['best_final_trial']} / {summary['best_final_score']:.6f}")
    if director_confidence is not None:
        lines.append(f"- Preprocess director confidence: {float(director_confidence):.4f}")
    lines.append("")

    lines.append("## Overview")
    lines.append(f"- Status: {summary['status']}")
    lines.append(f"- Target: {summary['target']}")
    lines.append(f"- Task Type: {summary['task_type']}")
    lines.append(f"- Rows (full/sample): {summary['full_rows']} / {summary['sample_rows']}")
    lines.append(f"- Features: {summary['n_features']}")
    lines.append(f"- Optuna trials: {summary['optuna_trials']}")
    lines.append(f"- Best stage-1 trial/value: {summary['best_stage1_trial']} / {summary['best_stage1_value']:.6f}")
    lines.append(f"- Best final trial/score: {summary['best_final_trial']} / {summary['best_final_score']:.6f}")
    lines.append("")

    lines.append("## Problem + Model")
    lines.append(f"- Problem type: {summary.get('problem_type', 'unknown')}")
    lines.append(f"- Task type: {summary.get('task_type', 'unknown')}")
    lines.append(f"- Model backend: {summary.get('model_backend', 'tensorflow')}")
    lines.append(f"- Model family: {summary.get('model_family', 'dense_neural_network')}")
    lines.append(f"- Search engine: {summary.get('search_engine', 'optuna')}")
    lines.append("")

    lines.append("## Preprocessing Summary")
    if preprocessing_summary.get("available"):
        lines.append(f"- Source run: {preprocessing_summary.get('source_run_dir', '')}")
        lines.append(f"- Text backend: {preprocessing_summary.get('text_backend', 'unknown')}")
        lines.append(f"- Normalize numeric: {preprocessing_summary.get('normalize_numeric', 'unknown')}")
        lines.append(f"- One-hot categorical: {preprocessing_summary.get('one_hot_categorical', 'unknown')}")
        lines.append(f"- Director source: {preprocessing_summary.get('director_source', 'n/a')}")
        confidence = preprocessing_summary.get("director_confidence")
        if confidence is not None:
            lines.append(f"- Director confidence: {float(confidence):.4f}")
        lines.append(f"- Mapped columns: {preprocessing_summary.get('mapped_columns', 0)}")
        lines.append(f"- Dropped columns: {preprocessing_summary.get('dropped_columns', 0)}")
        lines.append(f"- Narrative provided: {preprocessing_summary.get('narrative_present', False)}")
    else:
        lines.append("- Preprocess metadata unavailable for this parquet input.")
    lines.append("")

    progress = summary.get("training_progress") or {}
    lines.append("## Training Progress")
    lines.append(f"- Stage-1 sample fraction: {progress.get('stage1_sample_fraction', 'n/a')}")
    lines.append(f"- Stage-1 epochs: {progress.get('epochs_tuning', 'n/a')}")
    lines.append(f"- Final epochs: {progress.get('epochs_final', 'n/a')}")
    lines.append(f"- Early stopping patience: {progress.get('patience', 'n/a')}")
    lines.append(f"- Trials completed: {progress.get('completed_trials', 'n/a')} / {progress.get('total_trials', 'n/a')}")
    lines.append(f"- Finalists retrained: {progress.get('finalists_retrained', 'n/a')}")
    lines.append("")

    if primary_metric_decision:
        lines.append("## Primary Metric Selection")
        lines.append(f"- Selected metric: {primary_metric_decision.get('metric', 'n/a')}")
        lines.append(f"- Source: {primary_metric_decision.get('source', 'unknown')}")
        lines.append(f"- Confidence: {float(primary_metric_decision.get('confidence', 0.0)):.4f}")
        lines.append(f"- Rationale: {primary_metric_decision.get('rationale', '')}")
        lines.append("")

    lines.append("## Best Hyperparameters")
    for key, value in best_params.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Evaluation")
    for key, value in evaluation.get("test_metrics", {}).items():
        lines.append(f"- {key}: {value:.6f}")

    if evaluation.get("task_type") == "binary":
        lines.append(f"- selected_threshold: {evaluation.get('selected_threshold', 0.5):.4f}")

    cm = evaluation.get("confusion_matrix")
    if cm:
        lines.append("")
        lines.append("### Confusion Matrix")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(cm, indent=2))
        lines.append("```")

    if llm_summary:
        lines.append("")
        lines.append("## LLM Summary")
        lines.append("")
        lines.append(llm_summary)

    if training_policy:
        lines.append("")
        lines.append("## Training Policy")
        lines.append(f"- Source: {training_policy.get('source', 'unknown')}")
        lines.append(f"- Loss: {training_policy.get('loss', 'n/a')}")
        lines.append(f"- Use class weights: {training_policy.get('use_class_weights', False)}")
        lines.append(f"- Confidence: {training_policy.get('confidence', 0.0):.4f}")
        lines.append(f"- Rationale: {training_policy.get('rationale', '')}")
        class_weights = training_policy.get("class_weight_by_label") or {}
        if class_weights:
            lines.append("- Class weights:")
            for class_label, weight in class_weights.items():
                lines.append(f"  - {class_label}: {weight:.6f}")

    metric_interpretation = summary.get("metric_interpretation") or {}
    if metric_interpretation:
        lines.append("")
        lines.append("## Metric Interpretation")
        lines.append(f"- Source: {metric_interpretation.get('source', 'unknown')}")
        lines.append(f"- Overall assessment: {metric_interpretation.get('overall_assessment', 'unknown')}")
        lines.append("")
        lines.append(metric_interpretation.get("text", ""))

    if plot_paths:
        lines.append("")
        lines.append("## Plots")
        lines.append("")
        for path in plot_paths:
            image_name = Path(path).name
            lines.append(f"### {image_name}")
            lines.append("")
            lines.append(f"![{image_name}]({image_name})")
            lines.append("")

    report_path = run_dir / "REPORT.md"
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return report_path


def train_with_optuna(
    parquet_path: Path,
    target: str,
    run_dir: Path,
    config: AppConfig,
    primary_metric_override: str | None = None,
) -> dict[str, Any]:
    tf, optuna = _import_train_deps()
    tf.keras.utils.set_random_seed(config.random_seed)
    preprocess_summary = _load_preprocess_summary(parquet_path)

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
    classes: list[Any] | None = None
    if task_type in {"binary", "multiclass"} and not pd.api.types.is_numeric_dtype(y_series):
        cat = pd.Categorical(y_series)
        classes = cat.categories.tolist()
        y = np.asarray(cat.codes, dtype=np.int32)
    elif task_type in {"binary", "multiclass"}:
        y = np.asarray(y_series, dtype=np.int32)
        classes = sorted(pd.Series(y).dropna().unique().tolist())
    else:
        y = np.asarray(y_series, dtype=np.float32)

    training_policy = build_training_policy(
        y_encoded=np.asarray(y),
        task_type=task_type,
        n_classes=max(n_classes, 1),
        config=config,
    )

    sampled_df = stage_sample(df, config, target=target)
    data_characteristics = _build_data_characteristics(X_df=X_df, y=np.asarray(y), task_type=task_type)
    sampled_X = _to_dense_float32(sampled_df.drop(columns=[target]))
    if task_type in {"binary", "multiclass"}:
        sampled_y = y[sampled_df.index.to_numpy()]
    else:
        sampled_y = np.asarray(sampled_df[target], dtype=np.float32)

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
            training_policy=training_policy,
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
            class_weight=_fit_class_weight(training_policy, y_train),
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
            training_policy=training_policy,
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
            class_weight=_fit_class_weight(training_policy, y_train_f),
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

    pred_val = best_model.predict(X_val_f, verbose=0)
    pred_test = best_model.predict(X_test_f, verbose=0)

    evaluation, evaluation_plot_paths = _evaluate_and_plot(
        task_type=task_type,
        y_val=np.asarray(y_val_f),
        y_test=np.asarray(y_test_f),
        pred_val=np.asarray(pred_val),
        pred_test=np.asarray(pred_test),
        run_dir=run_dir,
        make_plots=config.train.generate_plots,
    )

    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model.export(str(model_dir))

    optuna_plot = _save_optuna_progress_plot(trials_df, run_dir) if config.train.generate_plots else None
    plot_paths = list(evaluation_plot_paths)
    if optuna_plot is not None:
        plot_paths.append(str(optuna_plot))

    summary = {
        "status": "ok",
        "message": "Training completed with TensorFlow + Optuna",
        "target": target,
        "problem_type": "classification" if task_type in {"binary", "multiclass"} else "regression",
        "task_type": task_type,
        "model_backend": "tensorflow",
        "model_family": "dense_neural_network",
        "search_engine": "optuna",
        "full_rows": int(df.shape[0]),
        "sample_rows": int(sampled_df.shape[0]),
        "n_features": int(X_df.shape[1]),
        "optuna_trials": int(len(study.trials)),
        "best_stage1_trial": int(study.best_trial.number),
        "best_stage1_value": float(study.best_value),
        "best_final_trial": best_trial_number,
        "best_final_score": best_final_score,
        "final_candidates": final_scores,
        "test_metrics": evaluation.get("test_metrics", {}),
        "evaluation": evaluation,
        "class_labels": classes,
        "training_policy": training_policy.model_dump(mode="json"),
        "preprocessing_summary": preprocess_summary,
        "training_progress": {
            "stage1_sample_fraction": config.train.stage1_sample_fraction,
            "epochs_tuning": config.train.epochs_tuning,
            "epochs_final": config.train.epochs_final,
            "patience": config.train.patience,
            "completed_trials": int(len(completed)),
            "total_trials": int(len(study.trials)),
            "finalists_retrained": int(len(finalists)),
        },
        "plots": plot_paths,
        "model_dir": str(model_dir),
        "trials_path": str(trials_path),
    }

    primary_metric = decide_primary_metric(
        task_type=task_type,
        evaluation_metrics=evaluation.get("test_metrics", {}),
        y_encoded=np.asarray(y),
        preprocess_summary=preprocess_summary,
        config=config,
    )

    if primary_metric_override:
        available_metrics = evaluation.get("test_metrics", {})
        if primary_metric_override not in available_metrics:
            raise ValueError(
                f"Requested primary metric override '{primary_metric_override}' is unavailable. "
                f"Available metrics: {', '.join(available_metrics.keys())}"
            )
        primary_metric = primary_metric.model_copy(
            update={
                "metric": primary_metric_override,
                "confidence": 1.0,
                "rationale": "User override via CLI option.",
                "source": "user_override",
            }
        )

    summary["primary_metric"] = primary_metric.model_dump(mode="json")

    deterministic_interp = _deterministic_metric_interpretation(
        task_type=task_type,
        metrics=evaluation.get("test_metrics", {}),
        primary_metric=summary["primary_metric"],
        data_characteristics=data_characteristics,
    )
    llm_interp = _maybe_llm_metric_interpretation(
        config=config,
        payload={
            "problem_type": summary.get("problem_type"),
            "task_type": task_type,
            "primary_metric": summary.get("primary_metric"),
            "metrics": evaluation.get("test_metrics", {}),
            "data_characteristics": data_characteristics,
            "deterministic": deterministic_interp,
        },
    )
    summary["metric_interpretation"] = llm_interp or deterministic_interp

    write_json(run_dir / "training_summary.json", summary)
    best_params = study.best_trial.params
    (run_dir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    if config.train.generate_report:
        llm_summary = _maybe_llm_summary(config, summary)
        report_path = _write_markdown_report(
            run_dir=run_dir,
            summary=summary,
            best_params=best_params,
            evaluation=evaluation,
            plot_paths=plot_paths,
            llm_summary=llm_summary,
            training_policy=summary.get("training_policy"),
        )
        summary["report_path"] = str(report_path)
        write_json(run_dir / "training_summary.json", summary)

    return summary
