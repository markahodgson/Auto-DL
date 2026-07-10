from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ProfileResult:
    numeric_cols: list[str]
    categorical_cols: list[str]
    text_cols: list[str]
    passthrough_cols: list[str]
    missing_rates: dict[str, float]
    skew: dict[str, float]
    sparse_ratio_estimate: float


def _is_text_series(series: pd.Series) -> bool:
    if series.dtype.name not in {"object", "string"}:
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.astype(str).head(2000)
    avg_len = sample.str.len().mean()
    unique_ratio = sample.nunique(dropna=True) / max(len(sample), 1)
    return bool(avg_len >= 20 or unique_ratio > 0.7)


def profile_dataframe(df: pd.DataFrame, target: str, passthrough_cols: list[str] | None = None) -> ProfileResult:
    passthrough = set(passthrough_cols or [])
    feature_cols = [c for c in df.columns if c != target]

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and c not in passthrough]
    object_like = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c]) and c not in passthrough]
    text_cols = [c for c in object_like if _is_text_series(df[c])]
    categorical_cols = [c for c in object_like if c not in text_cols]

    missing_rates = {c: float(df[c].isna().mean()) for c in feature_cols}
    skew = {
        c: float(df[c].dropna().skew())
        for c in numeric_cols
        if df[c].dropna().shape[0] > 2
    }

    non_zero_counts = 0
    total_cells = len(df) * max(len(feature_cols), 1)
    for c in feature_cols:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            non_zero_counts += int((col.fillna(0) != 0).sum())
        else:
            non_zero_counts += int(col.notna().sum())
    sparse_ratio_estimate = 1.0 - (non_zero_counts / max(total_cells, 1))

    return ProfileResult(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        text_cols=text_cols,
        passthrough_cols=list(passthrough),
        missing_rates=missing_rates,
        skew=skew,
        sparse_ratio_estimate=float(np.clip(sparse_ratio_estimate, 0.0, 1.0)),
    )
