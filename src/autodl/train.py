from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from autodl.config import AppConfig
from autodl.utils import write_json


def stage_sample(df: pd.DataFrame, config: AppConfig, target: str) -> pd.DataFrame:
    frac = config.sample_fraction_tuning
    if frac >= 1.0:
        return df
    try:
        if target in df.columns and df[target].nunique(dropna=True) > 1:
            return (
                df.groupby(target, group_keys=False)
                .apply(lambda g: g.sample(frac=frac, random_state=config.random_seed), include_groups=True)
                .reset_index(drop=True)
            )
    except Exception:
        pass
    return df.sample(frac=frac, random_state=config.random_seed).reset_index(drop=True)


def train_placeholder(
    parquet_path: Path,
    target: str,
    run_dir: Path,
    config: AppConfig,
) -> dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    sampled = stage_sample(df, config, target=target)

    summary = {
        "status": "scaffold_only",
        "message": "Training scaffold created. Plug in TensorFlow+Optuna pipeline next.",
        "full_rows": int(df.shape[0]),
        "sample_rows": int(sampled.shape[0]),
        "n_features": int(max(sampled.shape[1] - 1, 0)),
        "target": target,
    }

    write_json(run_dir / "training_summary.json", summary)
    return summary
