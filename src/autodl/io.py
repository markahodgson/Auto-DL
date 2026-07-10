from __future__ import annotations

from pathlib import Path
import pandas as pd


SUPPORTED_INPUTS = {".csv"}


def read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_INPUTS:
        raise ValueError(f"Unsupported input type: {suffix}. Use CSV for now.")
    return pd.read_csv(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
