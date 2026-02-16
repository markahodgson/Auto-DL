from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: Literal["none", "ollama", "openai"] = "none"
    model: str = "llama3.1"
    base_url: str | None = "http://localhost:11434"
    api_key_env: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=60, ge=5)


class TrackingConfig(BaseModel):
    backend: Literal["local", "wandb"] = "local"
    enabled: bool = True
    project: str = "dnn-automation"
    entity: str | None = None
    mode: Literal["offline", "online"] = "offline"


class PreprocessConfig(BaseModel):
    normalize_numeric: bool = True
    one_hot_categorical: bool = True
    text_backend: Literal["none", "hashing", "tfidf", "tf_text_vectorization"] = "hashing"
    passthrough_columns: list[str] = Field(default_factory=list)
    max_text_vocab: int = 50000
    text_hash_features: int = 2**18


class PolicyConfig(BaseModel):
    llm_fallback_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    user_confirmation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class DataConfig(BaseModel):
    input_format: Literal["csv"] = "csv"
    parquet_intermediate: bool = True


class TrainConfig(BaseModel):
    enabled: bool = True
    max_trials: int = Field(default=20, ge=1)
    timeout_minutes: int = Field(default=30, ge=1)
    stage1_sample_fraction: float = Field(default=0.2, ge=0.01, le=1.0)
    validation_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    test_fraction: float = Field(default=0.1, ge=0.05, le=0.4)
    epochs_tuning: int = Field(default=20, ge=1)
    epochs_final: int = Field(default=40, ge=1)
    patience: int = Field(default=5, ge=1)
    top_k_final: int = Field(default=2, ge=1)
    direction: Literal["maximize", "minimize"] = "maximize"


class AppConfig(BaseModel):
    runs_dir: str = "runs"
    sample_fraction_tuning: float = Field(default=0.1, ge=0.01, le=1.0)
    random_seed: int = 42
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    llm: LLMConfig = LLMConfig()
    tracking: TrackingConfig = TrackingConfig()
    policy: PolicyConfig = PolicyConfig()


DEFAULT_CONFIG = AppConfig()


def load_config(config_path: Path | None) -> AppConfig:
    if config_path is None:
        return DEFAULT_CONFIG
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)


def save_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = DEFAULT_CONFIG.model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
