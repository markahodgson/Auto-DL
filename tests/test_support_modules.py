from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from autodl.config import AppConfig, LLMConfig, TrackingConfig, load_config, save_default_config
from autodl.io import read_input, write_parquet
from autodl.llm.factory import NullProvider, make_llm_provider
from autodl.policy.engine import select_text_backend, should_call_llm
from autodl.tracking.factory import make_tracker
from autodl.tracking.local_tracker import LocalTracker
from autodl.tracking.wandb_tracker import WandbTracker
from autodl.utils import ensure_dir, utc_run_id, write_json, write_jsonl


def test_read_input_csv_and_unsupported(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    df = read_input(csv_path)

    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)

    bad_path = tmp_path / "sample.txt"
    bad_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported input type"):
        read_input(bad_path)


def test_write_parquet_creates_parent_and_calls_dataframe_method(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "nested" / "frame.parquet"
    df = pd.DataFrame({"x": [1, 2]})
    called: dict[str, object] = {}

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        called["path"] = path
        called["index"] = index

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=True)

    write_parquet(df, out)

    assert out.parent.exists()
    assert called["path"] == out
    assert called["index"] is False


def test_utils_writers_and_run_id(tmp_path: Path) -> None:
    run_id = utc_run_id("demo")
    assert re.match(r"^demo_\d{8}T\d{6}Z$", run_id)

    created = ensure_dir(tmp_path / "a" / "b")
    assert created.exists()

    json_path = tmp_path / "meta" / "data.json"
    write_json(json_path, {"x": 1})
    assert '"x": 1' in json_path.read_text(encoding="utf-8")

    jsonl_path = tmp_path / "meta" / "events.jsonl"
    write_jsonl(jsonl_path, [{"msg": "hello"}, {"msg": "café"}])
    content = jsonl_path.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert '"msg": "café"' in content


def test_config_save_and_load_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"

    save_default_config(cfg_path)
    loaded = load_config(cfg_path)
    from_none = load_config(None)

    assert isinstance(loaded, AppConfig)
    assert loaded.train.max_trials >= 1
    assert isinstance(from_none, AppConfig)


def test_llm_factory_selects_provider() -> None:
    provider_none = make_llm_provider(LLMConfig(provider="none"))
    provider_ollama = make_llm_provider(LLMConfig(provider="ollama"))
    provider_openai = make_llm_provider(LLMConfig(provider="openai"))

    assert isinstance(provider_none, NullProvider)
    assert provider_none.suggest("sys", "user").startswith('{"version"')
    assert provider_ollama.__class__.__name__ == "OllamaProvider"
    assert provider_openai.__class__.__name__ == "OpenAIProvider"


def test_tracking_factory_selects_tracker() -> None:
    disabled = make_tracker(TrackingConfig(enabled=False, backend="wandb"))
    local_enabled = make_tracker(TrackingConfig(enabled=True, backend="local"))
    wandb_enabled = make_tracker(TrackingConfig(enabled=True, backend="wandb"))

    assert isinstance(disabled, LocalTracker)
    assert isinstance(local_enabled, LocalTracker)
    assert isinstance(wandb_enabled, WandbTracker)


def test_policy_engine_branches_and_llm_gate() -> None:
    large = select_text_backend(n_rows=2_000_000, time_budget_minutes=90)
    tight = select_text_backend(n_rows=1000, time_budget_minutes=30)
    tf_target = select_text_backend(n_rows=1000, time_budget_minutes=120, deployment_target="tensorflow_saved_model")
    other_target = select_text_backend(n_rows=1000, time_budget_minutes=120, deployment_target="onnx")

    assert large["action"] == "hashing"
    assert tight["action"] == "hashing"
    assert tf_target["action"] == "tfidf"
    assert other_target["action"] == "tfidf"

    assert should_call_llm(confidence=0.49, threshold=0.5) is True
    assert should_call_llm(confidence=0.5, threshold=0.5) is False
