from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import yaml

import pytest
import torch

import utils.storage as storage
from utils.storage import load_history, save_completed_experiment


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "HISTORY_FILE", tmp_path / "experiments" / "history.json")
    monkeypatch.setattr(storage, "MODELS_DIR", tmp_path / "models")
    return tmp_path


def _fake_model():
    mock = MagicMock()
    # MagicMock 대신 실제 텐서나 빈 딕셔너리를 반환하도록 수정
    mock.state_dict.return_value = {"layer_weight": torch.randn(1, 1)} 
    return mock


class TestSaveCompletedExperiment:
    def test_creates_model_dir(self, tmp_path):
        model = _fake_model()
        record = {
            "name": "exp_test",
            "created_at": "2026-01-01T00:00:00",
            "preprocessing_config": {},
            "model_config": {},
        }
        save_completed_experiment("exp_001", model, record)
        assert (tmp_path / "models" / "exp_001").is_dir()

    def test_creates_pth_file(self, tmp_path):
        model = _fake_model()
        record = {"name": "x", "created_at": "", "preprocessing_config": {}, "model_config": {}}
        save_completed_experiment("exp_002", model, record)
        pth = tmp_path / "models" / "exp_002" / "model_state_dict.pth"
        assert pth.exists()

    def test_creates_configs_yaml(self, tmp_path):
        model = _fake_model()
        record = {"name": "y", "created_at": "", "preprocessing_config": {}, "model_config": {}}
        save_completed_experiment("exp_003", model, record)
        configs = tmp_path / "models" / "exp_003" / "configs.yaml"
        assert configs.exists()

    def test_appends_to_history(self):
        model = _fake_model()
        record = {"name": "z", "created_at": "", "preprocessing_config": {}, "model_config": {}}
        save_completed_experiment("exp_004", model, record)
        history = load_history()
        assert len(history) == 1
        assert history[0]["name"] == "z"

    def test_record_contains_model_path(self, tmp_path):
        model = _fake_model()
        record = {"name": "w", "created_at": "", "preprocessing_config": {}, "model_config": {}}
        save_completed_experiment("exp_005", model, record)
        history = load_history()
        assert "model_path" in history[0]
        assert "configs_path" in history[0]


class TestSaveExperimentCleanupOnFailure:
    def test_model_dir_removed_on_stage1_failure(self, tmp_path, monkeypatch):
        import torch
        monkeypatch.setattr(torch, "save", MagicMock(side_effect=RuntimeError("disk full")))
        model = _fake_model()
        record = {"name": "fail", "created_at": "", "preprocessing_config": {}, "model_config": {}}
        with pytest.raises(RuntimeError, match="disk full"):
            save_completed_experiment("exp_fail", model, record)
        assert not (tmp_path / "models" / "exp_fail").exists()

    def test_history_not_written_on_failure(self, tmp_path, monkeypatch):
        import torch
        monkeypatch.setattr(torch, "save", MagicMock(side_effect=RuntimeError("fail")))
        model = _fake_model()
        record = {"name": "fail2", "created_at": "", "preprocessing_config": {}, "model_config": {}}
        with pytest.raises(RuntimeError):
            save_completed_experiment("exp_fail2", model, record)
        assert load_history() == []


class TestSaveCompletedExperimentConfigSnapshot:
    def test_explicit_config_params_take_priority_over_record_keys(self, tmp_path):
        model = _fake_model()
        explicit_preproc = {"resize": 256, "normalize": True}
        explicit_model = {"model_size": "medium", "max_steps": 70000}
        record = {
            "name": "snap_test",
            "created_at": "2026-01-01T00:00:00",
            "preprocessing_config": {"resize": 128},  # should be overridden
            "model_config": {"model_size": "small"},  # should be overridden
        }
        save_completed_experiment(
            "exp_snap_001",
            model,
            record,
            preprocessing_config=explicit_preproc,
            model_config=explicit_model,
        )
        configs_path = tmp_path / "models" / "exp_snap_001" / "configs.yaml"
        with open(configs_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["preprocessing"]["resize"] == 256
        assert data["model"]["model_size"] == "medium"

    def test_fallback_to_record_keys_when_params_omitted(self, tmp_path):
        model = _fake_model()
        record = {
            "name": "snap_fallback",
            "created_at": "2026-01-01T00:00:00",
            "preprocessing_config": {"resize": 64},
            "model_config": {"backbone": "wrn50"},
        }
        save_completed_experiment("exp_snap_002", model, record)
        configs_path = tmp_path / "models" / "exp_snap_002" / "configs.yaml"
        with open(configs_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["preprocessing"]["resize"] == 64
        assert data["model"]["backbone"] == "wrn50"

    def test_experiment_section_written_from_record(self, tmp_path):
        model = _fake_model()
        record = {
            "name": "snap_meta",
            "created_at": "2026-06-01T12:00:00",
            "preprocessing_config": {},
            "model_config": {},
        }
        save_completed_experiment("exp_snap_003", model, record)
        configs_path = tmp_path / "models" / "exp_snap_003" / "configs.yaml"
        with open(configs_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["experiment"]["name"] == "snap_meta"
        assert data["experiment"]["created_at"] == "2026-06-01T12:00:00"
