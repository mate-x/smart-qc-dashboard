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


class TestSaveCompletedExperimentConfigSnapshot:
    """configs.yaml 스냅샷 내용 검증 (PRD Z.5)."""

    def test_explicit_config_params_take_priority_over_record_keys(self, tmp_path):
        """preprocessing_config/model_config 명시 시 record 키보다 우선 적용."""
        import yaml
        model = _fake_model()
        record = {
            "name": "cfg_priority_test",
            "created_at": "2026-01-01T00:00:00",
            "preprocessing_config": {"method": "none_from_record"},
            "model_config": {"model_type": "wrong_from_record"},
        }
        preproc = {"method": "clahe", "params": {"clip_limit": 2.0}}
        mdl = {"model_type": "efficientad", "batch_size": 8, "image_size": 256}
        save_completed_experiment(
            "exp_cfgprio",
            model,
            record,
            preprocessing_config=preproc,
            model_config=mdl,
        )
        cfg_path = tmp_path / "models" / "exp_cfgprio" / "configs.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["preprocessing"]["method"] == "clahe"
        assert data["model"]["model_type"] == "efficientad"

    def test_fallback_to_record_keys_when_params_omitted(self, tmp_path):
        """preprocessing_config/model_config 파라미터 미전달 시 record 키 사용."""
        import yaml
        model = _fake_model()
        record = {
            "name": "cfg_fallback_test",
            "created_at": "2026-01-01T00:00:00",
            "preprocessing_config": {"method": "he"},
            "model_config": {"model_type": "patchcore"},
        }
        save_completed_experiment("exp_cfgfb", model, record)
        cfg_path = tmp_path / "models" / "exp_cfgfb" / "configs.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["preprocessing"]["method"] == "he"
        assert data["model"]["model_type"] == "patchcore"

    def test_experiment_section_written_from_record(self, tmp_path):
        """configs.yaml의 experiment 섹션에 name/created_at 기록."""
        import yaml
        model = _fake_model()
        record = {
            "name": "my_experiment",
            "created_at": "2026-05-10T12:00:00+09:00",
            "preprocessing_config": {},
            "model_config": {},
        }
        save_completed_experiment("exp_expsec", model, record)
        cfg_path = tmp_path / "models" / "exp_expsec" / "configs.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["experiment"]["name"] == "my_experiment"
        assert data["experiment"]["created_at"] == "2026-05-10T12:00:00+09:00"


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
