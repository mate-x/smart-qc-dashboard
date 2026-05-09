from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import utils.storage as storage
from utils.storage import (
    append_experiment,
    load_history,
    save_history,
    validate_imagenet_penalty_dir,
)


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    history_file = tmp_path / "experiments" / "history.json"
    monkeypatch.setattr(storage, "HISTORY_FILE", history_file)
    return history_file


class TestLoadSaveHistory:
    def test_load_returns_empty_list_when_no_file(self):
        assert load_history() == []

    def test_save_and_load_roundtrip(self, isolated_history):
        records = [{"id": "exp1", "score": 0.95}]
        save_history(records)
        assert load_history() == records

    def test_save_is_atomic_tmpfile(self, isolated_history):
        records = [{"id": "exp_atomic"}]
        save_history(records)
        assert isolated_history.exists()
        tmp = isolated_history.with_suffix(".tmp")
        assert not tmp.exists()

    def test_append_adds_record(self):
        append_experiment({"id": "a"})
        append_experiment({"id": "b"})
        records = load_history()
        assert len(records) == 2
        assert records[0]["id"] == "a"
        assert records[1]["id"] == "b"


class TestValidateImagenetPenaltyDir:
    def test_passes_with_valid_dir(self, tmp_path, monkeypatch):
        penalty_dir = tmp_path / "imagenet_penalty"
        penalty_dir.mkdir()
        (penalty_dir / "sample.jpg").write_bytes(b"fake")
        monkeypatch.setattr(storage, "IMAGENET_PENALTY_DIR", penalty_dir)
        validate_imagenet_penalty_dir()

    def test_raises_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(storage, "IMAGENET_PENALTY_DIR", tmp_path / "nonexistent")
        with pytest.raises(ValueError, match="존재하지 않습니다"):
            validate_imagenet_penalty_dir()

    def test_raises_when_dir_empty(self, tmp_path, monkeypatch):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr(storage, "IMAGENET_PENALTY_DIR", empty_dir)
        with pytest.raises(ValueError, match="이미지 파일이 없습니다"):
            validate_imagenet_penalty_dir()
