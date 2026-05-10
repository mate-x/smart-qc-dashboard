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

    def test_append_raises_on_duplicate_experiment_id(self):
        """PRD §3.5 중복 ID 금지 — 동일 experiment_id 재삽입 시 RuntimeError."""
        append_experiment({"experiment_id": "exp_dup_001", "name": "first"})
        with pytest.raises(RuntimeError, match="ERR_DUPLICATE_EXPERIMENT_ID"):
            append_experiment({"experiment_id": "exp_dup_001", "name": "duplicate"})

    def test_append_no_experiment_id_does_not_raise(self):
        """experiment_id 키 없는 레코드는 중복 검사 건너뜀."""
        append_experiment({"name": "no_id_1"})
        append_experiment({"name": "no_id_2"})  # 예외 없이 추가돼야 함
        assert len(load_history()) == 2


class TestValidateImagenetPenaltyDir:
    def test_returns_true_with_valid_dir(self, tmp_path, monkeypatch):
        penalty_dir = tmp_path / "imagenet_penalty"
        penalty_dir.mkdir()
        (penalty_dir / "sample.jpg").write_bytes(b"fake")
        monkeypatch.setattr(storage, "IMAGENET_PENALTY_DIR", penalty_dir)
        ok, count = validate_imagenet_penalty_dir()
        assert ok is True
        assert count == 1

    def test_returns_false_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(storage, "IMAGENET_PENALTY_DIR", tmp_path / "nonexistent")
        ok, count = validate_imagenet_penalty_dir()
        assert ok is False
        assert count == 0

    def test_returns_false_when_dir_empty(self, tmp_path, monkeypatch):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr(storage, "IMAGENET_PENALTY_DIR", empty_dir)
        ok, count = validate_imagenet_penalty_dir()
        assert ok is False
        assert count == 0

    def test_counts_only_supported_formats(self, tmp_path, monkeypatch):
        penalty_dir = tmp_path / "imagenet_penalty"
        penalty_dir.mkdir()
        (penalty_dir / "a.jpg").write_bytes(b"fake")
        (penalty_dir / "b.png").write_bytes(b"fake")
        (penalty_dir / "c.tiff").write_bytes(b"fake")   # 미지원 포맷
        (penalty_dir / "d.txt").write_bytes(b"fake")    # 미지원 포맷
        monkeypatch.setattr(storage, "IMAGENET_PENALTY_DIR", penalty_dir)
        ok, count = validate_imagenet_penalty_dir()
        assert ok is True
        assert count == 2
