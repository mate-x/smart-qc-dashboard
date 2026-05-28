"""
tests/inspection/test_test_sampler.py

FR-INSP-T1-06: test_pool 구성 + 샘플링 검증
  - build_test_pool(): 레이블 규칙, FileNotFoundError, 빈 풀
  - sample_from_pool(): 인덱스 증가, 풀 소진 재셔플, 빈 풀 RuntimeError
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import streamlit as _st
from PIL import Image


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def test_dataset(tmp_path):
    """good/ 3장 + scratch/ 3장 MVTec 테스트 디렉토리."""
    good = tmp_path / "test" / "good"
    bad = tmp_path / "test" / "scratch"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)

    rng = np.random.default_rng(seed=0)
    for i in range(3):
        arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(good / f"{i:03d}.png")
        Image.fromarray(arr).save(bad / f"{i:03d}.png")

    return tmp_path


@pytest.fixture
def pool_state(monkeypatch):
    """insp_test_pool 3개, insp_pool_index=0 인 session_state."""
    pool = [("/img/a.png", "양품"), ("/img/b.png", "불량"), ("/img/c.png", "양품")]
    state: dict = {"insp_test_pool": list(pool), "insp_pool_index": 0}
    monkeypatch.setattr(_st, "session_state", state)
    return state


# ── build_test_pool ──────────────────────────────────────────────────────────


class TestBuildTestPool:
    def test_raises_when_test_dir_missing(self, tmp_path):
        from inspection.utils.test_sampler import build_test_pool

        with pytest.raises(FileNotFoundError, match="테스트 디렉토리가 없습니다"):
            build_test_pool(str(tmp_path / "nonexistent"))

    def test_returns_empty_when_no_images(self, tmp_path):
        from inspection.utils.test_sampler import build_test_pool

        (tmp_path / "test" / "good").mkdir(parents=True)
        assert build_test_pool(str(tmp_path)) == []

    def test_good_folder_labeled_yangtum(self, test_dataset):
        from inspection.utils.test_sampler import build_test_pool

        pool = build_test_pool(str(test_dataset))
        good_labels = [label for _, label in pool if label == "양품"]
        assert len(good_labels) == 3

    def test_defect_folder_labeled_bulyang(self, test_dataset):
        from inspection.utils.test_sampler import build_test_pool

        pool = build_test_pool(str(test_dataset))
        bad_labels = [label for _, label in pool if label == "불량"]
        assert len(bad_labels) == 3

    def test_returns_all_images_as_tuples(self, test_dataset):
        from inspection.utils.test_sampler import build_test_pool

        pool = build_test_pool(str(test_dataset))
        assert len(pool) == 6
        assert all(len(item) == 2 for item in pool)

    def test_image_paths_exist(self, test_dataset):
        from inspection.utils.test_sampler import build_test_pool

        pool = build_test_pool(str(test_dataset))
        for path, _ in pool:
            assert Path(path).exists(), f"path does not exist: {path}"


# ── sample_from_pool ─────────────────────────────────────────────────────────


class TestSampleFromPool:
    def test_returns_item_at_current_index(self, pool_state):
        from inspection.utils.test_sampler import sample_from_pool

        state = pool_state
        expected_path = state["insp_test_pool"][0][0]
        expected_label = state["insp_test_pool"][0][1]

        path, label, reshuffled = sample_from_pool()

        assert path == expected_path
        assert label == expected_label
        assert reshuffled is False

    def test_increments_pool_index(self, pool_state):
        from inspection.utils.test_sampler import sample_from_pool

        state = pool_state
        sample_from_pool()
        assert state["insp_pool_index"] == 1

    def test_raises_when_pool_empty(self, monkeypatch):
        from inspection.utils.test_sampler import sample_from_pool

        monkeypatch.setattr(_st, "session_state", {"insp_test_pool": [], "insp_pool_index": 0})

        with pytest.raises(RuntimeError, match="ERR_INSP_TEST_POOL_EMPTY"):
            sample_from_pool()

    def test_reshuffles_when_index_exhausted(self, monkeypatch):
        from inspection.utils.test_sampler import sample_from_pool

        pool = [("/img/a.png", "양품"), ("/img/b.png", "불량")]
        state: dict = {"insp_test_pool": list(pool), "insp_pool_index": len(pool)}
        monkeypatch.setattr(_st, "session_state", state)

        _, _, reshuffled = sample_from_pool()

        assert reshuffled is True

    def test_pool_index_resets_after_reshuffle(self, monkeypatch):
        from inspection.utils.test_sampler import sample_from_pool

        pool = [("/img/a.png", "양품"), ("/img/b.png", "불량")]
        state: dict = {"insp_test_pool": list(pool), "insp_pool_index": len(pool)}
        monkeypatch.setattr(_st, "session_state", state)

        sample_from_pool()

        assert state["insp_pool_index"] == 1  # 0 (리셋) + 1 (다음 인덱스)

    def test_consecutive_samples_advance_index(self, pool_state):
        from inspection.utils.test_sampler import sample_from_pool

        state = pool_state
        sample_from_pool()
        sample_from_pool()
        assert state["insp_pool_index"] == 2
