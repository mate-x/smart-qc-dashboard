from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

import utils.cache_manager as cm
from utils.cache_manager import (
    MAX_ANOMALY_MAP_CACHE,
    get_anomaly_map_cache,
    invalidate_anomaly_map_cache,
    set_anomaly_map_cache,
)


@pytest.fixture(autouse=True)
def fake_session_state(monkeypatch):
    state: dict = {}
    mock_st = MagicMock()
    mock_st.session_state = state
    monkeypatch.setattr(cm, "st", mock_st)
    return state


class TestSetAndGetCache:
    def test_set_stores_data(self, fake_session_state):
        set_anomaly_map_cache("exp1", {"map": [1, 2, 3]})
        cached = get_anomaly_map_cache("exp1")
        assert cached is not None
        assert cached["map"] == [1, 2, 3]

    def test_cached_at_is_added(self, fake_session_state):
        set_anomaly_map_cache("exp2", {"map": []})
        cached = get_anomaly_map_cache("exp2")
        assert "cached_at" in cached

    def test_get_nonexistent_returns_none(self):
        assert get_anomaly_map_cache("nonexistent_exp") is None


class TestLruEviction:
    def test_evicts_oldest_on_overflow(self, fake_session_state):
        for i in range(MAX_ANOMALY_MAP_CACHE):
            set_anomaly_map_cache(f"exp{i}", {"idx": i})
            time.sleep(0.01)

        oldest_key = "exp0"
        assert get_anomaly_map_cache(oldest_key) is not None

        set_anomaly_map_cache("exp_new", {"idx": 99})

        cached_keys = [k for k in fake_session_state if k.startswith(cm._KEY_PREFIX)]
        assert len(cached_keys) <= MAX_ANOMALY_MAP_CACHE
        assert get_anomaly_map_cache(oldest_key) is None

    def test_does_not_evict_below_max(self, fake_session_state):
        for i in range(MAX_ANOMALY_MAP_CACHE):
            set_anomaly_map_cache(f"slot{i}", {"v": i})
        cached_keys = [k for k in fake_session_state if k.startswith(cm._KEY_PREFIX)]
        assert len(cached_keys) == MAX_ANOMALY_MAP_CACHE


class TestInvalidateAnomalyMapCache:
    def test_removes_existing_entry(self, fake_session_state):
        set_anomaly_map_cache("exp_del", {"map": [1]})
        assert get_anomaly_map_cache("exp_del") is not None
        invalidate_anomaly_map_cache("exp_del")
        assert get_anomaly_map_cache("exp_del") is None

    def test_noop_when_key_absent(self, fake_session_state):
        """존재하지 않는 키 제거 시도 — 예외 없이 no-op."""
        invalidate_anomaly_map_cache("nonexistent_exp")  # should not raise

    def test_only_removes_target(self, fake_session_state):
        set_anomaly_map_cache("keep_me", {"x": 1})
        set_anomaly_map_cache("remove_me", {"x": 2})
        invalidate_anomaly_map_cache("remove_me")
        assert get_anomaly_map_cache("keep_me") is not None
        assert get_anomaly_map_cache("remove_me") is None
