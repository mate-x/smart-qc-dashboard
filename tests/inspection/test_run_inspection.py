"""
tests/inspection/test_run_inspection.py

FR-INSP-T1-02, FR-INSP-T1-05 자동 검사 루프 상태 전이 검증:
  - 양품 → insp_records에 record 추가, popup 미설정
  - 불량 + auto_active → insp_auto_active=False, insp_defect_popup=True
  - 불량 + auto_active=False → popup 미설정 (수동 검사 시 팝업 없음)
  - 모델 로드 실패 → False 반환, auto_active=False
  - 풀 샘플 실패 → False 반환, auto_active=False
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import streamlit as _st


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_state(auto_active: bool = True) -> dict:
    return {
        "insp_active_model": {
            "threshold": 0.5,
            "preprocessing_config": {"method": "none", "params": {}, "image_size": 64},
        },
        "insp_records": [],
        "insp_seq_counter": 0,
        "insp_last_result": None,
        "insp_last_anomaly_map": None,
        "insp_auto_active": auto_active,
        "insp_defect_popup": False,
    }


def _fake_image_tensor():
    """apply_preprocessing 의 반환값을 모사하는 (pil, tensor) 쌍."""
    tensor = MagicMock()
    tensor.unsqueeze.return_value = MagicMock()
    return MagicMock(), tensor


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def session(monkeypatch):
    """기본 session_state (auto_active=True)."""
    state = _make_state(auto_active=True)
    monkeypatch.setattr(_st, "session_state", state)
    monkeypatch.setattr(_st, "toast", MagicMock())
    monkeypatch.setattr(_st, "error", MagicMock())
    return state


@pytest.fixture
def low_score_map():
    """threshold(0.5) 미만 → 양품 판정 anomaly map."""
    return np.ones((32, 32), dtype=np.float32) * 0.3


@pytest.fixture
def high_score_map():
    """threshold(0.5) 이상 → 불량 판정 anomaly map."""
    return np.ones((32, 32), dtype=np.float32) * 0.8


# ── tests ──────────────────────────────────────────────────────────────────────


class TestRunSingleInspection:
    def test_yangpum_appends_record(self, session, low_score_map):
        """양품 판정 시 insp_records에 record 1개 추가."""
        with patch(
            "inspection.utils.test_sampler.sample_from_pool",
            return_value=("/img/good_01.png", "양품", False),
        ), patch(
            "inspection.tabs.insp_tab1_realtime.get_insp_model",
            return_value=MagicMock(),
        ), patch(
            "utils.image_utils.apply_preprocessing",
            return_value=_fake_image_tensor(),
        ), patch(
            "utils.model_factory.run_inference",
            return_value=low_score_map,
        ):
            from inspection.tabs.insp_tab1_realtime import _run_single_inspection

            result = _run_single_inspection()

        assert result is True
        assert len(session["insp_records"]) == 1
        assert session["insp_records"][0]["verdict"] == "양품"
        assert session["insp_defect_popup"] is False

    def test_yangpum_does_not_set_popup(self, session, low_score_map):
        """양품 판정 시 팝업 미설정."""
        with patch(
            "inspection.utils.test_sampler.sample_from_pool",
            return_value=("/img/good_01.png", "양품", False),
        ), patch(
            "inspection.tabs.insp_tab1_realtime.get_insp_model",
            return_value=MagicMock(),
        ), patch(
            "utils.image_utils.apply_preprocessing",
            return_value=_fake_image_tensor(),
        ), patch(
            "utils.model_factory.run_inference",
            return_value=low_score_map,
        ):
            from inspection.tabs.insp_tab1_realtime import _run_single_inspection

            _run_single_inspection()

        assert session["insp_defect_popup"] is False
        assert session["insp_auto_active"] is True

    def test_bulyang_auto_sets_popup_and_stops_loop(self, session, high_score_map):
        """FR-INSP-T1-02: 자동 검사 중 불량 감지 → auto_active=False, defect_popup=True."""
        with patch(
            "inspection.utils.test_sampler.sample_from_pool",
            return_value=("/img/bad_01.png", "불량", False),
        ), patch(
            "inspection.tabs.insp_tab1_realtime.get_insp_model",
            return_value=MagicMock(),
        ), patch(
            "utils.image_utils.apply_preprocessing",
            return_value=_fake_image_tensor(),
        ), patch(
            "utils.model_factory.run_inference",
            return_value=high_score_map,
        ):
            from inspection.tabs.insp_tab1_realtime import _run_single_inspection

            result = _run_single_inspection()

        assert result is True
        assert session["insp_auto_active"] is False
        assert session["insp_defect_popup"] is True

    def test_bulyang_manual_does_not_set_popup(self, monkeypatch, high_score_map):
        """수동 검사(auto_active=False) 중 불량 감지 → 팝업 미설정 (FR-INSP-T1-05)."""
        state = _make_state(auto_active=False)
        monkeypatch.setattr(_st, "session_state", state)
        monkeypatch.setattr(_st, "toast", MagicMock())
        monkeypatch.setattr(_st, "error", MagicMock())

        with patch(
            "inspection.utils.test_sampler.sample_from_pool",
            return_value=("/img/bad_01.png", "불량", False),
        ), patch(
            "inspection.tabs.insp_tab1_realtime.get_insp_model",
            return_value=MagicMock(),
        ), patch(
            "utils.image_utils.apply_preprocessing",
            return_value=_fake_image_tensor(),
        ), patch(
            "utils.model_factory.run_inference",
            return_value=high_score_map,
        ):
            from inspection.tabs.insp_tab1_realtime import _run_single_inspection

            result = _run_single_inspection()

        assert result is True
        assert state["insp_defect_popup"] is False

    def test_seq_counter_increments(self, session, low_score_map):
        """검사 1회 완료 시 insp_seq_counter +1, record.seq 일치."""
        with patch(
            "inspection.utils.test_sampler.sample_from_pool",
            return_value=("/img/good_01.png", "양품", False),
        ), patch(
            "inspection.tabs.insp_tab1_realtime.get_insp_model",
            return_value=MagicMock(),
        ), patch(
            "utils.image_utils.apply_preprocessing",
            return_value=_fake_image_tensor(),
        ), patch(
            "utils.model_factory.run_inference",
            return_value=low_score_map,
        ):
            from inspection.tabs.insp_tab1_realtime import _run_single_inspection

            _run_single_inspection()

        assert session["insp_seq_counter"] == 1
        assert session["insp_records"][0]["seq"] == 1

    def test_model_load_failure_returns_false(self, session):
        """모델 로드 실패 시 False 반환, auto_active=False."""
        with patch(
            "inspection.utils.test_sampler.sample_from_pool",
            return_value=("/img/good_01.png", "양품", False),
        ), patch(
            "inspection.tabs.insp_tab1_realtime.get_insp_model",
            return_value=None,
        ):
            from inspection.tabs.insp_tab1_realtime import _run_single_inspection

            result = _run_single_inspection()

        assert result is False
        assert session["insp_auto_active"] is False

    def test_pool_sample_failure_returns_false(self, session):
        """pool 샘플링 실패(RuntimeError) 시 False 반환, auto_active=False."""
        with patch(
            "inspection.utils.test_sampler.sample_from_pool",
            side_effect=RuntimeError("ERR_INSP_TEST_POOL_EMPTY"),
        ):
            from inspection.tabs.insp_tab1_realtime import _run_single_inspection

            result = _run_single_inspection()

        assert result is False
        assert session["insp_auto_active"] is False

    def test_reshuffle_triggers_toast(self, session, low_score_map):
        """pool 재셔플 발생 시 st.toast 호출 (FR-INSP-T1-06)."""
        mock_toast = MagicMock()
        session  # already set by fixture
        _st.toast = mock_toast  # monkeypatch already set this but override to capture

        with patch(
            "inspection.utils.test_sampler.sample_from_pool",
            return_value=("/img/good_01.png", "양품", True),  # was_reshuffled=True
        ), patch(
            "inspection.tabs.insp_tab1_realtime.get_insp_model",
            return_value=MagicMock(),
        ), patch(
            "utils.image_utils.apply_preprocessing",
            return_value=_fake_image_tensor(),
        ), patch(
            "utils.model_factory.run_inference",
            return_value=low_score_map,
        ):
            from inspection.tabs.insp_tab1_realtime import _run_single_inspection

            _run_single_inspection()

        mock_toast.assert_called_once()
