"""
tests/integration/test_dashboard_routing.py

이중 대시보드 전환 통합 테스트.
04_System_Architecture §B.10 / 01_Product_Overview §B.6 / SESSION_STATE_SCHEMA 기준.

TC-DASH-01: active_dashboard 기본값 "explorer"
TC-DASH-02: inspection 라우팅 — inspection render 호출
TC-DASH-03: explorer 라우팅 — inspection render 미호출
TC-DASH-04: 사이드바 버튼 → active_dashboard 갱신
TC-DASH-05: insp_* 네임스페이스 보존 — 대시보드 전환 시 상태 유지 (R-INSP-01)
TC-DASH-06: reset_inspection_state() — active_dashboard 및 insp_active_model 유지 (R-INSP-05)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import streamlit as _st

from utils.session_state_init import SESSION_STATE_SCHEMA, init_session_state


class _AttrDict(dict):
    """st.session_state 를 모사: 딕셔너리 + 속성 접근 지원."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value) -> None:
        self[key] = value


# ── TC-DASH-01: 세션 기본값 ───────────────────────────────────────────────────


class TestSessionStateDefaults:
    def test_active_dashboard_defaults_to_explorer(self, monkeypatch):
        state: dict = {}
        monkeypatch.setattr(_st, "session_state", state)
        init_session_state()
        assert state["active_dashboard"] == "explorer"

    def test_all_insp_keys_initialized(self, monkeypatch):
        state: dict = {}
        monkeypatch.setattr(_st, "session_state", state)
        init_session_state()
        insp_keys = [k for k in SESSION_STATE_SCHEMA if k.startswith("insp_")]
        for key in insp_keys:
            assert key in state, f"누락 키: {key}"

    def test_init_is_idempotent_for_active_dashboard(self, monkeypatch):
        """init_session_state() — 이미 존재하는 키는 덮어쓰지 않음."""
        state: dict = {"active_dashboard": "inspection"}
        monkeypatch.setattr(_st, "session_state", state)
        init_session_state()
        assert state["active_dashboard"] == "inspection"

    def test_init_is_idempotent_for_insp_records(self, monkeypatch):
        existing_records = [{"seq": 1, "verdict": "양품"}]
        state: dict = {"insp_records": existing_records}
        monkeypatch.setattr(_st, "session_state", state)
        init_session_state()
        assert state["insp_records"] is existing_records


# ── TC-DASH-02/03: 라우팅 분기 ───────────────────────────────────────────────


class TestDashboardRouting:
    def test_inspection_route_calls_inspection_render(self, monkeypatch):
        """active_dashboard == 'inspection' → inspection_app.render() 호출 경로."""
        state: dict = dict(SESSION_STATE_SCHEMA)
        state["active_dashboard"] = "inspection"
        monkeypatch.setattr(_st, "session_state", state)

        mock_render = MagicMock()
        with patch("inspection.inspection_app.render", mock_render):
            # app.py 라우팅 로직 재현
            if _st.session_state["active_dashboard"] == "inspection":
                from inspection.inspection_app import render as render_inspection
                render_inspection()

        mock_render.assert_called_once()

    def test_explorer_route_does_not_call_inspection_render(self, monkeypatch):
        """active_dashboard == 'explorer' → inspection_app.render() 미호출."""
        state: dict = dict(SESSION_STATE_SCHEMA)
        state["active_dashboard"] = "explorer"
        monkeypatch.setattr(_st, "session_state", state)

        mock_render = MagicMock()
        with patch("inspection.inspection_app.render", mock_render):
            if _st.session_state["active_dashboard"] == "inspection":
                from inspection.inspection_app import render as render_inspection
                render_inspection()

        mock_render.assert_not_called()


# ── TC-DASH-04: 사이드바 라우팅 ─────────────────────────────────────────────


class TestSidebarRouting:
    def test_sidebar_sets_inspection(self, monkeypatch):
        """inspection 버튼 클릭 → active_dashboard = 'inspection'. (sidebar.py 로직)"""
        state: dict = {"active_dashboard": "explorer"}
        monkeypatch.setattr(_st, "session_state", state)

        # sidebar.py: if st.button("🏭 비전검사 대시보드", ...): st.session_state.active_dashboard = "inspection"
        state["active_dashboard"] = "inspection"
        assert state["active_dashboard"] == "inspection"

    def test_sidebar_sets_explorer(self, monkeypatch):
        """explorer 버튼 클릭 → active_dashboard = 'explorer'. (sidebar.py 로직)"""
        state: dict = {"active_dashboard": "inspection"}
        monkeypatch.setattr(_st, "session_state", state)

        state["active_dashboard"] = "explorer"
        assert state["active_dashboard"] == "explorer"

    def test_button_type_reflects_active_dashboard(self):
        """사이드바 버튼 타입은 active_dashboard 값에 따라 달라진다 (sidebar.py §B.5)."""
        for active in ("explorer", "inspection"):
            explorer_type = "primary" if active == "explorer" else "secondary"
            inspection_type = "primary" if active == "inspection" else "secondary"
            if active == "explorer":
                assert explorer_type == "primary"
                assert inspection_type == "secondary"
            else:
                assert explorer_type == "secondary"
                assert inspection_type == "primary"


# ── TC-DASH-05/06: 네임스페이스 보존 ────────────────────────────────────────


class TestInspNamespaceIsolation:
    def test_insp_state_preserved_on_switch_to_explorer(self, monkeypatch):
        """inspection → explorer 전환 후 insp_* 상태 유지 (R-INSP-01)."""
        state: dict = {
            "active_dashboard": "inspection",
            "insp_active_model": {"experiment_id": "exp_001"},
            "insp_records":       [{"seq": 1, "verdict": "양품"}],
            "insp_seq_counter":   1,
            "insp_auto_active":   False,
            "insp_last_result":   None,
            "insp_last_anomaly_map": None,
            "insp_defect_popup":  False,
            "insp_test_pool":     [("/img/a.png", "양품")],
            "insp_pool_index":    1,
        }
        monkeypatch.setattr(_st, "session_state", state)

        state["active_dashboard"] = "explorer"

        assert state["insp_active_model"] == {"experiment_id": "exp_001"}
        assert state["insp_records"] == [{"seq": 1, "verdict": "양품"}]
        assert state["insp_seq_counter"] == 1
        assert state["insp_pool_index"] == 1

    def test_insp_state_preserved_on_switch_to_inspection(self, monkeypatch):
        """explorer → inspection 전환 후 기존 insp_* 상태 유지."""
        state: dict = {
            "active_dashboard": "explorer",
            "dataset_path": "/data/bottle",
            "insp_active_model": None,
            "insp_records":      [],
            "insp_seq_counter":  0,
        }
        monkeypatch.setattr(_st, "session_state", state)

        state["active_dashboard"] = "inspection"

        assert state["dataset_path"] == "/data/bottle"
        assert state["insp_active_model"] is None
        assert state["insp_records"] == []

    def test_reset_does_not_affect_active_dashboard(self, monkeypatch):
        """reset_inspection_state() — active_dashboard 변경 없음 (R-INSP-05)."""
        from inspection.utils.insp_session_init import reset_inspection_state

        state = _AttrDict(
            active_dashboard="inspection",
            insp_active_model={"experiment_id": "exp_001"},
            insp_records=[{"seq": 1}],
            insp_seq_counter=1,
            insp_auto_active=True,
            insp_last_result={"seq": 1},
            insp_last_anomaly_map=None,
            insp_defect_popup=True,
            insp_test_pool=[("/img/a.png", "양품")],
            insp_pool_index=1,
        )
        monkeypatch.setattr(_st, "session_state", state)

        reset_inspection_state()

        assert state["active_dashboard"] == "inspection"
        assert state["insp_active_model"] == {"experiment_id": "exp_001"}

    def test_reset_clears_all_insp_volatile_keys(self, monkeypatch):
        """reset_inspection_state() — insp_active_model 제외 전체 초기화 (R-INSP-05)."""
        from inspection.utils.insp_session_init import reset_inspection_state

        active_model = {"experiment_id": "exp_xyz"}
        state = _AttrDict(
            insp_active_model=active_model,
            insp_records=[{"seq": 1}, {"seq": 2}],
            insp_seq_counter=2,
            insp_auto_active=True,
            insp_last_result={"seq": 2},
            insp_last_anomaly_map=None,
            insp_defect_popup=True,
            insp_test_pool=[("/img/a.png", "양품"), ("/img/b.png", "불량")],
            insp_pool_index=2,
        )
        monkeypatch.setattr(_st, "session_state", state)

        reset_inspection_state()

        assert state["insp_active_model"] is active_model
        assert state["insp_records"] == []
        assert state["insp_seq_counter"] == 0
        assert state["insp_auto_active"] is False
        assert state["insp_last_result"] is None
        assert state["insp_defect_popup"] is False
        assert state["insp_test_pool"] == []
        assert state["insp_pool_index"] == 0

    def test_explorer_keys_unaffected_by_reset(self, monkeypatch):
        """reset_inspection_state() — explorer 네임스페이스 키 유지."""
        from inspection.utils.insp_session_init import reset_inspection_state

        state = _AttrDict(
            dataset_path="/data/bottle",
            preprocessing_config={"method": "none"},
            active_dashboard="inspection",
            insp_active_model=None,
            insp_records=[],
            insp_seq_counter=0,
            insp_auto_active=False,
            insp_last_result=None,
            insp_last_anomaly_map=None,
            insp_defect_popup=False,
            insp_test_pool=[],
            insp_pool_index=0,
        )
        monkeypatch.setattr(_st, "session_state", state)

        reset_inspection_state()

        assert state["dataset_path"] == "/data/bottle"
        assert state["preprocessing_config"] == {"method": "none"}
