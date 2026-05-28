"""
tests/integration/test_insp_flow.py

검사 E2E 통합 테스트 (TC-INSP-01 ~ TC-INSP-04).
13_QA §I.2 / 04_System_Architecture §B.10 / 08_AI_ML_Integration 기준.

TC-INSP-01: 모델 적용 → insp_active_model 설정 + 검사 실행 → seq/record 갱신
TC-INSP-02: 자동 검사 중 불량 → auto_active=False, defect_popup=True
TC-INSP-03: 모델 교체 → reset_inspection_state() → 이력 초기화, 새 모델 적용
TC-INSP-04: test_pool 소진 → 재셔플 + 인덱스 리셋 (A-16)
E2E:  N회 혼합 검사 → KPI 수치 + CSV 구조 검증
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import streamlit as _st

from inspection.utils.test_sampler import build_test_pool, sample_from_pool


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_insp_state(
    *,
    threshold: float = 0.5,
    auto_active: bool = False,
    records: list | None = None,
    seq_counter: int = 0,
    pool: list | None = None,
    pool_index: int = 0,
) -> dict:
    return {
        "insp_active_model": {
            "threshold": threshold,
            "preprocessing_config": {"method": "none", "params": {}, "image_size": 64},
        },
        "insp_records":          records if records is not None else [],
        "insp_seq_counter":      seq_counter,
        "insp_auto_active":      auto_active,
        "insp_last_result":      None,
        "insp_last_anomaly_map": None,
        "insp_defect_popup":     False,
        "insp_test_pool":        pool if pool is not None else [("/img/a.png", "양품")],
        "insp_pool_index":       pool_index,
    }


def _fake_preprocessing():
    tensor = MagicMock()
    tensor.unsqueeze.return_value = MagicMock()
    return MagicMock(), tensor


def _run_inspection_with_score(score: float, state: dict, monkeypatch) -> bool:
    """score 값을 anomaly_map 최댓값으로 갖는 단일 검사 실행."""
    anomaly_map = np.full((32, 32), score, dtype=np.float32)
    monkeypatch.setattr(_st, "session_state", state)
    monkeypatch.setattr(_st, "toast", MagicMock())
    monkeypatch.setattr(_st, "error", MagicMock())

    with patch(
        "inspection.utils.test_sampler.sample_from_pool",
        return_value=("/img/test.png", "양품", False),
    ), patch(
        "inspection.tabs.insp_tab1_realtime.get_insp_model",
        return_value=MagicMock(),
    ), patch(
        "utils.image_utils.apply_preprocessing",
        return_value=_fake_preprocessing(),
    ), patch(
        "utils.model_factory.run_inference",
        return_value=anomaly_map,
    ):
        from inspection.tabs.insp_tab1_realtime import _run_single_inspection
        return _run_single_inspection()


# ── TC-INSP-01: 모델 적용 후 검사 실행 ──────────────────────────────────────


class TestModelApplyAndInspect:
    def test_inspection_appends_record(self, monkeypatch):
        """TC-INSP-01: 검사 1회 실행 → insp_records에 레코드 1개 추가."""
        state = _make_insp_state()
        result = _run_inspection_with_score(0.3, state, monkeypatch)

        assert result is True
        assert len(state["insp_records"]) == 1

    def test_inspection_increments_seq_counter(self, monkeypatch):
        """TC-INSP-01: 검사 완료 → insp_seq_counter +1, record.seq 일치."""
        state = _make_insp_state(seq_counter=0)
        _run_inspection_with_score(0.3, state, monkeypatch)

        assert state["insp_seq_counter"] == 1
        assert state["insp_records"][0]["seq"] == 1

    def test_record_fields_complete(self, monkeypatch):
        """TC-INSP-01: record는 00_Global §1.10 스키마 필드 포함."""
        state = _make_insp_state()
        _run_inspection_with_score(0.3, state, monkeypatch)

        record = state["insp_records"][0]
        for field in ("seq", "inspected_at", "image_name", "image_path", "verdict", "anomaly_score"):
            assert field in record, f"누락 필드: {field}"

    def test_good_verdict_below_threshold(self, monkeypatch):
        """threshold 미만 score → 양품 판정."""
        state = _make_insp_state(threshold=0.5)
        _run_inspection_with_score(0.3, state, monkeypatch)

        assert state["insp_records"][0]["verdict"] == "양품"

    def test_defect_verdict_at_threshold(self, monkeypatch):
        """threshold 이상 score → 불량 판정."""
        state = _make_insp_state(threshold=0.5)
        _run_inspection_with_score(0.5, state, monkeypatch)

        assert state["insp_records"][0]["verdict"] == "불량"

    def test_last_result_updated(self, monkeypatch):
        """검사 후 insp_last_result 갱신."""
        state = _make_insp_state()
        _run_inspection_with_score(0.3, state, monkeypatch)

        assert state["insp_last_result"] is not None
        assert state["insp_last_result"]["seq"] == 1

    def test_last_anomaly_map_updated(self, monkeypatch):
        """검사 후 insp_last_anomaly_map 갱신."""
        state = _make_insp_state()
        _run_inspection_with_score(0.3, state, monkeypatch)

        assert state["insp_last_anomaly_map"] is not None
        assert isinstance(state["insp_last_anomaly_map"], np.ndarray)

    def test_sequential_seq_numbers(self, monkeypatch):
        """3회 연속 검사 → seq = 1, 2, 3."""
        state = _make_insp_state()
        for _ in range(3):
            _run_inspection_with_score(0.3, state, monkeypatch)

        seqs = [r["seq"] for r in state["insp_records"]]
        assert seqs == [1, 2, 3]


# ── TC-INSP-02: 자동 검사 중 불량 감지 ──────────────────────────────────────


class TestAutoInspectionDefect:
    def test_defect_during_auto_stops_loop(self, monkeypatch):
        """TC-INSP-02: 자동 검사 중 불량 → auto_active=False, defect_popup=True."""
        state = _make_insp_state(threshold=0.5, auto_active=True)
        _run_inspection_with_score(0.8, state, monkeypatch)

        assert state["insp_auto_active"] is False
        assert state["insp_defect_popup"] is True

    def test_defect_during_manual_no_popup(self, monkeypatch):
        """수동 검사(auto_active=False) 중 불량 → popup 미설정."""
        state = _make_insp_state(threshold=0.5, auto_active=False)
        _run_inspection_with_score(0.8, state, monkeypatch)

        assert state["insp_defect_popup"] is False

    def test_good_during_auto_preserves_loop(self, monkeypatch):
        """자동 검사 중 양품 → auto_active 유지 (True → True)."""
        state = _make_insp_state(threshold=0.5, auto_active=True)
        _run_inspection_with_score(0.3, state, monkeypatch)

        assert state["insp_auto_active"] is True
        assert state["insp_defect_popup"] is False


# ── TC-INSP-03: 모델 교체 ────────────────────────────────────────────────────


class TestModelReplacement:
    def test_reset_clears_records_and_counter(self, monkeypatch):
        """TC-INSP-03: reset_inspection_state() → records=[], seq_counter=0."""
        from inspection.utils.insp_session_init import reset_inspection_state

        class _AttrDict(dict):
            __getattr__ = dict.__getitem__  # type: ignore[assignment]
            def __setattr__(self, k, v): self[k] = v  # type: ignore[override]

        state = _AttrDict(
            insp_active_model={"experiment_id": "exp_old"},
            insp_records=[{"seq": i} for i in range(5)],
            insp_seq_counter=5,
            insp_auto_active=True,
            insp_last_result={"seq": 5},
            insp_last_anomaly_map=None,
            insp_defect_popup=True,
            insp_test_pool=[("/img/a.png", "양품")],
            insp_pool_index=1,
        )
        monkeypatch.setattr(_st, "session_state", state)

        reset_inspection_state()

        assert state["insp_records"] == []
        assert state["insp_seq_counter"] == 0
        assert state["insp_auto_active"] is False
        assert state["insp_last_result"] is None
        assert state["insp_defect_popup"] is False
        assert state["insp_test_pool"] == []
        assert state["insp_pool_index"] == 0

    def test_reset_preserves_active_model(self, monkeypatch):
        """TC-INSP-03: reset_inspection_state() — insp_active_model 유지 (R-INSP-05)."""
        from inspection.utils.insp_session_init import reset_inspection_state

        class _AttrDict(dict):
            __getattr__ = dict.__getitem__  # type: ignore[assignment]
            def __setattr__(self, k, v): self[k] = v  # type: ignore[override]

        active_model = {"experiment_id": "exp_keep", "model_path": "/models/exp_keep"}
        state = _AttrDict(
            insp_active_model=active_model,
            insp_records=[{"seq": 1}],
            insp_seq_counter=1,
            insp_auto_active=False,
            insp_last_result=None,
            insp_last_anomaly_map=None,
            insp_defect_popup=False,
            insp_test_pool=[],
            insp_pool_index=0,
        )
        monkeypatch.setattr(_st, "session_state", state)

        reset_inspection_state()

        assert state["insp_active_model"] is active_model

    def test_new_model_overwrites_active_model(self, monkeypatch):
        """모델 교체: reset 후 새 insp_active_model 설정 → 반영 확인."""
        class _AttrDict(dict):
            __getattr__ = dict.__getitem__  # type: ignore[assignment]
            def __setattr__(self, k, v): self[k] = v  # type: ignore[override]

        from inspection.utils.insp_session_init import reset_inspection_state

        state = _AttrDict(
            insp_active_model={"experiment_id": "exp_old"},
            insp_records=[{"seq": 1}],
            insp_seq_counter=1,
            insp_auto_active=False,
            insp_last_result=None,
            insp_last_anomaly_map=None,
            insp_defect_popup=False,
            insp_test_pool=[],
            insp_pool_index=0,
        )
        monkeypatch.setattr(_st, "session_state", state)

        reset_inspection_state()

        new_model = {"experiment_id": "exp_new", "threshold": 0.6}
        state["insp_active_model"] = new_model

        assert state["insp_active_model"]["experiment_id"] == "exp_new"
        assert state["insp_records"] == []  # reset 유지


# ── TC-INSP-04: 테스트 풀 소진 재셔플 ───────────────────────────────────────


class TestPoolExhaustion:
    def test_reshuffle_on_index_exhausted(self, monkeypatch):
        """TC-INSP-04: pool_index >= len(pool) → 재셔플 + was_reshuffled=True."""
        pool = [("/img/a.png", "양품"), ("/img/b.png", "불량"), ("/img/c.png", "양품")]
        state: dict = {"insp_test_pool": list(pool), "insp_pool_index": len(pool)}
        monkeypatch.setattr(_st, "session_state", state)

        _, _, was_reshuffled = sample_from_pool()

        assert was_reshuffled is True

    def test_pool_index_resets_after_reshuffle(self, monkeypatch):
        """TC-INSP-04: 재셔플 후 insp_pool_index == 1 (sample 1회 소비)."""
        pool = [("/img/a.png", "양품"), ("/img/b.png", "불량")]
        state: dict = {"insp_test_pool": list(pool), "insp_pool_index": len(pool)}
        monkeypatch.setattr(_st, "session_state", state)

        sample_from_pool()

        assert state["insp_pool_index"] == 1

    def test_no_reshuffle_before_exhaustion(self, monkeypatch):
        """풀 미소진 상태에서는 재셔플 없음."""
        pool = [("/img/a.png", "양품"), ("/img/b.png", "불량")]
        state: dict = {"insp_test_pool": list(pool), "insp_pool_index": 0}
        monkeypatch.setattr(_st, "session_state", state)

        _, _, was_reshuffled = sample_from_pool()

        assert was_reshuffled is False
        assert state["insp_pool_index"] == 1

    def test_pool_exhaustion_with_real_dataset(self, tmp_path, monkeypatch):
        """TC-INSP-04: 실제 데이터셋으로 build_test_pool → 소진 → 재셔플."""
        from PIL import Image

        rng = np.random.default_rng(seed=7)
        good_dir = tmp_path / "test" / "good"
        good_dir.mkdir(parents=True)
        for i in range(3):
            arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(good_dir / f"{i:03d}.png")

        pool = build_test_pool(str(tmp_path))
        assert len(pool) == 3

        state: dict = {"insp_test_pool": pool, "insp_pool_index": 0}
        monkeypatch.setattr(_st, "session_state", state)

        # 풀 전체 소비
        for _ in range(3):
            sample_from_pool()

        # 다음 샘플에서 재셔플
        _, _, was_reshuffled = sample_from_pool()
        assert was_reshuffled is True


# ── E2E: 혼합 검사 → KPI + CSV 검증 ──────────────────────────────────────────


class TestInspectionE2E:
    """4회 검사 (양품 3, 불량 1) → KPI 수치 + CSV 구조 검증."""

    @pytest.fixture
    def e2e_records(self, monkeypatch):
        """3 양품 + 1 불량 검사 이력 생성."""
        state = _make_insp_state(threshold=0.5)
        scores = [0.2, 0.3, 0.8, 0.1]  # index 2: 불량
        for score in scores:
            _run_inspection_with_score(score, state, monkeypatch)
        return state["insp_records"]

    def test_e2e_total_record_count(self, e2e_records):
        assert len(e2e_records) == 4

    def test_e2e_seq_sequential(self, e2e_records):
        seqs = [r["seq"] for r in e2e_records]
        assert seqs == [1, 2, 3, 4]

    def test_e2e_kpi_good_count(self, e2e_records):
        good = sum(1 for r in e2e_records if r["verdict"] == "양품")
        assert good == 3

    def test_e2e_kpi_bad_count(self, e2e_records):
        bad = sum(1 for r in e2e_records if r["verdict"] == "불량")
        assert bad == 1

    def test_e2e_kpi_defect_rate(self, e2e_records):
        total = len(e2e_records)
        bad = sum(1 for r in e2e_records if r["verdict"] == "불량")
        rate_str = f"{bad / total * 100:.1f}%"
        assert rate_str == "25.0%"

    def test_e2e_csv_has_bom(self, e2e_records):
        from inspection.tabs.insp_tab2_history import _build_csv

        csv_bytes = _build_csv(e2e_records)
        assert csv_bytes[:3] == b"\xef\xbb\xbf"

    def test_e2e_csv_row_count(self, e2e_records):
        from inspection.tabs.insp_tab2_history import _build_csv

        csv_text = _build_csv(e2e_records).decode("utf-8-sig")
        lines = [l for l in csv_text.splitlines() if l.strip()]
        assert len(lines) == 1 + 4  # 헤더 + 4개 레코드

    def test_e2e_csv_no_emoji_in_verdict(self, e2e_records):
        from inspection.tabs.insp_tab2_history import _build_csv

        csv_text = _build_csv(e2e_records).decode("utf-8-sig")
        assert "🟢" not in csv_text
        assert "🔴" not in csv_text
        assert "양품" in csv_text
        assert "불량" in csv_text

    def test_e2e_csv_columns_order(self, e2e_records):
        from inspection.tabs.insp_tab2_history import _build_csv

        csv_text = _build_csv(e2e_records).decode("utf-8-sig")
        header = csv_text.splitlines()[0]
        assert header == "번호,시각,이미지명,판정결과,Anomaly Score"

    def test_e2e_anomaly_score_rounded(self, e2e_records):
        """anomaly_score는 round(score, 6) 정밀도로 저장."""
        for record in e2e_records:
            score = record["anomaly_score"]
            assert isinstance(score, float)
            assert score == round(score, 6)
