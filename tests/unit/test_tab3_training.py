"""
탭3 단위 테스트

PRD 참조:
  06_API_Specification.md §5.2  (_handle_stopped, _drain_queue, _reset_run_state)
  06_API_Specification.md §6    (stop_event 경쟁 조건 처리 규칙 R-RACE-01/02)
  07_Backend_Service_Design.md §6.1 (완료 후처리 순서)
  07_Backend_Service_Design.md §6.2 (중단 후처리 순서)
  07_Backend_Service_Design.md §2.1 (experiment_id R-NAMING-03)
  07_Backend_Service_Design.md §8.1 (R-UI-02 학습 시작 버튼 idle 전용)
  07_Backend_Service_Design.md §9.2 (GPU 메모리 해제 조건)
"""
from __future__ import annotations

import queue
import re
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import tabs.tab3_training as t3
from utils.messages import MSG


# ── 공용 헬퍼 ──────────────────────────────────────────────────────────────────

EXP_ID = "efficientad_20260509_143022_3b9f"

_EXP_CFG_MOCK = {"experiment": {"name": "Test Exp", "created_at": "2026-05-09T14:30:22+09:00"}}


def _make_ss(**overrides) -> dict:
    """테스트용 session_state 기본값. 키 이름에 underscore 포함 가능."""
    ss: dict = {
        "current_run_status": "running",
        "current_exp_id": EXP_ID,
        "_stop_event": threading.Event(),
        "_result_queue": queue.Queue(),
        "_worker": None,
        "_progress": {"step": 0, "total": 70000, "loss": None, "elapsed": 0.0},
        "_log_lines": [],
        "_loss_history": [],
        "_current_stage_idx": None,
        "_current_stage_name": None,
        "_batch_queue_mode":     False,
        "_batch_total_count":    0,
        "_batch_skip_current":   False,
        "_batch_advance_pending": False,
        "experiment_queue":      [],
        "experiments": {},
        "model_config": {
            "model_type": "efficientad",
            "image_size": 256,
            "batch_size": 8,
            "random_seed": 42,
            "threshold_method": "percentile",
            "threshold_value": 95.0,
            "params": {"train_steps": 70000, "learning_rate": 1e-4},
        },
        "preprocessing_config": {
            "method": "clahe",
            "params": {"clip_limit": 2.0},
        },
        "dataset_path": "/data/mvtec",
    }
    ss.update(overrides)
    return ss


class _SessionCtx:
    """t3.st.session_state를 fake dict로 교체하는 컨텍스트 매니저."""

    def __init__(self, ss: dict) -> None:
        self._ss = ss
        self._orig = None

    def __enter__(self) -> dict:
        self._orig = getattr(t3.st, "session_state", None)
        t3.st.session_state = self._ss
        return self._ss

    def __exit__(self, *_) -> None:
        if self._orig is not None:
            t3.st.session_state = self._orig


# ── TestResetRunState ──────────────────────────────────────────────────────────

class TestResetRunState:
    """_reset_run_state() — session_state 5개 키 초기화 (PRD 06 §5.2)."""

    def _run(self, **overrides) -> dict:
        ss = _make_ss(**overrides)
        with _SessionCtx(ss):
            t3._reset_run_state()
        return ss

    def test_status_becomes_idle(self):
        ss = self._run()
        assert ss["current_run_status"] == "idle"

    def test_exp_id_becomes_none(self):
        ss = self._run()
        assert ss["current_exp_id"] is None

    def test_stop_event_becomes_none(self):
        ss = self._run()
        assert ss["_stop_event"] is None

    def test_result_queue_becomes_none(self):
        """R-RACE-02: _reset_run_state 이후 _result_queue=None → Q 참조 소멸."""
        ss = self._run()
        assert ss["_result_queue"] is None

    def test_worker_becomes_none(self):
        mock_worker = MagicMock()
        ss = _make_ss()
        ss["_worker"] = mock_worker
        with _SessionCtx(ss):
            t3._reset_run_state()
        assert ss["_worker"] is None


# ── TestBuildExperimentRecordStopped ──────────────────────────────────────────

class TestBuildExperimentRecordStopped:
    """_build_experiment_record(status="중단") 불변 조건 (PRD 07 §6.3, 00_Global §2 R-05)."""

    def _build(self, exp_id: str = EXP_ID, ss: dict | None = None) -> dict:
        ss = ss or _make_ss()
        with _SessionCtx(ss), patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK):
            return t3._build_experiment_record(exp_id, "중단", None, None)

    def test_status_is_stopped(self):
        assert self._build()["status"] == "중단"

    def test_metrics_is_none(self):
        """status="중단" → metrics 강제 None."""
        assert self._build()["metrics"] is None

    def test_model_path_is_none(self):
        assert self._build()["model_path"] is None

    def test_configs_path_is_none(self):
        assert self._build()["configs_path"] is None

    def test_experiment_id_matches(self):
        assert self._build(exp_id="my_exp_001")["experiment_id"] == "my_exp_001"

    def test_model_type_from_session_state(self):
        assert self._build()["model_type"] == "efficientad"

    def test_preprocessing_method_from_session_state(self):
        assert self._build()["preprocessing_method"] == "clahe"

    def test_dataset_path_from_session_state(self):
        assert self._build()["dataset_path"] == "/data/mvtec"

    def test_required_keys_present(self):
        required = {
            "experiment_id", "name", "status", "created_at",
            "model_type", "preprocessing_method", "preprocessing_params",
            "model_params", "threshold_method", "threshold_value",
            "dataset_path", "image_size", "duration_seconds",
            "metrics", "model_path", "configs_path",
        }
        assert required <= set(self._build().keys())


# ── TestHandleStopped ─────────────────────────────────────────────────────────

class TestHandleStopped:
    """_handle_stopped() — PRD 07 §6.2 중단 후처리 순서 검증."""

    def _run(self, msg: dict, ss: dict | None = None):
        ss = ss or _make_ss()
        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.append_experiment") as mock_append, \
             patch.object(t3.st, "warning") as mock_warn:
            t3._handle_stopped(msg)
        return ss, mock_append, mock_warn

    def test_append_experiment_called_once(self):
        """PRD 07 §6.2 step2: append_experiment(record) 1회 호출."""
        _, mock_append, _ = self._run({"type": "stopped", "step": 100})
        mock_append.assert_called_once()

    def test_append_experiment_record_status_stopped(self):
        _, mock_append, _ = self._run({"type": "stopped", "step": 100})
        record = mock_append.call_args[0][0]
        assert record["status"] == "중단"

    def test_append_experiment_record_metrics_none(self):
        _, mock_append, _ = self._run({"type": "stopped", "step": 100})
        record = mock_append.call_args[0][0]
        assert record["metrics"] is None

    def test_st_warning_called_once(self):
        """PRD 07 §6.2 step4: _last_result level=warning 설정."""
        ss, _, _ = self._run({"type": "stopped", "step": 50})
        assert ss["_last_result"]["level"] == "warning"

    def test_warning_contains_train_stopped_message(self):
        ss, _, _ = self._run({"type": "stopped", "step": 50})
        assert MSG["TRAIN_STOPPED"] in ss["_last_result"]["text"]

    def test_status_reset_to_idle(self):
        """PRD 07 §6.2 step5: _reset_run_state() → current_run_status="idle"."""
        ss, _, _ = self._run({"type": "stopped", "step": 0})
        assert ss["current_run_status"] == "idle"

    def test_result_queue_none_after_stopped(self):
        """R-RACE-02: _reset_run_state() 이후 _result_queue=None."""
        ss, _, _ = self._run({"type": "stopped", "step": 0})
        assert ss["_result_queue"] is None

    def test_experiments_dict_updated(self):
        """PRD 07 §6.2 step3: session_state["experiments"][exp_id] = record."""
        ss, _, _ = self._run({"type": "stopped", "step": 10})
        assert EXP_ID in ss["experiments"]
        assert ss["experiments"][EXP_ID]["status"] == "중단"

    def test_step_nonzero_appears_in_warning(self):
        """step≠0이면 경고 메시지에 step 수가 포함됨."""
        ss, _, _ = self._run({"type": "stopped", "step": 100})
        assert "100" in ss["_last_result"]["text"]

    def test_step_zero_no_step_suffix(self):
        """step=0이면 경고 메시지에 step 수 미포함."""
        ss, _, _ = self._run({"type": "stopped", "step": 0})
        # f"({step:,} step 완료 후 중단)" 패턴 없어야 함
        assert " step " not in ss["_last_result"]["text"]

    def test_empty_exp_id_does_not_raise(self):
        """current_exp_id='' 상태에서도 예외 없이 reset 완료."""
        ss = _make_ss()
        ss["current_exp_id"] = ""
        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value={}), \
             patch("tabs.tab3_training.append_experiment"), \
             patch.object(t3.st, "warning"):
            t3._handle_stopped({"type": "stopped", "step": 0})
        assert ss["current_run_status"] == "idle"

    def test_state_reset_even_when_append_raises(self):
        """append_experiment RuntimeError 발생해도 _reset_run_state()는 실행됨."""
        ss = _make_ss()
        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.append_experiment", side_effect=RuntimeError("I/O")), \
             patch.object(t3.st, "warning"):
            t3._handle_stopped({"type": "stopped", "step": 0})
        assert ss["current_run_status"] == "idle"


# ── TestDrainQueueStopHandling ────────────────────────────────────────────────

class TestDrainQueueStopHandling:
    """_drain_queue() — stopped 메시지 처리 및 break (PRD 06 §5.1, R-RACE-01)."""

    def _drain(self, messages: list[dict], ss: dict | None = None) -> dict:
        """메시지를 큐에 채운 뒤 _drain_queue() 실행."""
        ss = ss or _make_ss()
        q: queue.Queue = queue.Queue()
        for m in messages:
            q.put(m)
        ss["_result_queue"] = q
        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.append_experiment"), \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "error"), \
             patch.object(t3.st, "success"):
            t3._drain_queue()
        return ss

    def test_stopped_message_resets_status(self):
        """stopped → _handle_stopped → status="idle"."""
        ss = self._drain([{"type": "stopped", "step": 100}])
        assert ss["current_run_status"] == "idle"

    def test_breaks_after_first_terminal_message(self):
        """R-RACE-01: 첫 종료 메시지(stopped) 이후 큐 잔여 메시지 미처리."""
        ss = _make_ss()
        q: queue.Queue = queue.Queue()
        q.put({"type": "stopped", "step": 10})
        q.put({"type": "error", "exception": Exception("untouched"), "traceback": "tb"})
        ss["_result_queue"] = q

        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.append_experiment"), \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "error") as mock_err:
            t3._drain_queue()

        # error 핸들러 미호출 (두 번째 메시지는 드레인되지 않음)
        mock_err.assert_not_called()

    def test_progress_messages_processed_before_stopped(self):
        """progress 여러 개 처리 후 stopped에서 중단."""
        ss = self._drain([
            {"type": "progress", "step": 100, "total": 70000, "loss": 0.1, "elapsed": 5.0},
            {"type": "progress", "step": 200, "total": 70000, "loss": 0.09, "elapsed": 10.0},
            {"type": "stopped", "step": 200},
        ])
        assert ss["_progress"]["step"] == 200
        assert ss["current_run_status"] == "idle"

    def test_log_messages_appended_before_stopped(self):
        """log 메시지 처리 후 stopped에서 중단."""
        ss = self._drain([
            {"type": "log", "message": "학습 중..."},
            {"type": "stopped", "step": 5},
        ])
        assert len(ss["_log_lines"]) == 1
        assert "학습 중..." in ss["_log_lines"][0]

    def test_empty_queue_no_state_change(self):
        """큐가 비어 있으면 상태 변화 없음."""
        ss = _make_ss()
        ss["_result_queue"] = queue.Queue()
        with _SessionCtx(ss):
            t3._drain_queue()
        assert ss["current_run_status"] == "running"

    def test_none_queue_no_crash(self):
        """_result_queue=None이면 드레인 생략 (R-RACE-02 이후 재진입 방어)."""
        ss = _make_ss()
        ss["_result_queue"] = None
        with _SessionCtx(ss):
            t3._drain_queue()
        assert ss["current_run_status"] == "running"


# ── TestGuardOrphanThread ──────────────────────────────────────────────────────

class TestGuardOrphanThread:
    """_guard() 미아 스레드 감지 (PRD 06 §6.3)."""

    def test_orphan_thread_triggers_reset(self):
        """worker 살아있고 _result_queue=None → _reset_run_state() 호출."""
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        ss = _make_ss()
        ss["_worker"] = mock_worker
        ss["_result_queue"] = None  # 새로고침 후 Queue 참조 소멸 시뮬레이션
        with _SessionCtx(ss), \
             patch.object(t3.st, "info"), \
             patch.object(t3.st, "warning"):
            t3._guard()
        assert ss["current_run_status"] == "idle"

    def test_alive_worker_with_valid_queue_does_not_reset(self):
        """worker 살아있고 _result_queue 유효 → reset 없음."""
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        ss = _make_ss()
        ss["_worker"] = mock_worker
        # _result_queue는 _make_ss()에서 queue.Queue()로 설정됨
        with _SessionCtx(ss), \
             patch.object(t3.st, "info"), \
             patch.object(t3.st, "warning"):
            t3._guard()
        assert ss["current_run_status"] == "running"

    def test_dead_worker_does_not_trigger_reset(self):
        """종료된 worker는 orphan 아님 → reset 없음."""
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = False
        ss = _make_ss()
        ss["_worker"] = mock_worker
        ss["_result_queue"] = None
        with _SessionCtx(ss), \
             patch.object(t3.st, "info"), \
             patch.object(t3.st, "warning"):
            t3._guard()
        # 죽은 worker는 orphan 아님 → _reset_run_state 미호출
        assert ss["current_run_status"] == "running"


# ── TestGuardMissingState ──────────────────────────────────────────────────────

class TestGuardMissingState:
    """_guard() — 3개 선행 조건 누락 시 False 반환 및 경고 출력 (ADR-04, FR-CMN-03).

    Guard 체인: 탭2 guard(dataset_meta) → 탭3 guard(dataset_path + preprocessing_config + model_config)
    구 탭3(model_params) guard(preprocessing_config is None 단독 체크)는 제거됨.
    """

    def _run(self, ss: dict) -> tuple[bool, list[str]]:
        """_guard() 실행 후 (반환값, 수집된 경고 메시지 목록) 반환."""
        warnings: list[str] = []

        def fake_warning(msg: str, **_kw) -> None:
            warnings.append(msg)

        with _SessionCtx(ss), \
             patch.object(t3.st, "warning", side_effect=fake_warning), \
             patch.object(t3.st, "info"):
            result = t3._guard()
        return result, warnings

    def test_all_present_returns_true(self):
        """선행 조건 3개 모두 충족 시 True 반환."""
        ss = _make_ss(current_run_status="idle")
        result, _ = self._run(ss)
        assert result is True

    def test_dataset_path_none_returns_false(self):
        """dataset_path 누락 → False."""
        ss = _make_ss(current_run_status="idle")
        ss["dataset_path"] = None
        result, _ = self._run(ss)
        assert result is False

    def test_preprocessing_config_none_returns_false(self):
        """preprocessing_config 누락 → False."""
        ss = _make_ss(current_run_status="idle")
        ss["preprocessing_config"] = None
        result, _ = self._run(ss)
        assert result is False

    def test_model_config_none_returns_false(self):
        """model_config 누락 → False."""
        ss = _make_ss(current_run_status="idle")
        ss["model_config"] = None
        result, _ = self._run(ss)
        assert result is False

    def test_dataset_path_none_shows_no_dataset_msg(self):
        """MSG["NO_DATASET"] 경고가 출력돼야 한다."""
        ss = _make_ss(current_run_status="idle")
        ss["dataset_path"] = None
        _, warnings = self._run(ss)
        assert any(MSG["NO_DATASET"] in w for w in warnings)

    def test_preprocessing_config_none_shows_no_preprocessing_msg(self):
        """MSG["NO_PREPROCESSING"] 경고가 출력돼야 한다."""
        ss = _make_ss(current_run_status="idle")
        ss["preprocessing_config"] = None
        _, warnings = self._run(ss)
        assert any(MSG["NO_PREPROCESSING"] in w for w in warnings)

    def test_model_config_none_shows_tab2_message(self):
        """MSG["NO_MODEL_CONFIG"]는 '탭2에서' 문구 포함 (탭 번호 cascade 확인)."""
        ss = _make_ss(current_run_status="idle")
        ss["model_config"] = None
        _, warnings = self._run(ss)
        assert any(MSG["NO_MODEL_CONFIG"] in w for w in warnings)
        assert any("탭2" in w for w in warnings)

    def test_all_missing_returns_false_and_three_warnings(self):
        """3개 조건 모두 누락 시 False 반환 + 경고 3개 출력."""
        ss = _make_ss(current_run_status="idle")
        ss["dataset_path"] = None
        ss["preprocessing_config"] = None
        ss["model_config"] = None
        result, warnings = self._run(ss)
        assert result is False
        assert len(warnings) == 3


# ── TestRUi02 — 버튼 렌더링 분리 ─────────────────────────────────────────────

class TestRUi02:
    """R-UI-02: [학습 시작] 버튼은 idle UI에만, [학습 중지] 버튼은 running UI에만."""

    def _collect_buttons(self, fn, ss: dict) -> list[str]:
        """fn() 실행 중 st.button에 전달된 레이블 목록 수집."""
        labels: list[str] = []

        def fake_button(label: str, **_kw) -> bool:
            labels.append(label)
            return False

        with _SessionCtx(ss), \
             patch.object(t3.st, "button", side_effect=fake_button), \
             patch.object(t3.st, "info"), \
             patch.object(t3.st, "progress"), \
             patch.object(t3.st, "text_area"), \
             patch.object(t3.st, "text_input", return_value=""), \
             patch.object(t3.st, "markdown"):
            fn()
        return labels

    def test_running_ui_has_no_start_button(self):
        """running UI에 [학습 시작] 버튼 부재."""
        ss = _make_ss()
        labels = self._collect_buttons(t3._render_running_ui, ss)
        assert "학습 시작" not in labels

    def test_running_ui_has_stop_button(self):
        """running UI에 [학습 중지] 버튼 존재."""
        ss = _make_ss()
        labels = self._collect_buttons(t3._render_running_ui, ss)
        assert any("학습 중지" in label for label in labels)

    def test_idle_ui_has_start_button(self):
        """idle UI에 [학습 시작] 버튼 존재."""
        ss = _make_ss(current_run_status="idle")
        with patch("tabs.tab3_training._render_pretrain_summary"):
            labels = self._collect_buttons(t3._render_idle_ui, ss)
        assert "학습 시작" in labels

    def test_idle_ui_has_no_stop_button(self):
        """idle UI에 [학습 중지] 버튼 부재."""
        ss = _make_ss(current_run_status="idle")
        with patch("tabs.tab3_training._render_pretrain_summary"):
            labels = self._collect_buttons(t3._render_idle_ui, ss)
        assert "학습 중지" not in labels


# ── TestGenerateExperimentId ──────────────────────────────────────────────────

class TestGenerateExperimentId:
    """R-NAMING-03 experiment_id 형식 (PRD 07 §2.1)."""

    def test_efficientad_prefix(self):
        assert t3.generate_experiment_id("efficientad").startswith("efficientad_")

    def test_patchcore_prefix(self):
        assert t3.generate_experiment_id("patchcore").startswith("patchcore_")

    def test_four_part_format(self):
        eid = t3.generate_experiment_id("efficientad")
        parts = eid.split("_")
        # "efficientad" + YYYYMMDD + HHMMSS + rand4
        assert len(parts) == 4

    def test_date_part_length_8(self):
        eid = t3.generate_experiment_id("efficientad")
        assert len(eid.split("_")[1]) == 8

    def test_time_part_length_6(self):
        eid = t3.generate_experiment_id("efficientad")
        assert len(eid.split("_")[2]) == 6

    def test_rand_part_is_4_char_lowercase_hex(self):
        """R-ID-01: uuid4().hex[:4] → 소문자 16진수 4자리."""
        for _ in range(10):
            eid = t3.generate_experiment_id("efficientad")
            rand_part = eid.split("_")[3]
            assert re.fullmatch(r"[0-9a-f]{4}", rand_part), f"rand_part={rand_part!r}"

    def test_uniqueness_across_calls(self):
        """10회 호출 중 최소 절반 이상 고유 (uuid4 덕분에 거의 항상)."""
        ids = [t3.generate_experiment_id("efficientad") for _ in range(10)]
        assert len(set(ids)) >= 5


# ── TestHandleCompleted ────────────────────────────────────────────────────────

class TestHandleCompleted:
    """_handle_completed() — PRD 07 §6.1 완료 후처리 6단계 검증."""

    _METRICS = {
        "auc": 0.95,
        "accuracy": 0.9,
        "precision": 0.9,
        "recall": 0.9,
        "f1_score": 0.9,
        "f2_score": 0.9,
        "confusion_matrix": {"tp": 9, "fp": 1, "tn": 9, "fn": 1},
        "anomaly_scores": [0.1, 0.2, 0.8, 0.9],
        "image_labels": [0, 0, 1, 1],
    }

    def _make_msg(self, with_anomaly_maps: bool = True) -> dict:
        msg: dict = {
            "type": "completed",
            "y_true": [0, 0, 1, 1],
            "anomaly_scores": [0.1, 0.2, 0.8, 0.9],
            "duration_seconds": 60,
            "model": MagicMock(),
        }
        if with_anomaly_maps:
            img_path = "/data/test/crack/000.png"
            msg["anomaly_maps"] = {img_path: np.zeros((64, 64), dtype=np.float32)}
            msg["image_paths"] = [img_path]
        else:
            msg["anomaly_maps"] = {}
            msg["image_paths"] = []
        return msg

    def _run(self, msg: dict, ss: dict | None = None, raise_on_save: Exception | None = None):
        ss = ss or _make_ss()
        save_side = raise_on_save
        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.compute_threshold", return_value=0.5) as mock_thresh, \
             patch("tabs.tab3_training.compute_metrics", return_value=self._METRICS) as mock_metrics, \
             patch("tabs.tab3_training.check_disk_before_save"), \
             patch("tabs.tab3_training.save_completed_experiment",
                   side_effect=save_side) as mock_save, \
             patch("tabs.tab3_training.set_anomaly_map_cache") as mock_cache, \
             patch.object(t3.st, "success") as mock_success, \
             patch.object(t3.st, "warning") as mock_warn, \
             patch.object(t3.st, "error") as mock_err:
            t3._handle_completed(msg)
        return ss, mock_thresh, mock_metrics, mock_save, mock_cache, mock_success, mock_warn, mock_err

    # ── step 1/2: threshold + metrics ─────────────────────────────────────────

    def test_compute_threshold_called(self):
        """PRD 07 §6.1 step1: compute_threshold() 호출."""
        _, mock_thresh, *_ = self._run(self._make_msg())
        mock_thresh.assert_called_once()

    def test_compute_metrics_called(self):
        """PRD 07 §6.1 step2: compute_metrics() 호출."""
        _, _, mock_metrics, *_ = self._run(self._make_msg())
        mock_metrics.assert_called_once()

    def test_normal_scores_filtered_for_threshold(self):
        """threshold 계산에 label=0 이미지 score만 전달되는지 확인."""
        _, mock_thresh, *_ = self._run(self._make_msg())
        args = mock_thresh.call_args[0]
        normal_scores_arg = args[0]
        # y_true=[0,0,1,1], anomaly_scores=[0.1,0.2,0.8,0.9] → 정상=[0.1,0.2]
        assert len(normal_scores_arg) == 2

    # ── step 5: save ───────────────────────────────────────────────────────────

    def test_save_completed_experiment_called(self):
        """PRD 07 §6.1 step5: save_completed_experiment() 호출."""
        _, _, _, mock_save, *_ = self._run(self._make_msg())
        mock_save.assert_called_once()

    def test_save_called_with_correct_exp_id(self):
        _, _, _, mock_save, *_ = self._run(self._make_msg())
        call_args = mock_save.call_args[0]
        assert call_args[0] == EXP_ID

    # ── step 6: session_state 갱신 ────────────────────────────────────────────

    def test_session_state_experiments_updated(self):
        """PRD 07 §6.1 step6: session_state["experiments"][exp_id] = record."""
        ss, *_ = self._run(self._make_msg())
        assert EXP_ID in ss["experiments"]

    def test_experiment_record_status_completed(self):
        ss, *_ = self._run(self._make_msg())
        assert ss["experiments"][EXP_ID]["status"] == "completed"

    # ── anomaly_map 캐시 ───────────────────────────────────────────────────────

    def test_cache_set_when_maps_present(self):
        """anomaly_maps + image_paths 있으면 set_anomaly_map_cache 호출."""
        _, _, _, _, mock_cache, *_ = self._run(self._make_msg(with_anomaly_maps=True))
        mock_cache.assert_called_once()

    def test_cache_not_set_when_maps_absent(self):
        """anomaly_maps 없으면 set_anomaly_map_cache 미호출."""
        _, _, _, _, mock_cache, *_ = self._run(self._make_msg(with_anomaly_maps=False))
        mock_cache.assert_not_called()

    def test_cache_call_includes_image_paths(self):
        """캐시 데이터에 image_paths 포함."""
        _, _, _, _, mock_cache, *_ = self._run(self._make_msg(with_anomaly_maps=True))
        cache_data = mock_cache.call_args[0][1]
        assert "image_paths" in cache_data
        assert len(cache_data["image_paths"]) == 1

    # ── step 7: st.success ────────────────────────────────────────────────────

    def test_st_success_called_on_normal_completion(self):
        """PRD 07 §6.1 step7: _last_result level=success 설정."""
        ss, *_ = self._run(self._make_msg())
        assert ss["_last_result"]["level"] == "success"

    def test_success_message_contains_auc(self):
        """성공 메시지에 AUC 값 포함 (탭3 UI 알림 조건 — PRD 7.4절)."""
        ss, *_ = self._run(self._make_msg())
        assert "0.9500" in ss["_last_result"]["text"]

    # ── step 8: _reset_run_state (finally 보장) ───────────────────────────────

    def test_status_reset_to_idle_on_success(self):
        """PRD 07 §6.1 step8: 정상 완료 후 current_run_status="idle"."""
        ss, *_ = self._run(self._make_msg())
        assert ss["current_run_status"] == "idle"

    def test_status_reset_to_idle_even_on_save_error(self):
        """RuntimeError 발생해도 finally 블록에서 _reset_run_state() 실행."""
        ss, *_ = self._run(
            self._make_msg(),
            raise_on_save=RuntimeError("ERR_MODEL_SAVE_FAILED (Stage1): disk full"),
        )
        assert ss["current_run_status"] == "idle"

    def test_result_queue_none_after_completed(self):
        """R-RACE-02: _reset_run_state() 이후 _result_queue=None."""
        ss, *_ = self._run(self._make_msg())
        assert ss["_result_queue"] is None

    def test_history_write_fail_shows_warning_not_error(self):
        """ERR_HISTORY_WRITE_FAILED → _last_result level=warning."""
        ss, *_ = self._run(
            self._make_msg(),
            raise_on_save=RuntimeError("ERR_HISTORY_WRITE_FAILED: details"),
        )
        assert ss["_last_result"]["level"] == "warning"

    def test_other_runtime_error_shows_error(self):
        """Stage1/2 실패 → _last_result level=error."""
        ss, *_ = self._run(
            self._make_msg(),
            raise_on_save=RuntimeError("ERR_MODEL_SAVE_FAILED (Stage1): disk full"),
        )
        assert ss["_last_result"]["level"] == "error"

    # ── PRD 07 §9.2 GPU 메모리 해제 ──────────────────────────────────────────

    def test_cuda_empty_cache_called_only_for_cuda_device(self):
        """device=="cuda"인 경우만 torch.cuda.empty_cache() 호출."""
        ss = _make_ss()
        ss["device_info"] = {"device": "cuda"}
        with patch("tabs.tab3_training.torch") as mock_torch, \
             _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.compute_threshold", return_value=0.5), \
             patch("tabs.tab3_training.compute_metrics", return_value=self._METRICS), \
             patch("tabs.tab3_training.check_disk_before_save"), \
             patch("tabs.tab3_training.save_completed_experiment"), \
             patch("tabs.tab3_training.set_anomaly_map_cache"), \
             patch.object(t3.st, "success"), \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "error"):
            t3._handle_completed(self._make_msg())
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_cuda_empty_cache_not_called_for_cpu_device(self):
        """device=="cpu"이면 torch.cuda.empty_cache() 미호출."""
        ss = _make_ss()
        ss["device_info"] = {"device": "cpu"}
        with patch("tabs.tab3_training.torch") as mock_torch, \
             _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.compute_threshold", return_value=0.5), \
             patch("tabs.tab3_training.compute_metrics", return_value=self._METRICS), \
             patch("tabs.tab3_training.check_disk_before_save"), \
             patch("tabs.tab3_training.save_completed_experiment"), \
             patch("tabs.tab3_training.set_anomaly_map_cache"), \
             patch.object(t3.st, "success"), \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "error"):
            t3._handle_completed(self._make_msg())
        mock_torch.cuda.empty_cache.assert_not_called()


# ── TestHandleError ───────────────────────────────────────────────────────────

class TestHandleError:
    """_handle_error() — 오류 핸들러 검증."""

    def _run(self, msg: dict, ss: dict | None = None):
        ss = ss or _make_ss()
        with _SessionCtx(ss), \
             patch.object(t3.st, "error") as mock_err:
            t3._handle_error(msg)
        return ss, mock_err

    def test_st_error_called_once(self):
        """오류 수신 시 _last_result level=error 설정."""
        ss, _ = self._run({
            "type": "error",
            "exception": RuntimeError("CUDA OOM"),
            "traceback": "Traceback ...\nRuntimeError: CUDA OOM",
        })
        assert ss["_last_result"]["level"] == "error"

    def test_error_message_contains_traceback(self):
        """오류 메시지에 traceback 일부 포함."""
        ss, _ = self._run({
            "type": "error",
            "exception": RuntimeError("CUDA OOM"),
            "traceback": "Traceback ...\nRuntimeError: CUDA OOM",
        })
        assert "CUDA OOM" in ss["_last_result"]["text"]

    def test_status_reset_to_idle(self):
        """오류 후 current_run_status="idle"."""
        ss, _ = self._run({
            "type": "error",
            "exception": ValueError("bad model"),
            "traceback": "ValueError: bad model",
        })
        assert ss["current_run_status"] == "idle"

    def test_result_queue_none_after_error(self):
        """R-RACE-02: 오류 후 _result_queue=None."""
        ss, _ = self._run({
            "type": "error",
            "exception": Exception("fail"),
            "traceback": "Exception: fail",
        })
        assert ss["_result_queue"] is None

    def test_empty_traceback_does_not_raise(self):
        """traceback 없는 메시지도 처리 가능."""
        ss, _ = self._run({"type": "error", "exception": Exception("x"), "traceback": ""})
        assert ss["current_run_status"] == "idle"


# ── TestHandleStage ────────────────────────────────────────────────────────────

class TestHandleStage:
    """_handle_stage() — stage 메시지 처리 (FR-T3-11/12)."""

    def test_sets_stage_idx(self):
        ss = _make_ss()
        with _SessionCtx(ss):
            t3._handle_stage({"type": "stage", "stage_idx": 2, "stage_name": "학습 루프"})
        assert ss["_current_stage_idx"] == 2

    def test_sets_stage_name(self):
        ss = _make_ss()
        with _SessionCtx(ss):
            t3._handle_stage({"type": "stage", "stage_idx": 2, "stage_name": "학습 루프"})
        assert ss["_current_stage_name"] == "학습 루프"

    def test_overwrites_previous_stage(self):
        ss = _make_ss()
        ss["_current_stage_idx"] = 1
        ss["_current_stage_name"] = "모델 초기화"
        with _SessionCtx(ss):
            t3._handle_stage({"type": "stage", "stage_idx": 3, "stage_name": "테스트 추론"})
        assert ss["_current_stage_idx"] == 3
        assert ss["_current_stage_name"] == "테스트 추론"

    def test_stage_zero_sets_idx_zero(self):
        ss = _make_ss()
        with _SessionCtx(ss):
            t3._handle_stage({"type": "stage", "stage_idx": 0, "stage_name": "데이터 로딩"})
        assert ss["_current_stage_idx"] == 0


# ── TestStageConstants ─────────────────────────────────────────────────────────

class TestStageConstants:
    """EFFICIENTAD_STAGES / PATCHCORE_STAGES 상수 검증 (FR-T3-11/12)."""

    def test_efficientad_has_5_stages(self):
        assert len(t3.EFFICIENTAD_STAGES) == 5

    def test_patchcore_has_7_stages(self):
        assert len(t3.PATCHCORE_STAGES) == 7

    def test_efficientad_stage_indices_sequential(self):
        indices = [idx for idx, _ in t3.EFFICIENTAD_STAGES]
        assert indices == list(range(5))

    def test_patchcore_stage_indices_sequential(self):
        indices = [idx for idx, _ in t3.PATCHCORE_STAGES]
        assert indices == list(range(7))

    def test_efficientad_last_stage_name(self):
        assert t3.EFFICIENTAD_STAGES[-1][1] == "완료"

    def test_patchcore_last_stage_name(self):
        assert t3.PATCHCORE_STAGES[-1][1] == "완료"

    def test_efficientad_includes_training_loop_stage(self):
        names = [name for _, name in t3.EFFICIENTAD_STAGES]
        assert "학습 루프" in names

    def test_patchcore_includes_coreset_stage(self):
        names = [name for _, name in t3.PATCHCORE_STAGES]
        assert "Coreset 구성" in names


# ── TestRenderStageIndicator ───────────────────────────────────────────────────

class TestRenderStageIndicator:
    """_render_stage_indicator() — HTML 인디케이터 렌더링 (FR-T3-11/12)."""

    def _run(self, ss: dict) -> str:
        """_render_stage_indicator() 실행 후 st.markdown에 전달된 HTML 반환."""
        captured: list[str] = []

        def fake_markdown(html: str, **kw) -> None:
            captured.append(html)

        with _SessionCtx(ss), patch.object(t3.st, "markdown", side_effect=fake_markdown):
            t3._render_stage_indicator()

        return captured[0] if captured else ""

    def _ss(self, model_type: str = "efficientad", stage_idx: int | None = None) -> dict:
        ss = _make_ss()
        ss["model_config"]["model_type"] = model_type
        ss["_current_stage_idx"] = stage_idx
        return ss

    def test_all_pending_when_stage_idx_none(self):
        """stage_idx=None → EfficientAD 5단계 모두 ○ 표시."""
        html = self._run(self._ss(stage_idx=None))
        assert html.count("○") == 5

    def test_current_stage_shows_blue_dot(self):
        """현재 단계 → 🔵 표시."""
        html = self._run(self._ss(stage_idx=2))
        assert "🔵" in html

    def test_completed_stages_show_checkmark(self):
        """stage_idx=3 → 이전 단계(0,1,2) ✅ 3개."""
        html = self._run(self._ss(stage_idx=3))
        assert html.count("✅") == 3

    def test_pending_count_after_current(self):
        """stage_idx=2 → 이후 단계(3,4) ○ 2개."""
        html = self._run(self._ss(stage_idx=2))
        assert html.count("○") == 2

    def test_patchcore_pending_count(self):
        """PatchCore stage_idx=0 → 이후 단계 ○ 6개."""
        html = self._run(self._ss(model_type="patchcore", stage_idx=0))
        assert html.count("○") == 6

    def test_final_stage_has_no_pending(self):
        """EfficientAD stage_idx=4(마지막) → ○ 없음, 🔵 1개(완료 현재), ✅ 4개."""
        html = self._run(self._ss(stage_idx=4))
        assert "○" not in html
        assert "🔵" in html
        assert html.count("✅") == 4

    def test_markdown_called_with_unsafe_html(self):
        """unsafe_allow_html=True 로 호출되어야 한다."""
        called_kwargs: list[dict] = []

        def fake_markdown(html: str, **kw) -> None:
            called_kwargs.append(kw)

        ss = self._ss(stage_idx=1)
        with _SessionCtx(ss), patch.object(t3.st, "markdown", side_effect=fake_markdown):
            t3._render_stage_indicator()

        assert called_kwargs and called_kwargs[0].get("unsafe_allow_html") is True

    def test_stage_name_appears_in_html(self):
        """단계 이름이 HTML에 포함되어야 한다."""
        html = self._run(self._ss(stage_idx=2))
        assert "학습 루프" in html


# ── TestDrainQueueStageHandling ────────────────────────────────────────────────

class TestDrainQueueStageHandling:
    """_drain_queue() stage 메시지 비종료(non-terminal) 처리 (FR-T3-11/12)."""

    def _drain_stage_only(self, stage_idx: int, stage_name: str) -> dict:
        """stage 메시지 1개만 큐에 넣고 drain (stopped 없음 → Empty에서 break)."""
        ss = _make_ss()
        q: queue.Queue = queue.Queue()
        q.put({"type": "stage", "stage_idx": stage_idx, "stage_name": stage_name})
        ss["_result_queue"] = q
        with _SessionCtx(ss):
            t3._drain_queue()
        return ss

    def test_stage_message_updates_stage_idx(self):
        """stage 메시지 → _current_stage_idx 갱신."""
        ss = self._drain_stage_only(2, "학습 루프")
        assert ss["_current_stage_idx"] == 2

    def test_stage_message_updates_stage_name(self):
        """stage 메시지 → _current_stage_name 갱신."""
        ss = self._drain_stage_only(2, "학습 루프")
        assert ss["_current_stage_name"] == "학습 루프"

    def test_stage_message_does_not_terminate_drain(self):
        """stage → 종료 메시지 아님 → 이후 progress도 처리됨."""
        ss = _make_ss()
        q: queue.Queue = queue.Queue()
        q.put({"type": "stage", "stage_idx": 0, "stage_name": "데이터 로딩"})
        q.put({"type": "progress", "step": 100, "total": 70000, "loss": 0.1, "elapsed": 5.0})
        ss["_result_queue"] = q
        with _SessionCtx(ss):
            t3._drain_queue()
        assert ss["_current_stage_idx"] == 0
        assert ss["_progress"]["step"] == 100

    def test_status_unchanged_after_stage_only(self):
        """stage 메시지만으로 current_run_status 변화 없음."""
        ss = self._drain_stage_only(1, "모델 초기화")
        assert ss["current_run_status"] == "running"


# ── TestComputeEta ─────────────────────────────────────────────────────────────

class TestComputeEta:
    """_compute_eta() — ETA 계산 (FR-T3-13)."""

    # ── 비-루프 단계 → None ────────────────────────────────────────────────────

    def test_none_for_none_stage(self):
        """stage_name=None → None."""
        assert t3._compute_eta(None, 1000, 70000, 10.0, "efficientad") is None

    def test_none_for_data_loading_stage(self):
        """데이터 로딩 → None."""
        assert t3._compute_eta("데이터 로딩", 0, 70000, 5.0, "efficientad") is None

    def test_none_for_model_init_stage(self):
        """모델 초기화 → None."""
        assert t3._compute_eta("모델 초기화", 0, 70000, 10.0, "efficientad") is None

    def test_none_for_test_inference_stage(self):
        """테스트 추론 → None."""
        assert t3._compute_eta("테스트 추론", 1, 1, 30.0, "efficientad") is None

    def test_none_for_coreset_stage(self):
        """Coreset 구성 → None."""
        assert t3._compute_eta("Coreset 구성", 1, 1, 5.0, "patchcore") is None

    def test_none_for_memory_bank_stage(self):
        """Memory Bank 설정 → None."""
        assert t3._compute_eta("Memory Bank 설정", 1, 1, 3.0, "patchcore") is None

    def test_none_for_done_stage(self):
        """완료 → None."""
        assert t3._compute_eta("완료", 1, 1, 600.0, "efficientad") is None

    # ── EfficientAD 루프 단계 ──────────────────────────────────────────────────

    def test_efficientad_none_before_100_steps(self):
        """EfficientAD step=99 → None (최소 100 step 미만)."""
        assert t3._compute_eta("학습 루프", 99, 70000, 10.0, "efficientad") is None

    def test_efficientad_none_at_0_steps(self):
        """EfficientAD step=0 → None."""
        assert t3._compute_eta("학습 루프", 0, 70000, 10.0, "efficientad") is None

    def test_efficientad_calculable_at_100_steps(self):
        """EfficientAD step=100 → ETA 반환 시작."""
        eta = t3._compute_eta("학습 루프", 100, 70000, 10.0, "efficientad")
        assert eta is not None

    def test_efficientad_calculable_above_100_steps(self):
        """EfficientAD step>100 → ETA 반환."""
        eta = t3._compute_eta("학습 루프", 1000, 70000, 10.0, "efficientad")
        assert eta is not None

    def test_efficientad_eta_value_correct(self):
        """ETA 계산: elapsed/step * remaining = 10/1000*69000 = 690s."""
        eta = t3._compute_eta("학습 루프", 1000, 70000, 10.0, "efficientad")
        assert eta == "690s"

    # ── PatchCore 루프 단계 ────────────────────────────────────────────────────

    def test_patchcore_calculable_from_first_batch(self):
        """PatchCore step=1 → ETA 반환 (100 step 제한 없음)."""
        eta = t3._compute_eta("특징 추출", 1, 100, 2.0, "patchcore")
        assert eta is not None

    def test_patchcore_eta_value_correct(self):
        """PatchCore ETA: 2/1*99 = 198s."""
        eta = t3._compute_eta("특징 추출", 1, 100, 2.0, "patchcore")
        assert eta == "198s"

    # ── 경계 조건 ──────────────────────────────────────────────────────────────

    def test_none_when_step_zero(self):
        """step=0 → None (0 나누기 방어)."""
        assert t3._compute_eta("학습 루프", 0, 70000, 10.0, "efficientad") is None

    def test_none_when_elapsed_zero(self):
        """elapsed=0.0 → None."""
        assert t3._compute_eta("학습 루프", 1000, 70000, 0.0, "efficientad") is None

    def test_none_when_all_steps_complete(self):
        """remaining=0 (step==total) → None."""
        assert t3._compute_eta("학습 루프", 70000, 70000, 600.0, "efficientad") is None

    def test_returns_string(self):
        """반환값 타입: str."""
        eta = t3._compute_eta("학습 루프", 1000, 70000, 10.0, "efficientad")
        assert isinstance(eta, str)

    def test_format_ends_with_s(self):
        """반환 형식: '{N}s'."""
        eta = t3._compute_eta("학습 루프", 1000, 70000, 10.0, "efficientad")
        assert eta is not None and eta.endswith("s")


# ── TestFullStageSequenceDrainQueue ───────────────────────────────────────────

class TestFullStageSequenceDrainQueue:
    """EfficientAD/PatchCore 전체 단계 시퀀스가 drain_queue를 통해 올바르게
    처리되는지 통합 검증 (FR-T3-11/12, FR-T3-13).

    실제 TrainingWorker 없이 큐에 메시지를 직접 주입하여
    각 단계 전환 시 session_state가 올바르게 갱신되는지 확인한다.
    """

    def _drain(self, messages: list[dict], ss: dict) -> None:
        """메시지를 큐에 넣고 _drain_queue() 실행 (stopped 종료 메시지 전용 헬퍼)."""
        q: queue.Queue = queue.Queue()
        for m in messages:
            q.put(m)
        ss["_result_queue"] = q
        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.append_experiment"):
            t3._drain_queue()

    # ── EfficientAD 단계 시퀀스 ────────────────────────────────────────────────

    def test_efficientad_each_stage_updates_session(self):
        """EfficientAD 각 단계(0~4)가 큐를 통해 처리되면 세션에 정확히 반영된다."""
        from utils.training_worker import EFFICIENTAD_STAGES

        for expected_idx, expected_name in EFFICIENTAD_STAGES:
            ss = _make_ss()
            q: queue.Queue = queue.Queue()
            q.put({"type": "stage", "stage_idx": expected_idx, "stage_name": expected_name})
            ss["_result_queue"] = q
            with _SessionCtx(ss):
                t3._drain_queue()
            assert ss["_current_stage_idx"] == expected_idx, \
                f"stage {expected_idx} '{expected_name}': _current_stage_idx 불일치"
            assert ss["_current_stage_name"] == expected_name, \
                f"stage {expected_idx}: _current_stage_name 불일치"

    def test_efficientad_stage_progression_0_to_2(self):
        """EfficientAD stage 0→1→2가 순서대로 처리될 때 마지막 단계가 세션에 반영된다."""
        ss = _make_ss()
        q: queue.Queue = queue.Queue()
        for msg in [
            {"type": "stage", "stage_idx": 0, "stage_name": "데이터 로딩"},
            {"type": "stage", "stage_idx": 1, "stage_name": "모델 초기화"},
            {"type": "stage", "stage_idx": 2, "stage_name": "학습 루프"},
        ]:
            q.put(msg)
        ss["_result_queue"] = q
        with _SessionCtx(ss):
            t3._drain_queue()
        assert ss["_current_stage_idx"] == 2
        assert ss["_current_stage_name"] == "학습 루프"
        assert ss["current_run_status"] == "running"  # 종료 없음 → 상태 유지

    def test_efficientad_loop_stage_with_enough_steps_has_eta(self):
        """EfficientAD '학습 루프' + step ≥ 100 → ETA 계산 가능 (FR-T3-13)."""
        eta = t3._compute_eta("학습 루프", 1000, 70000, 10.0, "efficientad")
        assert eta is not None
        assert eta == "690s"

    def test_efficientad_loop_stage_under_100_no_eta(self):
        """EfficientAD '학습 루프' + step < 100 → ETA None (신뢰도 미확보)."""
        assert t3._compute_eta("학습 루프", 50, 70000, 5.0, "efficientad") is None

    def test_efficientad_non_loop_stages_no_eta(self):
        """EfficientAD 비-루프 단계 4종 → ETA None."""
        for stage in ["데이터 로딩", "모델 초기화", "테스트 추론", "완료"]:
            assert t3._compute_eta(stage, 1000, 70000, 10.0, "efficientad") is None, \
                f"{stage}: ETA가 None이어야 하는데 값이 있음"

    def test_efficientad_complete_sequence_stopped(self):
        """EfficientAD 0→1→2 + progress + stopped → 최종 상태 idle, stage reset."""
        ss = _make_ss()
        self._drain([
            {"type": "stage",    "stage_idx": 0, "stage_name": "데이터 로딩"},
            {"type": "stage",    "stage_idx": 1, "stage_name": "모델 초기화"},
            {"type": "stage",    "stage_idx": 2, "stage_name": "학습 루프"},
            {"type": "progress", "step": 35000, "total": 70000, "loss": 0.02, "elapsed": 300.0},
            {"type": "stage",    "stage_idx": 3, "stage_name": "테스트 추론"},
            {"type": "stopped",  "step": 35000},
        ], ss)
        assert ss["current_run_status"] == "idle"
        assert ss["_current_stage_idx"] is None   # _reset_run_state 호출됨
        assert ss["_current_stage_name"] is None

    # ── PatchCore 단계 시퀀스 ──────────────────────────────────────────────────

    def test_patchcore_each_stage_updates_session(self):
        """PatchCore 각 단계(0~6)가 큐를 통해 처리되면 세션에 정확히 반영된다."""
        from utils.training_worker import PATCHCORE_STAGES

        for expected_idx, expected_name in PATCHCORE_STAGES:
            ss = _make_ss()
            q: queue.Queue = queue.Queue()
            q.put({"type": "stage", "stage_idx": expected_idx, "stage_name": expected_name})
            ss["_result_queue"] = q
            with _SessionCtx(ss):
                t3._drain_queue()
            assert ss["_current_stage_idx"] == expected_idx, \
                f"PatchCore stage {expected_idx} '{expected_name}': _current_stage_idx 불일치"
            assert ss["_current_stage_name"] == expected_name

    def test_patchcore_feature_extraction_eta(self):
        """PatchCore '특징 추출' + step ≥ 1 → ETA 계산 가능 (100 step 제한 없음)."""
        eta = t3._compute_eta("특징 추출", 5, 50, 10.0, "patchcore")
        assert eta is not None
        assert eta == "90s"  # 10/5 * (50-5) = 90

    def test_patchcore_non_feature_stages_no_eta(self):
        """PatchCore '특징 추출' 외 단계 → ETA None."""
        for stage in ["데이터 로딩", "모델 초기화", "Coreset 구성",
                      "Memory Bank 설정", "테스트 추론", "완료"]:
            assert t3._compute_eta(stage, 5, 50, 10.0, "patchcore") is None, \
                f"PatchCore {stage}: ETA가 None이어야 하는데 값이 있음"

    def test_patchcore_complete_sequence_stopped(self):
        """PatchCore 0→1→2→3→4 + stopped → 최종 상태 idle, stage reset."""
        ss = _make_ss()
        ss["model_config"]["model_type"] = "patchcore"
        self._drain([
            {"type": "stage",    "stage_idx": 0, "stage_name": "데이터 로딩"},
            {"type": "stage",    "stage_idx": 1, "stage_name": "모델 초기화"},
            {"type": "stage",    "stage_idx": 2, "stage_name": "특징 추출"},
            {"type": "progress", "step": 25, "total": 50, "loss": 0.0, "elapsed": 12.0},
            {"type": "stage",    "stage_idx": 3, "stage_name": "Coreset 구성"},
            {"type": "stage",    "stage_idx": 4, "stage_name": "Memory Bank 설정"},
            {"type": "stopped",  "step": 50},
        ], ss)
        assert ss["current_run_status"] == "idle"
        assert ss["_current_stage_idx"] is None
        assert ss["_current_stage_name"] is None

    # ── 공통 통합 시나리오 ─────────────────────────────────────────────────────

    def test_stage_and_progress_interleaved(self):
        """stage + progress 메시지가 섞여 있을 때 둘 다 올바르게 처리된다."""
        ss = _make_ss()
        q: queue.Queue = queue.Queue()
        for msg in [
            {"type": "stage",    "stage_idx": 2, "stage_name": "학습 루프"},
            {"type": "progress", "step": 100,  "total": 70000, "loss": 0.05, "elapsed": 10.0},
            {"type": "progress", "step": 200,  "total": 70000, "loss": 0.04, "elapsed": 20.0},
        ]:
            q.put(msg)
        ss["_result_queue"] = q
        with _SessionCtx(ss):
            t3._drain_queue()
        assert ss["_current_stage_idx"] == 2
        assert ss["_current_stage_name"] == "학습 루프"
        assert ss["_progress"]["step"] == 200
        assert ss["_progress"]["loss"] == pytest.approx(0.04)

    def test_progress_does_not_overwrite_stage(self):
        """progress 메시지는 _current_stage_idx / _current_stage_name을 변경하지 않는다."""
        ss = _make_ss()
        ss["_current_stage_idx"]  = 2
        ss["_current_stage_name"] = "학습 루프"
        q: queue.Queue = queue.Queue()
        q.put({"type": "progress", "step": 500, "total": 70000, "loss": 0.03, "elapsed": 50.0})
        ss["_result_queue"] = q
        with _SessionCtx(ss):
            t3._drain_queue()
        assert ss["_current_stage_idx"] == 2
        assert ss["_current_stage_name"] == "학습 루프"

    def test_stage_sequence_followed_by_log_messages(self):
        """stage → log 메시지 순서도 올바르게 처리된다."""
        ss = _make_ss()
        q: queue.Queue = queue.Queue()
        for msg in [
            {"type": "stage", "stage_idx": 0, "stage_name": "데이터 로딩"},
            {"type": "log",   "message": "[초기화] 데이터셋 로딩 중..."},
            {"type": "stage", "stage_idx": 1, "stage_name": "모델 초기화"},
            {"type": "log",   "message": "[초기화] EfficientAD 모델 준비 완료"},
        ]:
            q.put(msg)
        ss["_result_queue"] = q
        with _SessionCtx(ss):
            t3._drain_queue()
        assert ss["_current_stage_idx"] == 1
        assert len(ss["_log_lines"]) == 2


# ── TestQueueSessionKeys ───────────────────────────────────────────────────────

class TestQueueSessionKeys:
    """experiment_queue / _batch_queue_mode / _batch_total_count 키 스키마 및
    init_session_state() 초기화 검증 (FR-T2-16, FR-T3-15)."""

    # ── 스키마 기본값 ──────────────────────────────────────────────────────────

    def test_queue_schema_default_is_empty_list(self):
        """experiment_queue 기본값: 빈 리스트."""
        from utils.session_state_init import SESSION_STATE_SCHEMA
        assert SESSION_STATE_SCHEMA["experiment_queue"] == []
        assert isinstance(SESSION_STATE_SCHEMA["experiment_queue"], list)

    def test_batch_mode_schema_default_is_false(self):
        """_batch_queue_mode 기본값: False."""
        from utils.session_state_init import SESSION_STATE_SCHEMA
        assert SESSION_STATE_SCHEMA["_batch_queue_mode"] is False

    def test_batch_total_count_schema_default_is_zero(self):
        """_batch_total_count 기본값: 0."""
        from utils.session_state_init import SESSION_STATE_SCHEMA
        assert SESSION_STATE_SCHEMA["_batch_total_count"] == 0

    def test_queue_status_values_documented(self):
        """experiment_queue 항목의 status 값은 5종이어야 한다."""
        valid_statuses = {"대기중", "진행중", "완료", "실패", "건너뜀"}
        item = {
            "name": "test_exp",
            "preprocessing_config": {},
            "model_config": {},
            "status": "대기중",
        }
        assert item["status"] in valid_statuses

    # ── init_session_state() 초기화 ────────────────────────────────────────────

    def test_queue_key_initialized(self, monkeypatch):
        """init_session_state() → experiment_queue 키 생성 및 빈 리스트."""
        import streamlit as _st
        state: dict = {}
        monkeypatch.setattr(_st, "session_state", state)
        from utils.session_state_init import init_session_state
        init_session_state()
        assert "experiment_queue" in state
        assert state["experiment_queue"] == []

    def test_batch_mode_key_initialized(self, monkeypatch):
        """init_session_state() → _batch_queue_mode = False."""
        import streamlit as _st
        state: dict = {}
        monkeypatch.setattr(_st, "session_state", state)
        from utils.session_state_init import init_session_state
        init_session_state()
        assert "_batch_queue_mode" in state
        assert state["_batch_queue_mode"] is False

    def test_batch_total_count_key_initialized(self, monkeypatch):
        """init_session_state() → _batch_total_count = 0."""
        import streamlit as _st
        state: dict = {}
        monkeypatch.setattr(_st, "session_state", state)
        from utils.session_state_init import init_session_state
        init_session_state()
        assert "_batch_total_count" in state
        assert state["_batch_total_count"] == 0

    # ── 멱등성 (idempotency) ───────────────────────────────────────────────────

    def test_existing_queue_not_overwritten(self, monkeypatch):
        """experiment_queue가 이미 설정되어 있으면 덮어쓰지 않는다."""
        import streamlit as _st
        existing_queue = [{"name": "exp1", "status": "대기중"}]
        state: dict = {"experiment_queue": existing_queue}
        monkeypatch.setattr(_st, "session_state", state)
        from utils.session_state_init import init_session_state
        init_session_state()
        assert state["experiment_queue"] is existing_queue
        assert len(state["experiment_queue"]) == 1

    def test_active_batch_mode_not_overwritten(self, monkeypatch):
        """_batch_queue_mode=True 상태는 init으로 덮어쓰지 않는다."""
        import streamlit as _st
        state: dict = {"_batch_queue_mode": True}
        monkeypatch.setattr(_st, "session_state", state)
        from utils.session_state_init import init_session_state
        init_session_state()
        assert state["_batch_queue_mode"] is True

    # ── _make_ss() 반영 확인 ───────────────────────────────────────────────────

    def test_make_ss_includes_queue_key(self):
        """_make_ss()가 experiment_queue 키를 포함한다."""
        ss = _make_ss()
        assert "experiment_queue" in ss
        assert ss["experiment_queue"] == []

    def test_make_ss_includes_batch_mode_key(self):
        """_make_ss()가 _batch_queue_mode 키를 포함한다."""
        ss = _make_ss()
        assert "_batch_queue_mode" in ss
        assert ss["_batch_queue_mode"] is False

    def test_make_ss_includes_batch_total_key(self):
        """_make_ss()가 _batch_total_count 키를 포함한다."""
        ss = _make_ss()
        assert "_batch_total_count" in ss
        assert ss["_batch_total_count"] == 0


# ── TestHandleBatchStart ───────────────────────────────────────────────────────

class TestHandleBatchStart:
    """_handle_batch_start() — 일괄 학습 시작 처리 검증 (FR-T3-15)."""

    _PRE = {"method": "none", "image_size": 256, "params": None}
    _MDL = {"model_type": "patchcore", "batch_size": 16, "params": {"backbone": "wide_resnet50_2"}}

    def _queue_items(self, count: int = 2, start_status: str = "대기중") -> list[dict]:
        return [
            {
                "name": f"exp_{i}",
                "preprocessing_config": dict(self._PRE),
                "model_config": dict(self._MDL),
                "status": start_status,
            }
            for i in range(count)
        ]

    def _run(self, ss: dict):
        with _SessionCtx(ss), \
             patch("tabs.tab3_training._handle_start_training") as mock_start, \
             patch.object(t3.st, "warning") as mock_warn, \
             patch.object(t3.st, "error") as mock_err:
            t3._handle_batch_start()
        return ss, mock_start, mock_warn, mock_err

    # ── 정상 케이스 ────────────────────────────────────────────────────────────

    def test_sets_batch_mode_true(self):
        """일괄 학습 시작 시 _batch_queue_mode = True."""
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=self._queue_items(3),
                      dataset_path="/data/mvtec")
        ss, *_ = self._run(ss)
        assert ss["_batch_queue_mode"] is True

    def test_sets_batch_total_count(self):
        """pending 항목 수가 _batch_total_count에 저장됨."""
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=self._queue_items(3),
                      dataset_path="/data/mvtec")
        ss, *_ = self._run(ss)
        assert ss["_batch_total_count"] == 3

    def test_first_pending_item_marked_running(self):
        """첫 번째 대기중 항목이 '진행중'으로 변경됨."""
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=self._queue_items(3),
                      dataset_path="/data/mvtec")
        ss, *_ = self._run(ss)
        assert ss["experiment_queue"][0]["status"] == "진행중"

    def test_other_items_remain_pending(self):
        """첫 항목 이후의 항목들은 여전히 '대기중'이어야 한다."""
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=self._queue_items(3),
                      dataset_path="/data/mvtec")
        ss, *_ = self._run(ss)
        assert ss["experiment_queue"][1]["status"] == "대기중"
        assert ss["experiment_queue"][2]["status"] == "대기중"

    def test_first_completed_item_is_skipped(self):
        """이미 완료된 항목을 건너뛰고 첫 번째 대기중 항목이 진행중이 됨."""
        items = self._queue_items(3)
        items[0]["status"] = "완료"   # 첫 번째 항목은 이미 완료
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=items,
                      dataset_path="/data/mvtec")
        ss, *_ = self._run(ss)
        assert ss["experiment_queue"][0]["status"] == "완료"   # 변경 없음
        assert ss["experiment_queue"][1]["status"] == "진행중"  # 두 번째가 선택됨

    def test_loads_first_item_preprocessing_config(self):
        """첫 항목의 preprocessing_config가 session_state에 로드됨."""
        items = self._queue_items(2)
        expected_pre = dict(items[0]["preprocessing_config"])
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=items,
                      dataset_path="/data/mvtec")
        ss, *_ = self._run(ss)
        assert ss["preprocessing_config"] == expected_pre

    def test_loads_first_item_model_config(self):
        """첫 항목의 model_config가 session_state에 로드됨."""
        items = self._queue_items(2)
        expected_mdl = dict(items[0]["model_config"])
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=items,
                      dataset_path="/data/mvtec")
        ss, *_ = self._run(ss)
        assert ss["model_config"] == expected_mdl

    def test_calls_handle_start_training_with_item_name(self):
        """_handle_start_training()이 첫 항목의 name으로 호출됨."""
        items = self._queue_items(2)
        items[0]["name"] = "my_batch_exp"
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=items,
                      dataset_path="/data/mvtec")
        ss, mock_start, *_ = self._run(ss)
        mock_start.assert_called_once_with("my_batch_exp")

    # ── 예외 케이스 ────────────────────────────────────────────────────────────

    def test_no_action_when_no_pending_items(self):
        """대기중 항목 없으면 배치 모드 미설정 + 경고."""
        items = self._queue_items(2, start_status="완료")
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=items,
                      dataset_path="/data/mvtec")
        ss, mock_start, mock_warn, _ = self._run(ss)
        assert ss["_batch_queue_mode"] is False
        mock_warn.assert_called_once()
        mock_start.assert_not_called()

    def test_no_action_when_training_running(self):
        """학습 중(current_run_status='running') → 배치 모드 미설정 + 경고."""
        ss = _make_ss(current_run_status="running",
                      experiment_queue=self._queue_items(2),
                      dataset_path="/data/mvtec")
        ss, mock_start, mock_warn, _ = self._run(ss)
        assert ss["_batch_queue_mode"] is False
        mock_warn.assert_called_once()
        mock_start.assert_not_called()

    def test_no_action_when_dataset_path_is_none(self):
        """dataset_path 미설정 → 배치 모드 미설정 + 오류."""
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=self._queue_items(2),
                      dataset_path=None)
        ss, mock_start, _, mock_err = self._run(ss)
        assert ss["_batch_queue_mode"] is False
        mock_err.assert_called_once()
        mock_start.assert_not_called()

    def test_only_one_item_marked_running_at_a_time(self):
        """한 번에 하나의 항목만 '진행중'으로 변경됨."""
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=self._queue_items(4),
                      dataset_path="/data/mvtec")
        ss, *_ = self._run(ss)
        running_count = sum(
            1 for item in ss["experiment_queue"]
            if item.get("status") == "진행중"
        )
        assert running_count == 1


# ── TestMarkCurrentBatchItem ───────────────────────────────────────────────────

class TestMarkCurrentBatchItem:
    """_mark_current_batch_item() — '진행중' 항목 상태 변경 검증."""

    def _make_queue(self, statuses: list[str]) -> list[dict]:
        return [{"name": f"exp_{i}", "status": s,
                 "preprocessing_config": {}, "model_config": {}}
                for i, s in enumerate(statuses)]

    def test_marks_running_item_as_given_status(self):
        ss = _make_ss(experiment_queue=self._make_queue(["대기중", "진행중", "대기중"]))
        with _SessionCtx(ss):
            t3._mark_current_batch_item("완료")
        assert ss["experiment_queue"][1]["status"] == "완료"

    def test_other_items_unchanged(self):
        ss = _make_ss(experiment_queue=self._make_queue(["대기중", "진행중", "대기중"]))
        with _SessionCtx(ss):
            t3._mark_current_batch_item("완료")
        assert ss["experiment_queue"][0]["status"] == "대기중"
        assert ss["experiment_queue"][2]["status"] == "대기중"

    def test_no_running_item_no_change(self):
        original = self._make_queue(["대기중", "완료"])
        ss = _make_ss(experiment_queue=list(original))
        with _SessionCtx(ss):
            t3._mark_current_batch_item("실패")
        assert ss["experiment_queue"][0]["status"] == "대기중"
        assert ss["experiment_queue"][1]["status"] == "완료"

    def test_mark_as_failed(self):
        ss = _make_ss(experiment_queue=self._make_queue(["진행중"]))
        with _SessionCtx(ss):
            t3._mark_current_batch_item("실패")
        assert ss["experiment_queue"][0]["status"] == "실패"

    def test_mark_as_skipped(self):
        ss = _make_ss(experiment_queue=self._make_queue(["대기중", "진행중"]))
        with _SessionCtx(ss):
            t3._mark_current_batch_item("건너뜀")
        assert ss["experiment_queue"][1]["status"] == "건너뜀"


# ── TestAdvanceBatchQueue ──────────────────────────────────────────────────────

class TestAdvanceBatchQueue:
    """_advance_batch_queue() — 다음 항목 자동 진행 검증."""

    _PRE = {"method": "none", "image_size": 256}
    _MDL = {"model_type": "patchcore", "batch_size": 4,
             "threshold_method": "percentile", "threshold_value": 95.0,
             "params": {"backbone": "wide_resnet50_2"}}

    def _item(self, status: str, name: str = "exp") -> dict:
        return {"name": name, "status": status,
                "preprocessing_config": dict(self._PRE),
                "model_config": dict(self._MDL)}

    def _run(self, ss: dict) -> tuple[dict, "MagicMock"]:
        with _SessionCtx(ss), \
             patch("tabs.tab3_training._handle_start_training") as mock_start, \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "error"):
            t3._advance_batch_queue()
        return ss, mock_start

    def test_starts_first_pending_item(self):
        items = [self._item("완료"), self._item("대기중", "exp_next")]
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=items, dataset_path="/data")
        ss, mock_start = self._run(ss)
        mock_start.assert_called_once_with("exp_next")

    def test_marks_next_item_as_running(self):
        items = [self._item("완료"), self._item("대기중")]
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=items, dataset_path="/data")
        self._run(ss)
        assert ss["experiment_queue"][1]["status"] == "진행중"

    def test_loads_next_item_config(self):
        pre = {"method": "clahe", "image_size": 128}
        mdl = {"model_type": "efficientad", "params": {}}
        items = [self._item("완료"),
                 {"name": "next", "status": "대기중",
                  "preprocessing_config": pre, "model_config": mdl}]
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=items, dataset_path="/data")
        self._run(ss)
        assert ss["preprocessing_config"] == pre
        assert ss["model_config"] == mdl

    def test_ends_batch_when_no_pending(self):
        items = [self._item("완료"), self._item("실패")]
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=items, dataset_path="/data",
                      _batch_queue_mode=True)
        ss, mock_start = self._run(ss)
        assert ss["_batch_queue_mode"] is False
        mock_start.assert_not_called()

    def test_sets_last_result_on_batch_end(self):
        items = [self._item("완료"), self._item("건너뜀")]
        ss = _make_ss(current_run_status="idle",
                      experiment_queue=items, dataset_path="/data",
                      _batch_queue_mode=True)
        self._run(ss)
        assert "level" in ss["_last_result"]


# ── TestHandleCompletedBatch ───────────────────────────────────────────────────

class TestHandleCompletedBatch:
    """_handle_completed() 배치 모드 — 완료 마킹 + advance 예약 검증."""

    _METRICS = {
        "auc": 0.95, "accuracy": 0.9, "precision": 0.9,
        "recall": 0.9, "f1_score": 0.9, "f2_score": 0.9,
        "confusion_matrix": {"tp": 9, "fp": 1, "tn": 9, "fn": 1},
        "anomaly_scores": [0.1, 0.9], "image_labels": [0, 1],
    }

    def _run(self, ss: dict) -> dict:
        msg = {
            "type": "completed", "y_true": [0, 1],
            "anomaly_scores": [0.1, 0.9], "anomaly_maps": {},
            "image_paths": [], "model": MagicMock(), "duration_seconds": 30,
        }
        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.compute_threshold", return_value=0.5), \
             patch("tabs.tab3_training.compute_metrics", return_value=self._METRICS), \
             patch("tabs.tab3_training.check_disk_before_save"), \
             patch("tabs.tab3_training.save_completed_experiment"), \
             patch("tabs.tab3_training.set_anomaly_map_cache"), \
             patch.object(t3.st, "success"), \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "error"):
            t3._handle_completed(msg)
        return ss

    def test_marks_current_item_as_completed(self):
        items = [{"name": "e", "status": "진행중",
                  "preprocessing_config": {}, "model_config": {}}]
        ss = _make_ss(_batch_queue_mode=True, experiment_queue=items)
        self._run(ss)
        assert ss["experiment_queue"][0]["status"] == "완료"

    def test_sets_advance_pending(self):
        items = [{"name": "e", "status": "진행중",
                  "preprocessing_config": {}, "model_config": {}}]
        ss = _make_ss(_batch_queue_mode=True, experiment_queue=items)
        self._run(ss)
        assert ss["_batch_advance_pending"] is True

    def test_no_batch_effect_in_single_mode(self):
        ss = _make_ss(_batch_queue_mode=False, experiment_queue=[])
        self._run(ss)
        assert ss["_batch_advance_pending"] is False


# ── TestHandleErrorBatch ───────────────────────────────────────────────────────

class TestHandleErrorBatch:
    """_handle_error() 배치 모드 — 실패 기록 + advance 예약 검증."""

    def _run_error(self, ss: dict) -> dict:
        msg = {"type": "error", "exception": RuntimeError("GPU OOM"),
               "traceback": "OOM error"}
        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.append_experiment"), \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "error"):
            t3._handle_error(msg)
        return ss

    def test_marks_item_as_failed(self):
        items = [{"name": "e", "status": "진행중",
                  "preprocessing_config": {}, "model_config": {}}]
        ss = _make_ss(_batch_queue_mode=True, experiment_queue=items)
        self._run_error(ss)
        assert ss["experiment_queue"][0]["status"] == "실패"

    def test_sets_advance_pending_on_error(self):
        items = [{"name": "e", "status": "진행중",
                  "preprocessing_config": {}, "model_config": {}}]
        ss = _make_ss(_batch_queue_mode=True, experiment_queue=items)
        self._run_error(ss)
        assert ss["_batch_advance_pending"] is True

    def test_last_result_level_warning_in_batch_mode(self):
        ss = _make_ss(_batch_queue_mode=True, experiment_queue=[])
        self._run_error(ss)
        assert ss["_last_result"]["level"] == "warning"

    def test_last_result_level_error_in_single_mode(self):
        ss = _make_ss(_batch_queue_mode=False)
        self._run_error(ss)
        assert ss["_last_result"]["level"] == "error"


# ── TestHandleStoppedBatch ─────────────────────────────────────────────────────

class TestHandleStoppedBatch:
    """_handle_stopped() 배치 모드 — 중단/건너뜀 처리 검증."""

    def _run_stopped(self, ss: dict, step: int = 0) -> dict:
        msg = {"type": "stopped", "step": step}
        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.append_experiment"), \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "info"):
            t3._handle_stopped(msg)
        return ss

    def test_batch_full_stop_marks_item_as_stopped(self):
        """⏹ 전체 중단 → '진행중' 항목 '중단' 처리."""
        items = [{"name": "e", "status": "진행중",
                  "preprocessing_config": {}, "model_config": {}}]
        ss = _make_ss(_batch_queue_mode=True, experiment_queue=items,
                      _batch_advance_pending=False)
        self._run_stopped(ss)
        assert ss["experiment_queue"][0]["status"] == "중단"

    def test_batch_full_stop_ends_batch_mode(self):
        """⏹ 전체 중단 → _batch_queue_mode = False."""
        ss = _make_ss(_batch_queue_mode=True,
                      experiment_queue=[{"name": "e", "status": "진행중",
                                         "preprocessing_config": {}, "model_config": {}}],
                      _batch_advance_pending=False)
        self._run_stopped(ss)
        assert ss["_batch_queue_mode"] is False

    def test_ghost_stop_after_skip_is_ignored(self):
        """advance_pending=True이면 ghost stop 무시 → queue 상태 불변."""
        items = [{"name": "e", "status": "건너뜀",
                  "preprocessing_config": {}, "model_config": {}}]
        ss = _make_ss(_batch_queue_mode=True, experiment_queue=items,
                      _batch_advance_pending=True)
        self._run_stopped(ss)
        assert ss["experiment_queue"][0]["status"] == "건너뜀"  # 변경 없음

    def test_skip_via_stop_path_marks_item_skipped(self):
        """_batch_skip_current=True + stopped → '건너뜀' 처리 + advance 예약."""
        items = [{"name": "e", "status": "진행중",
                  "preprocessing_config": {}, "model_config": {}}]
        ss = _make_ss(_batch_queue_mode=True, _batch_skip_current=True,
                      experiment_queue=items, _batch_advance_pending=False)
        self._run_stopped(ss)
        assert ss["experiment_queue"][0]["status"] == "건너뜀"
        assert ss["_batch_advance_pending"] is True
        assert ss["_batch_skip_current"] is False


# ── TestBatchQueueIntegration ──────────────────────────────────────────────────

class TestBatchQueueIntegration:
    """대기열 배치 학습 시나리오 통합 검증 (FR-T3-15).

    실제 TrainingWorker 없이 메시지 시퀀스를 직접 주입하여
    완료→자동 다음, 건너뛰기(B안), 실패→계속 흐름을 종합 검증한다.
    """

    _PRE = {"method": "none", "image_size": 256,
            "normalization": "imagenet",
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "params": None, "resize_mode": "padding"}
    _MDL = {"model_type": "patchcore", "batch_size": 4, "random_seed": 42,
            "image_size": 256, "threshold_method": "percentile",
            "threshold_value": 95.0,
            "params": {"backbone": "wide_resnet50_2",
                       "coreset_sampling_ratio": 0.1,
                       "neighbourhood_kernel_size": 3,
                       "pretrained_source": "torchvision",
                       "pretrained_path": None,
                       "max_train": 100, "knn": 9, "top_k_ratio": 0.1}}
    _METRICS = {
        "auc": 0.95, "accuracy": 0.9, "precision": 0.9, "recall": 0.9,
        "f1_score": 0.9, "f2_score": 0.9,
        "confusion_matrix": {"tp": 9, "fp": 1, "tn": 9, "fn": 1},
        "anomaly_scores": [0.1, 0.9], "image_labels": [0, 1],
    }

    def _item(self, status: str, name: str) -> dict:
        return {"name": name, "status": status,
                "preprocessing_config": dict(self._PRE),
                "model_config": dict(self._MDL)}

    def _base_ss(self, items: list, **kw) -> dict:
        """배치 학습 기본 session_state 생성."""
        return _make_ss(
            current_run_status="running",
            current_exp_id=EXP_ID,
            _batch_queue_mode=True,
            _batch_total_count=len(items),
            experiment_queue=items,
            dataset_path="/data/mvtec",
            **kw,
        )

    def _simulate_complete(self, ss: dict) -> None:
        """completed 메시지 시뮬레이션 (저장 관련 모두 mock)."""
        # Ensure running state before each completion
        ss["current_run_status"] = "running"
        if not ss.get("current_exp_id"):
            ss["current_exp_id"] = EXP_ID
        msg = {
            "type": "completed", "y_true": [0, 1],
            "anomaly_scores": [0.1, 0.9], "anomaly_maps": {},
            "image_paths": [], "model": MagicMock(), "duration_seconds": 10,
        }
        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.compute_threshold", return_value=0.5), \
             patch("tabs.tab3_training.compute_metrics", return_value=self._METRICS), \
             patch("tabs.tab3_training.check_disk_before_save"), \
             patch("tabs.tab3_training.save_completed_experiment"), \
             patch("tabs.tab3_training.set_anomaly_map_cache"), \
             patch.object(t3.st, "success"), \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "error"):
            t3._handle_completed(msg)

    def _simulate_error(self, ss: dict) -> None:
        """error 메시지 시뮬레이션 (GPU OOM 등 학습 오류)."""
        ss["current_run_status"] = "running"
        if not ss.get("current_exp_id"):
            ss["current_exp_id"] = EXP_ID
        msg = {"type": "error", "exception": Exception("GPU OOM"),
               "traceback": "CUDA out of memory"}
        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.append_experiment"), \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "error"):
            t3._handle_error(msg)

    def _simulate_paused_skip(self, ss: dict) -> None:
        """⏭ 건너뛰기: pause(checkpoint저장) → _handle_paused 처리 (B안)."""
        ss["current_run_status"] = "running"
        if not ss.get("current_exp_id"):
            ss["current_exp_id"] = EXP_ID
        ss["_batch_skip_current"] = True
        msg = {"type": "paused", "ckpt_path": "/tmp/mock.ckpt"}
        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.append_experiment"), \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "error"):
            t3._handle_paused(msg)

    def _simulate_advance(self, ss: dict) -> "MagicMock":
        """_batch_advance_pending 소비 → 다음 항목 시작 시뮬레이션.

        _render_queue_at_tab3_top()이 다음 rerun에서 하는 동작을 모방.
        """
        ss["_batch_advance_pending"] = False
        with _SessionCtx(ss), \
             patch("tabs.tab3_training._handle_start_training") as mock_start, \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "error"):
            t3._advance_batch_queue()
        return mock_start

    # ── 시나리오 1: 3개 항목 순차 완료 ────────────────────────────────────────

    def test_three_items_complete_sequentially(self):
        """3개 항목이 순서대로 완료되어 배치가 종료된다."""
        items = [
            self._item("진행중", "exp_1"),
            self._item("대기중", "exp_2"),
            self._item("대기중", "exp_3"),
        ]
        ss = self._base_ss(items)

        # ── 1번 완료 ──
        self._simulate_complete(ss)
        assert ss["experiment_queue"][0]["status"] == "완료", "1번 완료 마킹 실패"
        assert ss["_batch_advance_pending"] is True

        # ── 2번 시작 ──
        mock_start = self._simulate_advance(ss)
        assert ss["experiment_queue"][1]["status"] == "진행중", "2번 진행중 마킹 실패"
        mock_start.assert_called_once_with("exp_2")

        # ── 2번 완료 ──
        self._simulate_complete(ss)
        assert ss["experiment_queue"][1]["status"] == "완료"
        assert ss["_batch_advance_pending"] is True

        # ── 3번 시작 ──
        mock_start = self._simulate_advance(ss)
        assert ss["experiment_queue"][2]["status"] == "진행중"
        mock_start.assert_called_once_with("exp_3")

        # ── 3번 완료 ──
        self._simulate_complete(ss)
        assert ss["experiment_queue"][2]["status"] == "완료"

        # ── 대기중 없음 → 배치 종료 ──
        mock_start = self._simulate_advance(ss)
        mock_start.assert_not_called()
        assert ss["_batch_queue_mode"] is False

    def test_all_items_completed_after_sequential_batch(self):
        """순차 완료 후 모든 항목이 '완료' 상태이어야 한다."""
        items = [
            self._item("진행중", "exp_1"),
            self._item("대기중", "exp_2"),
        ]
        ss = self._base_ss(items)

        self._simulate_complete(ss)
        self._simulate_advance(ss)
        self._simulate_complete(ss)
        self._simulate_advance(ss)  # batch ends

        assert all(item["status"] == "완료"
                   for item in ss["experiment_queue"])
        assert ss["_batch_queue_mode"] is False

    def test_batch_end_result_success_when_all_complete(self):
        """모든 항목 완료 시 배치 종료 _last_result.level == 'success'."""
        items = [self._item("진행중", "exp_1")]
        ss = self._base_ss(items)

        self._simulate_complete(ss)
        self._simulate_advance(ss)  # batch ends

        assert ss["_last_result"]["level"] == "success"

    # ── 시나리오 2: 건너뛰기(B안 — 체크포인트 후 건너뜀) ─────────────────────

    def test_skip_marks_item_skipped_and_advances_to_next(self):
        """⏭ 건너뛰기 시 항목이 '건너뜀'이 되고 다음 항목이 시작된다."""
        items = [
            self._item("진행중", "exp_1"),
            self._item("대기중", "exp_2"),
            self._item("대기중", "exp_3"),
        ]
        ss = self._base_ss(items)

        self._simulate_paused_skip(ss)

        assert ss["experiment_queue"][0]["status"] == "건너뜀"
        assert ss["_batch_advance_pending"] is True
        assert ss["_batch_skip_current"] is False

        mock_start = self._simulate_advance(ss)
        assert ss["experiment_queue"][1]["status"] == "진행중"
        mock_start.assert_called_once_with("exp_2")

    def test_ghost_stop_after_skip_does_not_overwrite_status(self):
        """건너뛰기 후 ghost stopped 메시지가 '건너뜀' 상태를 덮어쓰지 않는다."""
        items = [self._item("건너뜀", "exp_1"),
                 self._item("대기중", "exp_2")]
        ss = self._base_ss(items, _batch_advance_pending=True)
        ss["current_exp_id"] = EXP_ID

        msg = {"type": "stopped", "step": 100}
        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.append_experiment"), \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "info"):
            t3._handle_stopped(msg)

        # Ghost stop은 무시 → 건너뜀 상태 유지
        assert ss["experiment_queue"][0]["status"] == "건너뜀"
        # advance_pending은 _render_queue_at_tab3_top이 소비하므로 여기선 유지
        assert ss["_batch_advance_pending"] is True

    def test_skip_saves_to_history(self):
        """건너뛰기 시 history.json에 '건너뜀' 상태로 기록된다."""
        items = [self._item("진행중", "exp_skip")]
        ss = self._base_ss(items)

        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.append_experiment") as mock_append, \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "error"):
            ss["_batch_skip_current"] = True
            t3._handle_paused({"type": "paused", "ckpt_path": "/tmp/c.ckpt"})

        mock_append.assert_called_once()
        saved_record = mock_append.call_args[0][0]
        assert saved_record["status"] == "건너뜀"

    # ── 시나리오 3: 실패 시뮬레이션 ──────────────────────────────────────────

    def test_failure_marks_item_failed_and_continues(self):
        """GPU OOM 등 오류 발생 시 항목이 '실패'로 마킹되고 다음 항목이 시작된다."""
        items = [
            self._item("진행중", "exp_1"),
            self._item("대기중", "exp_2"),
            self._item("대기중", "exp_3"),
        ]
        ss = self._base_ss(items)

        self._simulate_error(ss)

        assert ss["experiment_queue"][0]["status"] == "실패"
        assert ss["_batch_advance_pending"] is True

        mock_start = self._simulate_advance(ss)
        assert ss["experiment_queue"][1]["status"] == "진행중"
        mock_start.assert_called_once_with("exp_2")

    def test_failure_saves_to_history(self):
        """학습 오류 시 history.json에 '실패' 상태로 기록된다."""
        items = [self._item("진행중", "exp_fail")]
        ss = self._base_ss(items)

        with _SessionCtx(ss), \
             patch("tabs.tab3_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab3_training.append_experiment") as mock_append, \
             patch.object(t3.st, "warning"), \
             patch.object(t3.st, "error"):
            t3._handle_error({
                "type": "error", "exception": Exception("OOM"),
                "traceback": "CUDA OOM",
            })

        mock_append.assert_called_once()
        saved_record = mock_append.call_args[0][0]
        assert saved_record["status"] == "실패"

    def test_batch_continues_after_failure(self):
        """항목 실패 후에도 배치 모드가 유지되고 다음 항목으로 진행된다."""
        items = [
            self._item("진행중", "exp_1"),
            self._item("대기중", "exp_2"),
        ]
        ss = self._base_ss(items)

        self._simulate_error(ss)

        # 배치 모드 유지 확인
        assert ss["_batch_queue_mode"] is True

        self._simulate_advance(ss)
        assert ss["experiment_queue"][1]["status"] == "진행중"

    # ── 시나리오 4: 완료 + 실패 + 건너뜀 혼합 ─────────────────────────────────

    def test_mixed_complete_fail_skip_scenario(self):
        """완료 1 + 실패 1 + 건너뜀 1 혼합 시나리오 — 최종 배치 종료 확인."""
        items = [
            self._item("진행중", "exp_1"),
            self._item("대기중", "exp_2"),
            self._item("대기중", "exp_3"),
        ]
        ss = self._base_ss(items)

        # exp_1 완료
        self._simulate_complete(ss)
        self._simulate_advance(ss)

        # exp_2 실패
        self._simulate_error(ss)
        self._simulate_advance(ss)

        # exp_3 건너뜀
        self._simulate_paused_skip(ss)
        mock_start = self._simulate_advance(ss)  # No more pending → batch ends

        mock_start.assert_not_called()
        assert ss["_batch_queue_mode"] is False

        statuses = [item["status"] for item in ss["experiment_queue"]]
        assert "대기중" not in statuses
        assert "완료" in statuses
        assert "실패" in statuses
        assert "건너뜀" in statuses

    def test_batch_end_result_warning_when_failure_exists(self):
        """실패 항목이 있으면 배치 종료 _last_result.level == 'warning'."""
        items = [
            self._item("진행중", "exp_1"),
            self._item("대기중", "exp_2"),
        ]
        ss = self._base_ss(items)

        # exp_1 fails, exp_2 completes
        self._simulate_error(ss)
        self._simulate_advance(ss)
        self._simulate_complete(ss)
        self._simulate_advance(ss)  # batch ends

        assert ss["_last_result"]["level"] == "warning"

    def test_advance_pending_cleared_before_next_starts(self):
        """advance_pending은 다음 항목 시작 전 False로 초기화된다."""
        items = [
            self._item("진행중", "exp_1"),
            self._item("대기중", "exp_2"),
        ]
        ss = self._base_ss(items)

        self._simulate_complete(ss)
        assert ss["_batch_advance_pending"] is True

        # _simulate_advance clears advance_pending (mirrors render logic)
        self._simulate_advance(ss)
        assert ss["_batch_advance_pending"] is False
