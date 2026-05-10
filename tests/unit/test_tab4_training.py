"""
탭4 단위 테스트

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

import tabs.tab4_training as t4
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
    """t4.st.session_state를 fake dict로 교체하는 컨텍스트 매니저."""

    def __init__(self, ss: dict) -> None:
        self._ss = ss
        self._orig = None

    def __enter__(self) -> dict:
        self._orig = getattr(t4.st, "session_state", None)
        t4.st.session_state = self._ss
        return self._ss

    def __exit__(self, *_) -> None:
        if self._orig is not None:
            t4.st.session_state = self._orig


# ── TestResetRunState ──────────────────────────────────────────────────────────

class TestResetRunState:
    """_reset_run_state() — session_state 5개 키 초기화 (PRD 06 §5.2)."""

    def _run(self, **overrides) -> dict:
        ss = _make_ss(**overrides)
        with _SessionCtx(ss):
            t4._reset_run_state()
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
            t4._reset_run_state()
        assert ss["_worker"] is None


# ── TestBuildExperimentRecordStopped ──────────────────────────────────────────

class TestBuildExperimentRecordStopped:
    """_build_experiment_record(status="중단") 불변 조건 (PRD 07 §6.3, 00_Global §2 R-05)."""

    def _build(self, exp_id: str = EXP_ID, ss: dict | None = None) -> dict:
        ss = ss or _make_ss()
        with _SessionCtx(ss), patch("tabs.tab4_training.load_config", return_value=_EXP_CFG_MOCK):
            return t4._build_experiment_record(exp_id, "중단", None, None)

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
             patch("tabs.tab4_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab4_training.append_experiment") as mock_append, \
             patch.object(t4.st, "warning") as mock_warn:
            t4._handle_stopped(msg)
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
        ss, _, _ = self._run({"type": "stopped", "step": 500})
        assert "500" in ss["_last_result"]["text"]

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
             patch("tabs.tab4_training.load_config", return_value={}), \
             patch("tabs.tab4_training.append_experiment"), \
             patch.object(t4.st, "warning"):
            t4._handle_stopped({"type": "stopped", "step": 0})
        assert ss["current_run_status"] == "idle"

    def test_state_reset_even_when_append_raises(self):
        """append_experiment RuntimeError 발생해도 _reset_run_state()는 실행됨."""
        ss = _make_ss()
        with _SessionCtx(ss), \
             patch("tabs.tab4_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab4_training.append_experiment", side_effect=RuntimeError("I/O")), \
             patch.object(t4.st, "warning"):
            t4._handle_stopped({"type": "stopped", "step": 0})
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
             patch("tabs.tab4_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab4_training.append_experiment"), \
             patch.object(t4.st, "warning"), \
             patch.object(t4.st, "error"), \
             patch.object(t4.st, "success"):
            t4._drain_queue()
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
             patch("tabs.tab4_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab4_training.append_experiment"), \
             patch.object(t4.st, "warning"), \
             patch.object(t4.st, "error") as mock_err:
            t4._drain_queue()

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
            t4._drain_queue()
        assert ss["current_run_status"] == "running"

    def test_none_queue_no_crash(self):
        """_result_queue=None이면 드레인 생략 (R-RACE-02 이후 재진입 방어)."""
        ss = _make_ss()
        ss["_result_queue"] = None
        with _SessionCtx(ss):
            t4._drain_queue()
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
             patch.object(t4.st, "info"), \
             patch.object(t4.st, "warning"):
            t4._guard()
        assert ss["current_run_status"] == "idle"

    def test_alive_worker_with_valid_queue_does_not_reset(self):
        """worker 살아있고 _result_queue 유효 → reset 없음."""
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        ss = _make_ss()
        ss["_worker"] = mock_worker
        # _result_queue는 _make_ss()에서 queue.Queue()로 설정됨
        with _SessionCtx(ss), \
             patch.object(t4.st, "info"), \
             patch.object(t4.st, "warning"):
            t4._guard()
        assert ss["current_run_status"] == "running"

    def test_dead_worker_does_not_trigger_reset(self):
        """종료된 worker는 orphan 아님 → reset 없음."""
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = False
        ss = _make_ss()
        ss["_worker"] = mock_worker
        ss["_result_queue"] = None
        with _SessionCtx(ss), \
             patch.object(t4.st, "info"), \
             patch.object(t4.st, "warning"):
            t4._guard()
        # 죽은 worker는 orphan 아님 → _reset_run_state 미호출
        assert ss["current_run_status"] == "running"


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
             patch.object(t4.st, "button", side_effect=fake_button), \
             patch.object(t4.st, "info"), \
             patch.object(t4.st, "progress"), \
             patch.object(t4.st, "text_area"), \
             patch.object(t4.st, "text_input", return_value=""):
            fn()
        return labels

    def test_running_ui_has_no_start_button(self):
        """running UI에 [학습 시작] 버튼 부재."""
        ss = _make_ss()
        labels = self._collect_buttons(t4._render_running_ui, ss)
        assert "학습 시작" not in labels

    def test_running_ui_has_stop_button(self):
        """running UI에 [학습 중지] 버튼 존재."""
        ss = _make_ss()
        labels = self._collect_buttons(t4._render_running_ui, ss)
        assert "학습 중지" in labels

    def test_idle_ui_has_start_button(self):
        """idle UI에 [학습 시작] 버튼 존재."""
        ss = _make_ss(current_run_status="idle")
        with patch("tabs.tab4_training._render_pretrain_summary"):
            labels = self._collect_buttons(t4._render_idle_ui, ss)
        assert "학습 시작" in labels

    def test_idle_ui_has_no_stop_button(self):
        """idle UI에 [학습 중지] 버튼 부재."""
        ss = _make_ss(current_run_status="idle")
        with patch("tabs.tab4_training._render_pretrain_summary"):
            labels = self._collect_buttons(t4._render_idle_ui, ss)
        assert "학습 중지" not in labels


# ── TestGenerateExperimentId ──────────────────────────────────────────────────

class TestGenerateExperimentId:
    """R-NAMING-03 experiment_id 형식 (PRD 07 §2.1)."""

    def test_efficientad_prefix(self):
        assert t4.generate_experiment_id("efficientad").startswith("efficientad_")

    def test_patchcore_prefix(self):
        assert t4.generate_experiment_id("patchcore").startswith("patchcore_")

    def test_four_part_format(self):
        eid = t4.generate_experiment_id("efficientad")
        parts = eid.split("_")
        # "efficientad" + YYYYMMDD + HHMMSS + rand4
        assert len(parts) == 4

    def test_date_part_length_8(self):
        eid = t4.generate_experiment_id("efficientad")
        assert len(eid.split("_")[1]) == 8

    def test_time_part_length_6(self):
        eid = t4.generate_experiment_id("efficientad")
        assert len(eid.split("_")[2]) == 6

    def test_rand_part_is_4_char_lowercase_hex(self):
        """R-ID-01: uuid4().hex[:4] → 소문자 16진수 4자리."""
        for _ in range(10):
            eid = t4.generate_experiment_id("efficientad")
            rand_part = eid.split("_")[3]
            assert re.fullmatch(r"[0-9a-f]{4}", rand_part), f"rand_part={rand_part!r}"

    def test_uniqueness_across_calls(self):
        """10회 호출 중 최소 절반 이상 고유 (uuid4 덕분에 거의 항상)."""
        ids = [t4.generate_experiment_id("efficientad") for _ in range(10)]
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
             patch("tabs.tab4_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab4_training.compute_threshold", return_value=0.5) as mock_thresh, \
             patch("tabs.tab4_training.compute_metrics", return_value=self._METRICS) as mock_metrics, \
             patch("tabs.tab4_training.check_disk_before_save"), \
             patch("tabs.tab4_training.save_completed_experiment",
                   side_effect=save_side) as mock_save, \
             patch("tabs.tab4_training.set_anomaly_map_cache") as mock_cache, \
             patch.object(t4.st, "success") as mock_success, \
             patch.object(t4.st, "warning") as mock_warn, \
             patch.object(t4.st, "error") as mock_err:
            t4._handle_completed(msg)
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
        """성공 메시지에 AUC 값 포함 (탭4 UI 알림 조건 — PRD 7.4절)."""
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
        with patch("tabs.tab4_training.torch") as mock_torch, \
             _SessionCtx(ss), \
             patch("tabs.tab4_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab4_training.compute_threshold", return_value=0.5), \
             patch("tabs.tab4_training.compute_metrics", return_value=self._METRICS), \
             patch("tabs.tab4_training.check_disk_before_save"), \
             patch("tabs.tab4_training.save_completed_experiment"), \
             patch("tabs.tab4_training.set_anomaly_map_cache"), \
             patch.object(t4.st, "success"), \
             patch.object(t4.st, "warning"), \
             patch.object(t4.st, "error"):
            t4._handle_completed(self._make_msg())
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_cuda_empty_cache_not_called_for_cpu_device(self):
        """device=="cpu"이면 torch.cuda.empty_cache() 미호출."""
        ss = _make_ss()
        ss["device_info"] = {"device": "cpu"}
        with patch("tabs.tab4_training.torch") as mock_torch, \
             _SessionCtx(ss), \
             patch("tabs.tab4_training.load_config", return_value=_EXP_CFG_MOCK), \
             patch("tabs.tab4_training.compute_threshold", return_value=0.5), \
             patch("tabs.tab4_training.compute_metrics", return_value=self._METRICS), \
             patch("tabs.tab4_training.check_disk_before_save"), \
             patch("tabs.tab4_training.save_completed_experiment"), \
             patch("tabs.tab4_training.set_anomaly_map_cache"), \
             patch.object(t4.st, "success"), \
             patch.object(t4.st, "warning"), \
             patch.object(t4.st, "error"):
            t4._handle_completed(self._make_msg())
        mock_torch.cuda.empty_cache.assert_not_called()


# ── TestHandleError ───────────────────────────────────────────────────────────

class TestHandleError:
    """_handle_error() — 오류 핸들러 검증."""

    def _run(self, msg: dict, ss: dict | None = None):
        ss = ss or _make_ss()
        with _SessionCtx(ss), \
             patch.object(t4.st, "error") as mock_err:
            t4._handle_error(msg)
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
