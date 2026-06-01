from __future__ import annotations

import queue
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.training_worker import TrainingWorker


def _make_worker(tmp_path, stop_event=None, q=None, **overrides):
    defaults = {
        "experiment_id": "test_exp_001",
        "model_config": {
            "model_type": "patchcore",
            "image_size": 64,
            "batch_size": 2,
            "random_seed": 42,
            "params": {
                "backbone": "wide_resnet50_2",
                "coreset_sampling_ratio": 0.1,
                "neighbourhood_kernel_size": 3,
                "pretrained_source": "torchvision",
                "max_train": 10,
            },
        },
        "preprocessing_config": {
            "method": "none",
            "image_size": 64,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "dataset_path": str(tmp_path),
        "device": "cpu",
        "stop_event": stop_event or threading.Event(),
        "result_queue": q or queue.Queue(),
    }
    defaults.update(overrides)
    return TrainingWorker(**defaults)


def _drain_to_terminal(q: queue.Queue) -> dict:
    """
    Queue에서 "log" 메시지를 건너뛰고 첫 번째 종료 메시지를 반환.
    TrainingWorker는 _run_impl() 시작 시 log 메시지를 먼저 emit하므로
    단순 get_nowait()으로는 terminal 메시지를 바로 얻을 수 없다.
    worker.join() 이후 호출 시 Queue에 모든 메시지가 쌓여 있다.
    """
    while True:
        msg = q.get_nowait()
        if msg["type"] in ("error", "stopped", "completed"):
            return msg


class TestWorkerStopBeforeStart:
    def test_pre_set_stop_event_emits_stopped(self, tmp_path):
        stop = threading.Event()
        stop.set()
        q = queue.Queue()
        worker = _make_worker(tmp_path, stop_event=stop, q=q)
        worker.start()
        worker.join(timeout=2)
        msg = q.get_nowait()
        assert msg["type"] == "stopped"

    def test_pre_set_stop_event_includes_step_zero(self, tmp_path):
        stop = threading.Event()
        stop.set()
        q = queue.Queue()
        worker = _make_worker(tmp_path, stop_event=stop, q=q)
        worker.start()
        worker.join(timeout=2)
        msg = q.get_nowait()
        assert msg["type"] == "stopped"
        assert "step" in msg
        assert msg["step"] == 0

    def test_pre_set_does_not_call_run_impl(self, tmp_path):
        stop = threading.Event()
        stop.set()
        q = queue.Queue()
        worker = _make_worker(tmp_path, stop_event=stop, q=q)
        with patch.object(worker, "_run_impl") as mock_impl:
            worker.start()
            worker.join(timeout=2)
        mock_impl.assert_not_called()


class TestWorkerErrorHandling:
    def test_invalid_dataset_path_puts_error_message(self, tmp_path):
        """빈 tmp_path는 MVTec AD 구조가 없으므로 error 메시지 발생."""
        q = queue.Queue()
        worker = _make_worker(tmp_path, q=q)
        worker.start()
        worker.join(timeout=5)
        msg = _drain_to_terminal(q)
        assert msg["type"] == "error"

    def test_error_message_contains_traceback_string(self, tmp_path):
        q = queue.Queue()
        worker = _make_worker(tmp_path, q=q)
        worker.start()
        worker.join(timeout=5)
        msg = _drain_to_terminal(q)
        assert msg["type"] == "error"
        assert isinstance(msg["traceback"], str)
        assert len(msg["traceback"]) > 0

    def test_error_message_contains_exception(self, tmp_path):
        q = queue.Queue()
        worker = _make_worker(tmp_path, q=q)
        worker.start()
        worker.join(timeout=5)
        msg = _drain_to_terminal(q)
        assert "exception" in msg
        assert isinstance(msg["exception"], Exception)


class TestWorkerBuildDatasetPatchable:
    def test_build_dataset_can_be_patched(self):
        import utils.training_worker as tw
        assert hasattr(tw, "build_dataset")

    def test_engine_importable_or_none(self):
        import utils.training_worker as tw
        assert tw.Engine is None or tw.Engine is not None

    def test_worker_is_daemon_thread(self, tmp_path):
        stop = threading.Event()
        stop.set()
        q = queue.Queue()
        worker = _make_worker(tmp_path, stop_event=stop, q=q)
        assert worker.daemon is True

    def test_worker_thread_name_contains_experiment_id(self, tmp_path):
        worker = _make_worker(tmp_path, experiment_id="myexp_123")
        assert "myexp_123" in worker.name


class TestWorkerStopDuringTraining:
    def test_stop_event_set_after_start_eventually_stops(self, tmp_path):
        """stop_event 설정 시 worker가 종료되어야 함."""
        q = queue.Queue()
        stop = threading.Event()
        worker = _make_worker(tmp_path, stop_event=stop, q=q)
        worker.start()
        # 즉시 stop 설정 (학습 시작 전 또는 직후)
        stop.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        # log 메시지를 건너뛰고 종료 메시지 확인
        msg = _drain_to_terminal(q)
        assert msg["type"] in ("stopped", "error")


# ── helpers ────────────────────────────────────────────────────────────────────

def _collect_all_messages(q: queue.Queue) -> list[dict]:
    """큐의 모든 메시지를 수집 (worker.join() 이후 호출)."""
    messages: list[dict] = []
    while True:
        try:
            messages.append(q.get_nowait())
        except queue.Empty:
            break
    return messages


# ── TestWorkerStageEmission ────────────────────────────────────────────────────

class TestWorkerStageEmission:
    """TrainingWorker가 stage 메시지를 올바르게 방출하는지 통합 검증 (FR-T3-11/12).

    빈 tmp_path는 MVTec AD 구조가 없으므로 DataLoader 실패 → error 메시지.
    그러나 DataLoader 시도 직전 stage(0, '데이터 로딩')는 반드시 방출된다.
    """

    def test_first_stage_is_data_loading(self, tmp_path):
        """worker 시작 시 첫 번째 stage 메시지가 stage_idx=0 '데이터 로딩'이어야 한다."""
        q = queue.Queue()
        worker = _make_worker(tmp_path, q=q)
        worker.start()
        worker.join(timeout=5)

        msgs = _collect_all_messages(q)
        stage_msgs = [m for m in msgs if m["type"] == "stage"]

        assert len(stage_msgs) >= 1
        assert stage_msgs[0]["stage_idx"] == 0
        assert stage_msgs[0]["stage_name"] == "데이터 로딩"

    def test_stage_messages_precede_error(self, tmp_path):
        """빈 데이터셋 → stage 메시지가 error 메시지 이전에 위치한다."""
        q = queue.Queue()
        worker = _make_worker(tmp_path, q=q)
        worker.start()
        worker.join(timeout=5)

        msgs = _collect_all_messages(q)
        types = [m["type"] for m in msgs]

        assert "error" in types
        last_stage_pos = max(
            (i for i, m in enumerate(msgs) if m["type"] == "stage"), default=-1
        )
        error_pos = types.index("error")
        assert last_stage_pos < error_pos

    def test_stage_message_has_required_fields(self, tmp_path):
        """stage 메시지는 type, stage_idx(int ≥ 0), stage_name(str) 필드를 포함해야 한다."""
        q = queue.Queue()
        worker = _make_worker(tmp_path, q=q)
        worker.start()
        worker.join(timeout=5)

        msgs = _collect_all_messages(q)
        for msg in (m for m in msgs if m["type"] == "stage"):
            assert isinstance(msg.get("stage_idx"), int)
            assert isinstance(msg.get("stage_name"), str)
            assert len(msg["stage_name"]) > 0
            assert msg["stage_idx"] >= 0

    def test_no_stage_emitted_when_stop_preset(self, tmp_path):
        """stop_event 사전 설정 → run() 진입 전 종료 → stage 메시지 미방출."""
        stop = threading.Event()
        stop.set()
        q = queue.Queue()
        worker = _make_worker(tmp_path, stop_event=stop, q=q)
        worker.start()
        worker.join(timeout=2)

        msgs = _collect_all_messages(q)
        assert not any(m["type"] == "stage" for m in msgs)

    def test_stage_idx_not_exceeds_patchcore_max(self, tmp_path):
        """PatchCore 모델의 stage_idx는 최대 6(7단계)을 초과하지 않아야 한다."""
        q = queue.Queue()
        pc_config = {
            "model_type": "patchcore",
            "image_size": 64,
            "batch_size": 2,
            "random_seed": 42,
            "params": {
                "backbone": "wide_resnet50_2",
                "coreset_sampling_ratio": 0.1,
                "neighbourhood_kernel_size": 3,
                "pretrained_source": "torchvision",
                "max_train": 10,
                "knn": 9,
                "top_k_ratio": 0.1,
                "pretrained_path": None,
            },
        }
        worker = _make_worker(tmp_path, q=q, model_config=pc_config)
        worker.start()
        worker.join(timeout=5)

        msgs = _collect_all_messages(q)
        for msg in (m for m in msgs if m["type"] == "stage"):
            assert msg["stage_idx"] <= 6, f"PatchCore stage_idx {msg['stage_idx']} > 6"
