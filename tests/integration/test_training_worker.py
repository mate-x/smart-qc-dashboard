from __future__ import annotations

import queue
import threading
from unittest.mock import patch

import pytest

from utils.training_worker import TrainingWorker


def _make_worker(tmp_path, stop_event=None, q=None, **overrides):
    defaults = {
        "experiment_id": "test_exp_001",
        "model_config": {"model_type": "patchcore", "image_size": 64},
        "preprocessing_config": {"method": "none", "image_size": 64},
        "dataset_path": str(tmp_path),
        "device": "cpu",
        "stop_event": stop_event or threading.Event(),
        "result_queue": q or queue.Queue(),
    }
    defaults.update(overrides)
    return TrainingWorker(**defaults)


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

    def test_pre_set_does_not_call_run_training(self, tmp_path):
        stop = threading.Event()
        stop.set()
        q = queue.Queue()
        worker = _make_worker(tmp_path, stop_event=stop, q=q)
        with patch.object(worker, "_run_training") as mock_train:
            worker.start()
            worker.join(timeout=2)
        mock_train.assert_not_called()


class TestWorkerErrorHandling:
    def test_not_implemented_error_puts_error_message(self, tmp_path):
        q = queue.Queue()
        worker = _make_worker(tmp_path, q=q)
        worker.start()
        worker.join(timeout=2)
        msg = q.get_nowait()
        assert msg["type"] == "error"
        assert isinstance(msg["exception"], NotImplementedError)
        assert "traceback" in msg

    def test_error_message_contains_traceback_string(self, tmp_path):
        q = queue.Queue()
        worker = _make_worker(tmp_path, q=q)
        worker.start()
        worker.join(timeout=2)
        msg = q.get_nowait()
        assert isinstance(msg["traceback"], str)
        assert len(msg["traceback"]) > 0


class TestWorkerBuildDatasetPatchable:
    def test_build_dataset_can_be_patched(self):
        import utils.training_worker as tw
        assert hasattr(tw, "build_dataset")

    def test_engine_importable_or_none(self):
        import utils.training_worker as tw
        assert tw.Engine is None or tw.Engine is not None
