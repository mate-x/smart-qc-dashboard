from __future__ import annotations

import queue
import threading
import traceback

# 08_AI_ML_Integration.md 구현 시 실제 import로 교체
try:
    from anomalib.engine import Engine
except ImportError:
    Engine = None  # type: ignore[assignment,misc]


def build_dataset(model_config: dict, preprocessing_config: dict, dataset_path: str):
    """Placeholder — 08_AI_ML_Integration.md 구현 예정."""
    raise NotImplementedError("build_dataset: 08_AI_ML_Integration.md 구현 예정")


class TrainingWorker(threading.Thread):
    """
    백그라운드 학습 스레드 (04_System_Architecture B.3.3 / B.5절).

    Queue 메시지 타입:
        {"type": "progress", "step": int, "total": int, "loss": float, "elapsed": float}
        {"type": "log", "message": str}
        {"type": "completed", "y_true": list, "scores": list, "model": object}
        {"type": "error", "exception": Exception, "traceback": str}
        {"type": "stopped"}
    """

    def __init__(
        self,
        experiment_id: str,
        model_config: dict,
        preprocessing_config: dict,
        dataset_path: str,
        device: str,
        stop_event: threading.Event,
        result_queue: queue.Queue,
    ) -> None:
        super().__init__(daemon=True)
        self.experiment_id = experiment_id
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.dataset_path = dataset_path
        self.device = device
        self.stop_event = stop_event
        self.result_queue = result_queue

    def run(self) -> None:
        # stop_event 선행 확인 (시작 전 중단 요청)
        if self.stop_event.is_set():
            self.result_queue.put({"type": "stopped"})
            return
        try:
            self._run_training()
        except Exception as e:
            self.result_queue.put({
                "type": "error",
                "exception": e,
                "traceback": traceback.format_exc(),
            })

    def _run_training(self) -> None:
        # TODO: 08_AI_ML_Integration.md 구현 시 실제 학습 로직으로 교체
        raise NotImplementedError(
            "_run_training: 08_AI_ML_Integration.md 구현 예정"
        )
