from __future__ import annotations

import queue
import threading
import time
import traceback


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
        model_config: dict,
        preprocessing_config: dict,
        dataset_path: str,
        device: str,
        exp_id: str,
        stop_event: threading.Event,
        result_queue: queue.Queue,
    ) -> None:
        super().__init__(daemon=True)
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.dataset_path = dataset_path
        self.device = device
        self.exp_id = exp_id
        self.stop_event = stop_event
        self.result_queue = result_queue

    def run(self) -> None:
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
