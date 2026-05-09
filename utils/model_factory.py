from __future__ import annotations

import queue
import threading

import numpy as np
import torch


def create_trainer(
    model_config: dict,
    preprocessing_config: dict,
    dataset_path: str,
    device: str,
    experiment_id: str,
    stop_event: threading.Event,
    result_queue: queue.Queue,
) -> "TrainingWorker":
    from utils.training_worker import TrainingWorker

    return TrainingWorker(
        experiment_id=experiment_id,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        dataset_path=dataset_path,
        device=device,
        stop_event=stop_event,
        result_queue=result_queue,
    )


def load_model_for_inference(
    exp_id: str,
    model_path: str,
    model_config: dict,
    device: str,
) -> object:
    # TODO: 08_AI_ML_Integration.md 구현 예정
    raise NotImplementedError(
        "load_model_for_inference: 08_AI_ML_Integration.md 구현 예정"
    )


def run_inference(
    model: object,
    image_tensor: torch.Tensor,
) -> np.ndarray:
    # TODO: 08_AI_ML_Integration.md 구현 예정
    raise NotImplementedError(
        "run_inference: 08_AI_ML_Integration.md 구현 예정"
    )
