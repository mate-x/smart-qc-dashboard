from __future__ import annotations

import queue
import random
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from utils.metrics import compute_all_metrics, compute_threshold

KST = timezone(timedelta(hours=9))


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def _append_log(log_file: Path, line: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class TrainingWorker(threading.Thread):
    """
    백그라운드 학습 스레드 (PRD 08 B.6 / Z.2).

    Queue 메시지 타입:
      {"type": "progress", "step": int, "total": int, "loss": float, "elapsed": float}
      {"type": "log",      "message": str}
      {"type": "completed", "model": object, "y_true": list, "anomaly_scores": list,
                             "anomaly_maps": dict[str, np.ndarray], "image_paths": list[str],
                             "threshold": float, "metrics": dict, "duration_seconds": int}
      {"type": "error",   "exception": Exception, "traceback": str}
      {"type": "stopped", "step": int}
    """

    def __init__(
        self,
        experiment_id: str,
        model_config: dict,
        preprocessing_config: dict,
        dataset_path: str,
        device: str,                 # Z.2: device_info dict 아님, 문자열 직접 전달
        stop_event: threading.Event,
        result_queue: queue.Queue,
    ) -> None:
        super().__init__(daemon=True)
        self.experiment_id        = experiment_id
        self.model_config         = model_config
        self.preprocessing_config = preprocessing_config
        self.dataset_path         = dataset_path
        self.device               = device
        self.stop_event           = stop_event
        self.result_queue         = result_queue
        self._log_writer          = None  # get_log_writer()로 lazy init (Z.2)

    def run(self) -> None:
        if self.stop_event.is_set():
            self.result_queue.put({"type": "stopped", "step": 0})
            return
        try:
            self._run_impl()
        except Exception as e:
            self.result_queue.put({
                "type":      "error",
                "exception": e,
                "traceback": traceback.format_exc(),
            })

    def _run_impl(self) -> None:
        from utils.mvtec_dataset import build_dataloaders
        from utils.model_factory import (
            _create_efficientad_model,
            _create_patchcore_model,
            _get_anomaly_map,
        )

        log_file = Path(f"./logs/{self.experiment_id}.log")

        # R-SEED-01: 재현성 시드 고정
        seed = self.model_config.get("random_seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device(self.device)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)

        start_msg = (
            f"{_now_kst()}\t[시작] 실험: {self.experiment_id} | "
            f"모델: {self.model_config.get('model_type', '?')} | "
            f"디바이스: {self.device}"
        )
        self.result_queue.put({"type": "log", "message": start_msg})
        _append_log(log_file, start_msg)

        # DataLoader 구성
        train_loader, test_loader = build_dataloaders(
            dataset_path=self.dataset_path,
            preprocessing_config=self.preprocessing_config,
            batch_size=self.model_config.get("batch_size", 32),
            random_seed=seed,
        )

        model_type = self.model_config.get("model_type", "patchcore")
        start_time = time.time()

        if model_type == "patchcore":
            model = _create_patchcore_model(self.model_config)
            init_msg = f"{_now_kst()}\t[초기화] patchcore 모델 준비 완료"
            self.result_queue.put({"type": "log", "message": init_msg})
            _append_log(log_file, init_msg)

            completed, _, last_step = _train_patchcore(
                model, train_loader,
                self.model_config, device,
                self.stop_event, self.result_queue, log_file,
            )

        elif model_type == "efficientad":
            raise NotImplementedError("EfficientAD 학습 루프는 별도 구현 예정입니다.")

        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

        if not completed:
            self.result_queue.put({"type": "stopped", "step": last_step})
            return

        # 테스트셋 추론
        inf_msg = f"{_now_kst()}\t[평가] 테스트셋 추론 중..."
        self.result_queue.put({"type": "log", "message": inf_msg})
        _append_log(log_file, inf_msg)

        y_true, anomaly_scores, anomaly_maps = _run_full_test_inference(
            model, test_loader, device, self.stop_event, _get_anomaly_map
        )

        if self.stop_event.is_set():
            self.result_queue.put({"type": "stopped", "step": last_step})
            return

        # Threshold + 메트릭 계산 (B.8.2)
        normal_scores = [s for s, l in zip(anomaly_scores, y_true) if l == 0]
        threshold = compute_threshold(
            normal_scores,
            self.model_config.get("threshold_method", "percentile"),
            self.model_config.get("threshold_value", 95.0),
        )
        metrics = compute_all_metrics(y_true, anomaly_scores, threshold)

        duration = int(time.time() - start_time)
        image_paths = list(anomaly_maps.keys())

        result_msg = (
            f"{_now_kst()}\t[결과] AUC: {metrics['auc']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | 소요: {duration}s"
        )
        self.result_queue.put({"type": "log", "message": result_msg})
        _append_log(log_file, result_msg)

        # Z.4: completed 메시지에 anomaly_maps + image_paths 포함
        self.result_queue.put({
            "type":            "completed",
            "model":           model.cpu(),
            "y_true":          y_true,
            "anomaly_scores":  anomaly_scores,
            "anomaly_maps":    anomaly_maps,
            "image_paths":     image_paths,
            "threshold":       threshold,
            "metrics":         metrics,
            "duration_seconds": duration,
        })


# ──────────────────────────────────────────────────────────────
# PatchCore 학습 루프 (PRD 08 B.5.2)
# ──────────────────────────────────────────────────────────────

def _train_patchcore(
    model,
    train_loader,
    model_config: dict,
    device: torch.device,
    stop_event: threading.Event,
    result_queue: queue.Queue,
    log_file: Path,
) -> tuple[bool, float, int]:
    """
    단일 에포크 특징 추출 → Coreset 서브샘플링 → 메모리 뱅크 구성.
    반환: (completed, 0.0, last_batch_idx)   — Z.3: last_batch_idx를 stopped.step에 사용
    """
    from utils.model_factory import _extract_patchcore_features

    params = model_config.get("params", {})
    max_train   = params.get("max_train", 1000)
    batch_size  = train_loader.batch_size or 1
    total_batches = min(len(train_loader), max_train // batch_size + 1)

    model = model.to(device)
    model.eval()

    all_features: list[torch.Tensor] = []
    start_time   = time.time()
    last_batch_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if stop_event.is_set():
                return False, 0.0, batch_idx   # Z.3

            if batch_idx >= total_batches:
                break

            images   = batch["image"].to(device)
            features = _extract_patchcore_features(model, images)
            all_features.append(features.cpu())

            elapsed = time.time() - start_time
            result_queue.put({
                "type":    "progress",
                "step":    batch_idx + 1,
                "total":   total_batches,
                "loss":    0.0,
                "elapsed": round(elapsed, 1),
            })
            log_line = (
                f"{_now_kst()}\t"
                f"[배치 {batch_idx + 1}/{total_batches}] "
                f"특징 추출 중 | 경과: {elapsed:.1f}s"
            )
            result_queue.put({"type": "log", "message": log_line})
            _append_log(log_file, log_line)
            last_batch_idx = batch_idx + 1

    if not all_features:
        return False, 0.0, 0

    feature_stack = torch.cat(all_features, dim=0)   # (N_total, C)

    # Coreset 서브샘플링
    coreset_ratio = params.get("coreset_sampling_ratio", 0.1)
    coreset_size  = max(1, int(len(feature_stack) * coreset_ratio))
    if coreset_size < len(feature_stack):
        indices     = torch.randperm(len(feature_stack))[:coreset_size]
        memory_bank = feature_stack[indices]
    else:
        memory_bank = feature_stack

    # 메모리 뱅크 등록: register_buffer로 state_dict에 포함되도록 처리
    try:
        model.register_buffer("memory_bank", memory_bank.to(device))
    except Exception:
        model.memory_bank = memory_bank.to(device)

    elapsed_total = time.time() - start_time
    done_line = (
        f"{_now_kst()}\t[완료] 메모리 뱅크 {len(memory_bank)}개 벡터 구성 완료 | "
        f"경과: {elapsed_total:.1f}s"
    )
    result_queue.put({"type": "log", "message": done_line})
    _append_log(log_file, done_line)

    return True, 0.0, last_batch_idx


# ──────────────────────────────────────────────────────────────
# 전체 테스트셋 추론 (PRD 08 B.7.1)
# ──────────────────────────────────────────────────────────────

def _run_full_test_inference(
    model,
    test_loader,
    device: torch.device,
    stop_event: threading.Event,
    get_anomaly_map_fn,
) -> tuple[list[int], list[float], dict]:
    """
    반환:
      y_true:         list[int]
      anomaly_scores: list[float]   — 이미지 레벨 score = anomaly_map.max()
      anomaly_maps:   dict[str, np.ndarray(H,W)]   — key = 이미지 경로
    """
    model = model.to(device)
    model.eval()

    y_true:         list[int]   = []
    anomaly_scores: list[float] = []
    anomaly_maps:   dict        = {}

    with torch.no_grad():
        for batch in test_loader:
            if stop_event.is_set():
                return y_true, anomaly_scores, anomaly_maps

            image      = batch["image"].to(device)
            label      = int(batch["label"].item())
            image_path = batch["image_path"][0]

            amap  = get_anomaly_map_fn(model, image)   # (H, W) float32
            score = float(amap.max())

            y_true.append(label)
            anomaly_scores.append(round(score, 6))
            anomaly_maps[image_path] = amap

    return y_true, anomaly_scores, anomaly_maps
