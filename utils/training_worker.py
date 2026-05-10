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
from torch.utils.data import DataLoader

# Anomalib import — 없으면 None
try:
    from anomalib.engine import Engine
except ImportError:
    Engine = None  # type: ignore[assignment,misc]

KST = timezone(timedelta(hours=9))


def _now_kst() -> str:
    return datetime.now(tz=KST).strftime("%Y-%m-%dT%H:%M:%S+09:00")


def build_dataset(model_config: dict, preprocessing_config: dict, dataset_path: str):
    """Placeholder — mvtec_dataset.build_dataloaders 사용."""
    from utils.mvtec_dataset import build_dataloaders

    return build_dataloaders(
        dataset_path=dataset_path,
        preprocessing_config=preprocessing_config,
        batch_size=model_config.get("batch_size", 16),
        random_seed=model_config.get("random_seed", 42),
    )


class TrainingWorker(threading.Thread):
    """
    백그라운드 학습 스레드 (04_System_Architecture B.5절).

    Queue 메시지 타입:
        {"type": "progress", "step": int, "total": int, "loss": float, "elapsed": float}
        {"type": "log", "message": str}
        {"type": "completed", "y_true": list, "anomaly_scores": list,
         "anomaly_maps": dict, "image_paths": list, "model": object,
         "duration_seconds": int}
        {"type": "error", "exception": Exception, "traceback": str}
        {"type": "stopped", "step": int}
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
        super().__init__(daemon=True, name=f"TrainingWorker-{experiment_id}")
        self.experiment_id = experiment_id
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.dataset_path = dataset_path
        self.device = device
        self.stop_event = stop_event
        self.result_queue = result_queue

        self._model = None
        self._start_time: float = 0.0
        self._log_writer = None
        self._last_step: int = 0

    def run(self) -> None:
        # stop_event 선행 확인 (시작 전 중단 요청)
        if self.stop_event.is_set():
            self.result_queue.put({"type": "stopped", "step": 0})
            return

        self._start_time = time.time()
        from utils.storage import get_log_writer
        self._log_writer = get_log_writer(self.experiment_id)

        try:
            self._run_impl()
        except Exception as e:
            self.result_queue.put({
                "type": "error",
                "exception": e,
                "traceback": traceback.format_exc(),
            })
            self._write_log(f"[오류] {traceback.format_exc()[:500]}")
        finally:
            if self._log_writer:
                try:
                    self._log_writer.close()
                except OSError:
                    pass

    def _write_log(self, message: str) -> None:
        """타임스탬프 + 메시지를 로그 파일과 Queue(log)에 동시 기록."""
        ts = datetime.now(tz=KST).strftime("%Y-%m-%dT%H:%M:%S+09:00")
        line = f"{ts}\t{message}"
        if self._log_writer:
            try:
                self._log_writer.write(line + "\n")
            except OSError:
                pass
        self.result_queue.put({"type": "log", "message": message})

    def _run_impl(self) -> None:
        # 1. 재현성 시드 고정 (R-SEED-01)
        seed = self.model_config.get("random_seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        device = torch.device(self.device)

        # 2. 시작 로그
        self._write_log(f"[시작] 실험: {self.experiment_id}")
        self._write_log(
            f"[설정] 모델: {self.model_config.get('model_type')} | "
            f"이미지 크기: {self.model_config.get('image_size', '?')} | "
            f"디바이스: {self.device}"
        )

        model_type = self.model_config.get("model_type", "")

        # 3. EfficientAD — imagenet penalty 사전 검증 (Z.1)
        if model_type == "efficientad":
            from utils.storage import validate_imagenet_penalty_dir
            validate_imagenet_penalty_dir()  # ValueError → error 메시지로 전파

        # 4. DataLoader 구성
        from utils.mvtec_dataset import build_dataloaders
        train_loader, test_loader = build_dataloaders(
            dataset_path=self.dataset_path,
            preprocessing_config=self.preprocessing_config,
            batch_size=self.model_config.get("batch_size", 16),
            random_seed=seed,
        )

        # 5. 모델 생성 + 학습
        from utils.model_factory import (
            _create_efficientad_model,
            _create_patchcore_model,
            build_imagenet_penalty_loader,
        )

        if model_type == "efficientad":
            model = _create_efficientad_model(self.model_config)
            penalty_loader = build_imagenet_penalty_loader(
                batch_size=self.model_config["params"].get("penalty_batch_size", 8),
                image_size=self.model_config.get("image_size", 256),
                device=self.device,
            )
            self._write_log("[초기화] EfficientAD 모델 준비 완료")
            completed, last_step = self._train_efficientad(
                model, train_loader, penalty_loader, device
            )

        elif model_type == "patchcore":
            model = _create_patchcore_model(self.model_config)
            self._write_log("[초기화] PatchCore 모델 준비 완료")
            completed, last_step = self._train_patchcore(model, train_loader, device)

        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

        if not completed:
            self.result_queue.put({"type": "stopped", "step": last_step})
            return

        # 6. 전체 테스트셋 추론
        self._write_log("[평가] 테스트셋 추론 중...")
        y_true, anomaly_scores, anomaly_maps = self._run_full_test_inference(
            model, test_loader, device
        )

        if self.stop_event.is_set():
            self.result_queue.put({"type": "stopped", "step": last_step})
            return

        # 7. 완료 메시지
        elapsed = time.time() - self._start_time
        image_paths = list(anomaly_maps.keys())
        self._write_log(
            f"[결과] 테스트 이미지 {len(y_true)}장 완료 | 소요: {elapsed:.1f}s"
        )

        self.result_queue.put({
            "type":            "completed",
            "y_true":          y_true,
            "anomaly_scores":  anomaly_scores,
            "anomaly_maps":    anomaly_maps,
            "image_paths":     image_paths,
            "model":           model.cpu(),
            "duration_seconds": int(elapsed),
        })

    # ── EfficientAD 학습 루프 ──────────────────────────────────────────────────

    def _train_efficientad(
        self,
        model: object,
        train_loader: DataLoader,
        penalty_loader: DataLoader,
        device: torch.device,
    ) -> tuple[bool, int]:
        """
        반환: (completed: bool, last_step: int)
        completed=False: stop_event로 중단
        """
        from utils.model_factory import _efficientad_training_step

        params = self.model_config["params"]
        total_steps = params.get("train_steps", 70000)
        report_every = 500

        model = model.to(device)
        model.train()

        optimizer_st = self._build_optimizer(model.student, params)
        optimizer_ae = self._build_optimizer(
            model.autoencoder,
            {
                **params,
                "learning_rate": params.get("autoencoder_lr", params.get("learning_rate", 1e-4)),
                "weight_decay": params.get("autoencoder_weight_decay", 1e-5),
            },
        )
        scheduler_st = self._build_scheduler(optimizer_st, params, total_steps)
        scheduler_ae = self._build_scheduler(optimizer_ae, params, total_steps)

        train_iter = self._infinite_loader(train_loader)
        penalty_iter = self._infinite_loader(penalty_loader)

        step = 0
        last_loss = 0.0

        while step < total_steps:
            if self.stop_event.is_set():
                return False, step

            batch = next(train_iter)
            penalty_batch = next(penalty_iter)

            images = batch["image"].to(device)
            # FakeData / ImageFolder 모두 (img, label) 튜플 반환
            if isinstance(penalty_batch, (list, tuple)):
                penalty = penalty_batch[0].to(device)
            else:
                penalty = penalty_batch["image"].to(device)

            loss_dict = _efficientad_training_step(model, images, penalty)
            total_loss = loss_dict["loss_total"]
            last_loss = total_loss.item()

            optimizer_st.zero_grad()
            optimizer_ae.zero_grad()
            total_loss.backward()
            optimizer_st.step()
            optimizer_ae.step()
            scheduler_st.step()
            scheduler_ae.step()

            step += 1

            if step % report_every == 0 or step == total_steps:
                elapsed = time.time() - self._start_time
                self.result_queue.put({
                    "type":    "progress",
                    "step":    step,
                    "total":   total_steps,
                    "loss":    round(last_loss, 6),
                    "elapsed": round(elapsed, 1),
                })
                self._write_log(
                    f"[Step {step}/{total_steps}] Loss: {last_loss:.4f} | "
                    f"경과: {elapsed:.1f}s"
                )

        self._last_step = step
        return True, step

    # ── PatchCore 학습 루프 ────────────────────────────────────────────────────

    def _train_patchcore(
        self,
        model: object,
        train_loader: DataLoader,
        device: torch.device,
    ) -> tuple[bool, int]:
        """
        PatchCore는 단일 에포크 특징 추출 후 메모리 뱅크 구성.
        반환: (completed: bool, 0)
        """
        from utils.model_factory import _extract_patchcore_features

        params = self.model_config.get("params", {})
        max_train = params.get("max_train", 1000)

        model = model.to(device)
        model.eval()

        batch_size = train_loader.batch_size or 1
        total_batches = min(len(train_loader), max_train // max(batch_size, 1) + 1)
        all_features: list[torch.Tensor] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                if self.stop_event.is_set():
                    return False, batch_idx

                if batch_idx >= total_batches:
                    break

                images = batch["image"].to(device)
                features = _extract_patchcore_features(model, images)
                all_features.append(features.cpu())

                elapsed = time.time() - self._start_time
                self.result_queue.put({
                    "type":    "progress",
                    "step":    batch_idx + 1,
                    "total":   total_batches,
                    "loss":    0.0,
                    "elapsed": round(elapsed, 1),
                })
                self._write_log(
                    f"[배치 {batch_idx+1}/{total_batches}] 특징 추출 중 | "
                    f"경과: {elapsed:.1f}s"
                )

        if not all_features:
            raise ValueError("특징 추출된 배치가 없습니다. 데이터셋을 확인해 주세요.")

        feature_stack = torch.cat(all_features, dim=0)
        self._write_log("[PatchCore] 특징 추출 완료. Coreset 구성 중...")

        if hasattr(model, "fit"):
            model.fit(feature_stack)
        else:
            coreset_ratio = params.get("coreset_sampling_ratio", 0.1)
            coreset_size = max(1, int(len(feature_stack) * coreset_ratio))
            indices = torch.randperm(len(feature_stack))[:coreset_size]
            model.memory_bank = feature_stack[indices].to(device)

        elapsed_total = time.time() - self._start_time
        self._write_log(f"[완료] 메모리 뱅크 구성 완료 | 경과: {elapsed_total:.1f}s")
        # coreset 완료 = 학습 완료 (step=1)
        self.result_queue.put({
            "type":    "progress",
            "step":    1,
            "total":   1,
            "loss":    0.0,
            "elapsed": round(elapsed_total, 1),
        })

        self._last_step = 1
        return True, 1

    # ── 전체 테스트셋 추론 ─────────────────────────────────────────────────────

    def _run_full_test_inference(
        self,
        model: object,
        test_loader: DataLoader,
        device: torch.device,
    ) -> tuple[list[int], list[float], dict[str, np.ndarray]]:
        """
        반환:
          y_true:         list[int]
          anomaly_scores: list[float]
          anomaly_maps:   dict[str → np.ndarray(H,W)]
        """
        from utils.model_factory import _get_anomaly_map

        model = model.to(device)
        model.eval()

        y_true: list[int] = []
        anomaly_scores: list[float] = []
        anomaly_maps: dict[str, np.ndarray] = {}

        with torch.no_grad():
            for batch in test_loader:
                if self.stop_event.is_set():
                    return y_true, anomaly_scores, anomaly_maps

                image = batch["image"].to(device)
                label = int(batch["label"].item())
                image_path = batch["image_path"][0]

                amap = _get_anomaly_map(model, image)
                score = float(amap.max())

                y_true.append(label)
                anomaly_scores.append(round(score, 6))
                anomaly_maps[image_path] = amap

        return y_true, anomaly_scores, anomaly_maps

    # ── 학습 헬퍼 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_optimizer(
        model_component,
        config: dict,
    ) -> torch.optim.Optimizer:
        opt_name = config.get("optimizer", "adam").lower()
        lr = config.get("learning_rate", 1e-4)
        wd = config.get("weight_decay", 1e-4)
        if opt_name == "adam":
            return torch.optim.Adam(model_component.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return torch.optim.AdamW(model_component.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return torch.optim.SGD(
                model_component.parameters(), lr=lr, weight_decay=wd, momentum=0.9
            )
        raise ValueError(f"지원하지 않는 옵티마이저: {opt_name}")

    @staticmethod
    def _build_scheduler(
        optimizer: torch.optim.Optimizer,
        params: dict,
        total_steps: int,
    ) -> "torch.optim.lr_scheduler._LRScheduler":
        scheduler_name = params.get("scheduler", "StepLR")
        if scheduler_name == "StepLR":
            step_size = params.get("lr_decay_epochs", 50000)
            gamma = params.get("lr_decay_factor", 0.1)
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_name == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps
            )
        raise ValueError(f"지원하지 않는 스케줄러: {scheduler_name}")

    @staticmethod
    def _infinite_loader(loader: DataLoader):
        """DataLoader를 무한 반복하는 generator."""
        while True:
            for batch in loader:
                yield batch
