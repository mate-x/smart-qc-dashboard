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
        device: str,                 # Z.2: device_info dict 아님, 문자열 직접 전달
        stop_event: threading.Event,
        result_queue: queue.Queue,
    ) -> None:
        super().__init__(daemon=True, name=f"TrainingWorker-{experiment_id}")
        self.experiment_id = experiment_id
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.dataset_path         = dataset_path
        self.device               = device
        self.stop_event           = stop_event
        self.result_queue         = result_queue

        self._model = None
        self._start_time: float = 0.0
        self._log_writer = None
        self._last_step: int = 0

    def run(self) -> None:
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
                "type":      "error",
                "exception": e,
                "traceback": traceback.format_exc(),
            })
            self._write_log(f"[오류] {traceback.format_exc()}")
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

        # 1-b. GPU 커널 호환성 사전 검사 (RTX 5060 / sm_120 등 신형 GPU 대응)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                _probe = torch.zeros(1, device=device)
                del _probe
            except RuntimeError as exc:
                if "no kernel image" in str(exc).lower():
                    cap = torch.cuda.get_device_capability()
                    sm = f"sm_{cap[0]}{cap[1]}"
                    raise RuntimeError(
                        f"GPU({sm})가 현재 PyTorch {torch.__version__}와 호환되지 않습니다. "
                        f"PyTorch 2.7.0 이상으로 업그레이드하세요.\n"
                        f"pip install torch torchvision torchaudio "
                        f"--index-url https://download.pytorch.org/whl/cu128"
                    ) from exc
                raise

        # 2. 시작 로그
        self._write_log(f"[시작] 실험: {self.experiment_id}")
        self._write_log(
            f"[설정] 모델: {self.model_config.get('model_type')} | "
            f"이미지 크기: {self.model_config.get('image_size', '?')} | "
            f"디바이스: {self.device}"
        )

        model_type = self.model_config.get("model_type", "")

        # 3. EfficientAD — imagenet penalty 사전 검증 (Z.1)
        # imagenet_penalty_weight == 0이면 penalty 미사용 → 디렉터리 불필요
        if model_type == "efficientad":
            penalty_weight = self.model_config.get("params", {}).get("imagenet_penalty_weight", 1.0)
            if penalty_weight > 0:
                from utils.storage import validate_imagenet_penalty_dir
                ok, _ = validate_imagenet_penalty_dir()
                if not ok:
                    raise ValueError("ImageNet penalty 디렉터리에 이미지가 없습니다.")

        # 4. DataLoader 구성
        self._write_log("[초기화] 데이터셋 로딩 중...")
        from utils.mvtec_dataset import build_dataloaders
        train_loader, test_loader = build_dataloaders(
            dataset_path=self.dataset_path,
            preprocessing_config=self.preprocessing_config,
            batch_size=self.model_config.get("batch_size", 16),
            random_seed=seed,
        )
        self._write_log(
            f"[초기화] 데이터셋 로딩 완료 — "
            f"train {len(train_loader.dataset)}장 / test {len(test_loader.dataset)}장"
        )

        # 5. 모델 생성 + 학습
        from utils.model_factory import (
            _create_efficientad_model,
            _create_patchcore_model,
            build_imagenet_penalty_loader,
        )

        if model_type == "efficientad":
            self._write_log("[초기화] EfficientAD 모델 생성 중... (사전학습 가중치 로딩 포함, 수십 초 소요)")
            model = _create_efficientad_model(self.model_config)
            if penalty_weight > 0:
                penalty_loader = build_imagenet_penalty_loader(
                    batch_size=self.model_config["params"].get("penalty_batch_size", 8),
                    image_size=self.model_config.get("image_size", 256),
                    device=self.device,
                )
            else:
                penalty_loader = None
            self._write_log("[초기화] EfficientAD 모델 준비 완료")
            completed, last_step = self._train_efficientad(
                model, train_loader, penalty_loader, device
            )

        elif model_type == "patchcore":
            self._write_log("[초기화] PatchCore 모델 생성 중...")
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
        penalty_loader: "DataLoader | None",
        device: torch.device,
    ) -> tuple[bool, int]:
        """
        반환: (completed: bool, last_step: int)
        completed=False: stop_event로 중단
        """
        from utils.model_factory import _efficientad_training_step

        params = self.model_config["params"]
        total_steps = params.get("train_steps", 70000)
        report_every = 100

        model = model.to(device)
        model.train()
        self._write_log("[학습] EfficientAD GPU 전송 완료. 학습 루프 시작...")

        # anomalib 버전별로 student/autoencoder 위치가 다름
        # 2.4.x: model.model.student / model.model.ae
        # 구버전: model.student / model.autoencoder
        _inner = getattr(model, "model", model)
        student = getattr(model, "student", getattr(_inner, "student", None))
        autoencoder = (
            getattr(model, "autoencoder", None) or getattr(model, "ae", None)
            or getattr(_inner, "autoencoder", None) or getattr(_inner, "ae", None)
        )

        if student is None or autoencoder is None:
            raise AttributeError(
                f"EfficientAd 모델에서 student/autoencoder를 찾을 수 없습니다. "
                f"model 속성: {[a for a in dir(model) if not a.startswith('_')]}"
            )

        optimizer_st = self._build_optimizer(student, params)
        optimizer_ae = self._build_optimizer(
            autoencoder,
            {
                **params,
                "learning_rate": params.get("autoencoder_lr", params.get("learning_rate", 1e-4)),
                "weight_decay": params.get("autoencoder_weight_decay", 1e-5),
            },
        )
        scheduler_st = self._build_scheduler(optimizer_st, params, total_steps)
        scheduler_ae = self._build_scheduler(optimizer_ae, params, total_steps)

        train_iter = self._infinite_loader(train_loader)
        penalty_iter = self._infinite_loader(penalty_loader) if penalty_loader is not None else None

        step = 0
        last_loss = 0.0

        while step < total_steps:
            if self.stop_event.is_set():
                return False, step

            batch = next(train_iter)
            images = batch["image"].to(device)

            if penalty_iter is not None:
                penalty_batch = next(penalty_iter)
                # FakeData / ImageFolder 모두 (img, label) 튜플 반환
                if isinstance(penalty_batch, (list, tuple)):
                    penalty = penalty_batch[0].to(device)
                else:
                    penalty = penalty_batch["image"].to(device)
            else:
                penalty = torch.zeros_like(images)

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

            if step == 1 or step % report_every == 0 or step == total_steps:
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
        PatchCore 학습: 단일 에포크 특징 추출 후 coreset 메모리 뱅크 구성.

        anomalib 2.4.x: PatchcoreModel.forward(training=True)가 embedding_store에 자동 축적,
                         model.fit()으로 coreset 구성.
        구버전 fallback: _extract_patchcore_features로 수동 추출 후 memory_bank 직접 설정.
        """
        params = self.model_config.get("params", {})
        max_train = params.get("max_train", 1000)

        model = model.to(device)
        torch_model = getattr(model, "model", None)

        batch_size = train_loader.batch_size or 1
        total_batches = min(len(train_loader), max_train // max(batch_size, 1) + 1)

        self._write_log(
            f"[학습] PatchCore 특징 추출 시작 — 총 {total_batches}배치"
        )
        self.result_queue.put({
            "type": "progress", "step": 0,
            "total": total_batches, "loss": 0.0,
            "elapsed": round(time.time() - self._start_time, 1),
        })

        if torch_model is not None:
            # anomalib 2.4.x: training mode에서 torch_model(images) → embedding_store 축적
            model.train()
            with torch.no_grad():
                for batch_idx, batch in enumerate(train_loader):
                    if self.stop_event.is_set():
                        return False, batch_idx
                    if batch_idx >= total_batches:
                        break
                    images = batch["image"].to(device)
                    torch_model(images)
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
            self._write_log("[PatchCore] 특징 추출 완료. Coreset 구성 중...")
            model.eval()
            model.fit()  # embedding_store → coreset subsample → memory_bank

        else:
            # Fallback: 구버전 anomalib — 수동 feature 추출
            from utils.model_factory import _extract_patchcore_features

            model.eval()
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
            coreset_ratio = params.get("coreset_sampling_ratio", 0.1)
            coreset_size = max(1, int(len(feature_stack) * coreset_ratio))
            indices = torch.randperm(len(feature_stack))[:coreset_size]
            model.memory_bank = feature_stack[indices].to(device)

        elapsed_total = time.time() - self._start_time
        self._write_log(f"[완료] 메모리 뱅크 구성 완료 | 경과: {elapsed_total:.1f}s")
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
