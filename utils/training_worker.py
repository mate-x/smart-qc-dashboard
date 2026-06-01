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

# FR-T3-11: EfficientAD 학습 단계 정의 (5단계)
EFFICIENTAD_STAGES: list[tuple[int, str]] = [
    (0, "데이터 로딩"),
    (1, "모델 초기화"),
    (2, "학습 루프"),
    (3, "테스트 추론"),
    (4, "완료"),
]

# FR-T3-12: PatchCore 학습 단계 정의 (7단계)
PATCHCORE_STAGES: list[tuple[int, str]] = [
    (0, "데이터 로딩"),
    (1, "모델 초기화"),
    (2, "특징 추출"),
    (3, "Coreset 구성"),
    (4, "Memory Bank 설정"),
    (5, "테스트 추론"),
    (6, "완료"),
]


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
        {"type": "paused", "step": int, "ckpt_path": str}
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
        # 일시정지 지원
        pause_event: threading.Event | None = None,
        # EfficientAD resume 파라미터
        start_step: int = 0,
        student_state_dict: dict | None = None,
        autoencoder_state_dict: dict | None = None,
        optimizer_st_state_dict: dict | None = None,
        optimizer_ae_state_dict: dict | None = None,
        scheduler_st_state_dict: dict | None = None,
        scheduler_ae_state_dict: dict | None = None,
        loss_history: list | None = None,
        # PatchCore resume 파라미터
        start_batch_idx: int = 0,
        accumulated_features: "torch.Tensor | None" = None,
    ) -> None:
        super().__init__(daemon=True, name=f"TrainingWorker-{experiment_id}")
        self.experiment_id        = experiment_id
        self.model_config         = model_config
        self.preprocessing_config = preprocessing_config
        self.dataset_path         = dataset_path
        self.device               = device
        self.stop_event           = stop_event
        self.result_queue         = result_queue
        self.pause_event          = pause_event if pause_event is not None else threading.Event()

        # EfficientAD resume
        self.start_step               = start_step
        self.student_state_dict       = student_state_dict
        self.autoencoder_state_dict   = autoencoder_state_dict
        self.optimizer_st_state_dict  = optimizer_st_state_dict
        self.optimizer_ae_state_dict  = optimizer_ae_state_dict
        self.scheduler_st_state_dict  = scheduler_st_state_dict
        self.scheduler_ae_state_dict  = scheduler_ae_state_dict
        self.loss_history: list       = list(loss_history) if loss_history else []

        # PatchCore resume
        self.start_batch_idx     = start_batch_idx
        self.accumulated_features = accumulated_features

        self._model      = None
        self._start_time: float = 0.0
        self._log_writer = None
        self._last_step: int = 0

    # ── 스레드 진입점 ──────────────────────────────────────────────────────────

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

    def _emit_stage(self, stage_idx: int, stage_name: str) -> None:
        """학습 단계 전환 시 stage 메시지를 Queue에 전송 (FR-T3-11, FR-T3-12)."""
        self.result_queue.put({
            "type": "stage",
            "stage_idx": stage_idx,
            "stage_name": stage_name,
        })

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
        if self.start_step > 0:
            self._write_log(f"[재개] EfficientAD step {self.start_step}부터 학습 재시작")
        if self.start_batch_idx > 0:
            self._write_log(f"[재개] PatchCore batch {self.start_batch_idx}부터 특징 추출 재시작")

        model_type = self.model_config.get("model_type", "")

        # 3. EfficientAD — 사전 검증
        use_imagenet_penalty = False
        if model_type == "efficientad":
            # 3-a. image_size 최소값 검사 (AutoEncoder 인코더 구조 제약)
            # 인코더가 입력을 2^5=32배 줄이고 마지막 Conv 커널=8×8이므로
            # 최소 image_size = 8 × 32 = 256 필요
            image_size = self.model_config.get("image_size", 256)
            if image_size < 256:
                raise ValueError(
                    f"EfficientAD는 image_size ≥ 256이 필요합니다. "
                    f"현재 설정: {image_size}px. "
                    f"탭2에서 image_size를 256 이상으로 변경한 뒤 다시 시도해 주세요."
                )

            # 3-b. imagenet penalty 사전 검증 (Z.1)
            use_imagenet_penalty = self.model_config.get("params", {}).get("use_imagenet_penalty", False)
            if use_imagenet_penalty:
                from utils.storage import validate_imagenet_penalty_dir
                ok, _ = validate_imagenet_penalty_dir()
                if not ok:
                    raise ValueError("ImageNet penalty 디렉터리에 이미지가 없습니다.")

        # 4. DataLoader 구성
        self._emit_stage(0, "데이터 로딩")
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
        self._emit_stage(1, "모델 초기화")

        # 5. 모델 생성 + 학습
        from utils.model_factory import (
            _create_efficientad_model,
            _create_patchcore_model,
            build_imagenet_penalty_loader,
        )

        if model_type == "efficientad":
            self._write_log("[초기화] EfficientAD 모델 생성 중... (사전학습 가중치 로딩 포함, 수십 초 소요)")
            model = _create_efficientad_model(self.model_config)
            penalty_loader = None
            if use_imagenet_penalty:
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
        _test_stage_idx = 3 if model_type == "efficientad" else 5
        self._emit_stage(_test_stage_idx, "테스트 추론")
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

        _done_stage_idx = 4 if model_type == "efficientad" else 6
        self._emit_stage(_done_stage_idx, "완료")
        self.result_queue.put({
            "type":             "completed",
            "y_true":           y_true,
            "anomaly_scores":   anomaly_scores,
            "anomaly_maps":     anomaly_maps,
            "image_paths":      image_paths,
            "model":            model.cpu(),
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
        completed=False: stop_event 또는 pause_event(→중지) 로 중단

        resume 지원:
          - start_step: 이 step부터 학습 시작 (이전 배치 소비 없이 바로 진행)
          - student/autoencoder/optimizer/scheduler 상태 복원
          - loss_history: 이전 Loss 곡선 이어붙이기
        """
        from utils.model_factory import _efficientad_training_step
        from utils.checkpoint_manager import save_checkpoint

        params      = self.model_config["params"]
        total_steps = params.get("train_steps", 70000)
        report_every = 100

        model = model.to(device)
        model.train()
        self._emit_stage(2, "학습 루프")
        self._write_log("[학습] EfficientAD GPU 전송 완료. 학습 루프 시작...")

        # anomalib 버전별로 student/autoencoder 위치가 다름
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
                "weight_decay":  params.get("autoencoder_weight_decay", 1e-5),
            },
        )
        scheduler_st = self._build_scheduler(optimizer_st, params, total_steps)
        scheduler_ae = self._build_scheduler(optimizer_ae, params, total_steps)

        # ── resume: 가중치 및 옵티마이저/스케줄러 상태 복원 ─────────────────
        if self.student_state_dict:
            try:
                student.load_state_dict(self.student_state_dict)
                self._write_log("[재개] Student 가중치 복원 완료")
            except RuntimeError:
                self._write_log("[경고] Student 가중치 불일치 — 초기값으로 시작")
        if self.autoencoder_state_dict:
            try:
                autoencoder.load_state_dict(self.autoencoder_state_dict)
                self._write_log("[재개] Autoencoder 가중치 복원 완료")
            except RuntimeError:
                self._write_log("[경고] Autoencoder 가중치 불일치 — 초기값으로 시작")
        for opt, state in [
            (optimizer_st, self.optimizer_st_state_dict),
            (optimizer_ae, self.optimizer_ae_state_dict),
        ]:
            if state:
                try:
                    opt.load_state_dict(state)
                except (ValueError, KeyError):
                    pass
        for sch, state in [
            (scheduler_st, self.scheduler_st_state_dict),
            (scheduler_ae, self.scheduler_ae_state_dict),
        ]:
            if state:
                try:
                    sch.load_state_dict(state)
                except (ValueError, KeyError):
                    pass

        train_iter   = self._infinite_loader(train_loader)
        penalty_iter = self._infinite_loader(penalty_loader) if penalty_loader is not None else None

        use_amp = device.type == "cuda"
        scaler  = torch.cuda.amp.GradScaler() if use_amp else None

        step      = self.start_step   # resume 시 이미 완료된 step부터 시작
        last_loss = 0.0
        _ckpt_saved = False           # 동일 pause에서 중복 저장 방지

        while step < total_steps:
            # ① 중지 체크
            if self.stop_event.is_set():
                return False, step

            # ② 일시정지 체크
            if self.pause_event.is_set():
                if not _ckpt_saved:
                    ckpt_path = save_checkpoint(
                        data={
                            "model_type":             "efficientad",
                            "experiment_id":          self.experiment_id,
                            "step":                   step,
                            "total_steps":            total_steps,
                            "model_config":           self.model_config,
                            "preprocessing_config":   self.preprocessing_config,
                            "dataset_path":           self.dataset_path,
                            "student_state_dict":     student.state_dict(),
                            "autoencoder_state_dict": autoencoder.state_dict(),
                            "optimizer_st_state_dict": optimizer_st.state_dict(),
                            "optimizer_ae_state_dict": optimizer_ae.state_dict(),
                            "scheduler_st_state_dict": scheduler_st.state_dict(),
                            "scheduler_ae_state_dict": scheduler_ae.state_dict(),
                            "loss_history":           self.loss_history,
                        },
                        exp_id=self.experiment_id,
                        label=step,
                    )
                    self._write_log(f"[체크포인트] 저장 완료: {ckpt_path.name}")
                    self.result_queue.put({
                        "type":     "paused",
                        "step":     step,
                        "ckpt_path": str(ckpt_path),
                    })
                    _ckpt_saved = True

                # pause_event 해제될 때까지 대기
                while self.pause_event.is_set():
                    time.sleep(0.1)
                    if self.stop_event.is_set():
                        return False, step
                _ckpt_saved = False
                self._write_log(f"[재개] step {step}부터 학습 재시작")

            # ── 학습 스텝
            batch  = next(train_iter)
            images = batch["image"].to(device, non_blocking=True)

            if penalty_iter is not None:
                penalty_batch = next(penalty_iter)
                if isinstance(penalty_batch, (list, tuple)):
                    penalty = penalty_batch[0].to(device, non_blocking=True)
                else:
                    penalty = penalty_batch["image"].to(device, non_blocking=True)
            else:
                penalty = torch.zeros_like(images)

            optimizer_st.zero_grad()
            optimizer_ae.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                alpha     = float(params.get("ae_loss_weight", 0.5))
                loss_dict = _efficientad_training_step(model, images, penalty, alpha=alpha)
                total_loss = loss_dict["loss_total"]

            last_loss = total_loss.item()

            if not np.isfinite(last_loss):
                self._write_log(f"[경고] Step {step}: loss={last_loss} — 학습 발산. 학습률을 낮춰 주세요.")
                raise ValueError(f"Loss가 {last_loss}이 되어 학습을 중단합니다. learning_rate를 낮춰 주세요.")

            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer_st)
                scaler.unscale_(optimizer_ae)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
                scaler.step(optimizer_st)
                scaler.step(optimizer_ae)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
                optimizer_st.step()
                optimizer_ae.step()

            scheduler_st.step()
            scheduler_ae.step()

            step += 1

            if step == self.start_step + 1 or step % report_every == 0 or step == total_steps:
                elapsed = time.time() - self._start_time
                self.loss_history.append({"step": step, "loss": round(last_loss, 6)})
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

    # ── PatchCore 학습 루프 (단일 경로) ───────────────────────────────────────

    def _train_patchcore(
        self,
        model: object,
        train_loader: DataLoader,
        device: torch.device,
    ) -> tuple[bool, int]:
        """
        PatchCore 특징 추출 → coreset 구성 → memory_bank 설정.

        단일 경로(_extract_patchcore_features)로 통합:
          - checkpoint/resume 일관성 보장
          - anomalib 버전에 무관하게 동일한 로직 사용
          - 마지막에 coreset 서브샘플링 후 memory_bank 직접 설정

        resume 지원:
          - start_batch_idx: 이 배치 이전은 건너뜀
          - accumulated_features: 이미 추출된 특징 텐서
        """
        from utils.model_factory import _extract_patchcore_features
        from utils.checkpoint_manager import save_checkpoint

        params    = self.model_config.get("params", {})
        max_train = params.get("max_train", 1000)

        model = model.to(device)
        model.eval()

        batch_size    = train_loader.batch_size or 1
        total_batches = min(len(train_loader), max_train // max(batch_size, 1) + 1)

        # resume: 이미 추출된 특징 복원
        all_features: list[torch.Tensor] = []
        if (
            self.accumulated_features is not None
            and self.accumulated_features.numel() > 0
        ):
            all_features.append(self.accumulated_features.cpu())
            self._write_log(
                f"[재개] batch {self.start_batch_idx}까지 추출된 특징 복원 "
                f"({self.accumulated_features.shape[0]:,}개 패치)"
            )

        self._emit_stage(2, "특징 추출")
        self._write_log(
            f"[학습] PatchCore 특징 추출 시작 — "
            f"총 {total_batches}배치 / 재개: {self.start_batch_idx}배치부터"
        )
        self.result_queue.put({
            "type":    "progress",
            "step":    self.start_batch_idx,
            "total":   total_batches,
            "loss":    0.0,
            "elapsed": round(time.time() - self._start_time, 1),
        })

        _ckpt_saved = False

        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                # ① 중지 체크
                if self.stop_event.is_set():
                    return False, batch_idx

                if batch_idx >= total_batches:
                    break

                # resume: 이미 처리된 배치 건너뜀
                if batch_idx < self.start_batch_idx:
                    continue

                # ② 일시정지 체크
                if self.pause_event.is_set():
                    if not _ckpt_saved:
                        feat_tensor = (
                            torch.cat(all_features, dim=0)
                            if all_features
                            else torch.empty(0)
                        )
                        ckpt_path = save_checkpoint(
                            data={
                                "model_type":           "patchcore",
                                "experiment_id":        self.experiment_id,
                                "batch_idx":            batch_idx,
                                "total_batches":        total_batches,
                                "model_config":         self.model_config,
                                "preprocessing_config": self.preprocessing_config,
                                "dataset_path":         self.dataset_path,
                                "accumulated_features": feat_tensor,
                            },
                            exp_id=self.experiment_id,
                            label=batch_idx,
                        )
                        self._write_log(f"[체크포인트] 저장 완료: {ckpt_path.name}")
                        self.result_queue.put({
                            "type":     "paused",
                            "step":     batch_idx,
                            "ckpt_path": str(ckpt_path),
                        })
                        _ckpt_saved = True

                    while self.pause_event.is_set():
                        time.sleep(0.1)
                        if self.stop_event.is_set():
                            return False, batch_idx
                    _ckpt_saved = False
                    self._write_log(f"[재개] batch {batch_idx}부터 특징 추출 재시작")

                # ── 특징 추출
                images   = batch["image"].to(device)
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

        # ── coreset 구성 → memory_bank 직접 설정
        self._emit_stage(3, "Coreset 구성")
        self._write_log("[PatchCore] 특징 추출 완료. Coreset 구성 중...")
        feature_stack = torch.cat(all_features, dim=0)
        coreset_ratio = params.get("coreset_sampling_ratio", 0.1)
        coreset_size  = max(1, int(len(feature_stack) * coreset_ratio))
        _g = torch.Generator()
        _g.manual_seed(self.model_config.get("random_seed", 42))
        indices       = torch.randperm(len(feature_stack), generator=_g)[:coreset_size]
        coreset       = feature_stack[indices].to(device)

        # model.model.memory_bank 또는 model.memory_bank 중 존재하는 쪽에 설정
        self._emit_stage(4, "Memory Bank 설정")
        torch_model = getattr(model, "model", None)
        if torch_model is not None and hasattr(torch_model, "memory_bank"):
            # anomalib 내부 forward()가 인식하는 방식
            torch_model.register_buffer("memory_bank", coreset)
        else:
            # torch_model이 없거나 memory_bank 속성이 없는 경우
            # state_dict() 포함을 위해 register_buffer 사용
            target = torch_model if torch_model is not None else model
            target.register_buffer("memory_bank", coreset)

        elapsed_total = time.time() - self._start_time
        self._write_log(
            f"[완료] 메모리 뱅크 구성 완료 "
            f"({coreset_size:,}/{len(feature_stack):,} 패치) | 경과: {elapsed_total:.1f}s"
        )
        self.result_queue.put({
            "type":    "progress",
            "step":    total_batches,
            "total":   total_batches,
            "loss":    0.0,
            "elapsed": round(elapsed_total, 1),
        })

        self._last_step = total_batches
        return True, total_batches

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

                image      = batch["image"].to(device)
                label      = int(batch["label"].item())
                image_path = batch["image_path"][0]

                amap  = _get_anomaly_map(model, image)
                score = float(amap.max())              # ← 이미 계산된 amap 재사용

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
            gamma     = params.get("lr_decay_factor", 0.1)
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
