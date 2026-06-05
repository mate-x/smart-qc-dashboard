from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# Anomalib import — ImportError 시 None으로 처리
try:
    from anomalib.models.image.efficient_ad.lightning_model import (
        EfficientAd,
        EfficientAdModelSize,
    )
    SIZE_MAP = {
        "small":  EfficientAdModelSize.S,
        "medium": EfficientAdModelSize.M,
    }
except ImportError:
    EfficientAd = None           # type: ignore[assignment,misc]
    EfficientAdModelSize = None  # type: ignore[assignment,misc]
    SIZE_MAP = {}

try:
    from anomalib.models.image.patchcore.lightning_model import Patchcore
except ImportError:
    Patchcore = None  # type: ignore[assignment,misc]


# ── ImageNet Penalty DataLoader ────────────────────────────────────────────────

def build_imagenet_penalty_loader(
    batch_size: int,
    image_size: int,
    device: str,
) -> "torch.utils.data.DataLoader":
    """
    ImageNet penalty 배치용 DataLoader.
    IMAGENET_PENALTY_DIR 존재 시 실제 이미지 사용.
    Z.1 정오표: FakeData fallback 없음 — validate_imagenet_penalty_dir()가 선행 보장.
    """
    from utils.storage import IMAGENET_PENALTY_DIR
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(str(IMAGENET_PENALTY_DIR), transform=transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )


# ── EfficientAD model creation ─────────────────────────────────────────────────

def _create_efficientad_model(model_config: dict) -> "EfficientAd":
    """PRD 08 B.4.1 매핑 테이블 기준 EfficientAd 생성.

    anomalib 버전별로 EfficientAd.__init__ 시그니처가 다르므로
    inspect로 실제 수용 파라미터만 골라서 전달한다.
    """
    import inspect

    if EfficientAd is None:
        raise ImportError(
            "anomalib이 설치되지 않았습니다. EfficientAD 모델을 생성할 수 없습니다."
        )
    params = model_config["params"]
    model_size_key = params.get("model_size", "medium")
    model_size_enum = SIZE_MAP.get(model_size_key, SIZE_MAP.get("medium"))

    # anomalib 2.4.1 생성자에 실제로 존재하는 파라미터만 전달
    # autoencoder_lr/autoencoder_weight_decay/ae_loss_weight/use_imagenet_penalty 은 학습 루프에서 처리
    candidates = {
        "teacher_out_channels": params.get("out_channels", 384),
        "model_size":           model_size_enum,
        "lr":                   params.get("learning_rate", 1e-4),
        "weight_decay":         params.get("weight_decay", 1e-4),
        "padding":              params.get("padding", False),
    }

    valid_params = set(inspect.signature(EfficientAd.__init__).parameters) - {"self"}
    kwargs = {k: v for k, v in candidates.items() if k in valid_params}

    return EfficientAd(**kwargs)


# ── PatchCore model creation ───────────────────────────────────────────────────

def _create_patchcore_model(model_config: dict) -> "Patchcore":
    """PRD 08 B.5.1 매핑 테이블 기준 Patchcore 생성."""
    if Patchcore is None:
        raise ImportError(
            "anomalib이 설치되지 않았습니다. PatchCore 모델을 생성할 수 없습니다."
        )
    params = model_config["params"]
    pre_trained = (params.get("pretrained_source", "torchvision") == "torchvision")
    num_neighbors = params.get("knn", 9)

    model = Patchcore(
        backbone=params.get("backbone", "wide_resnet50_2"),
        layers=["layer2", "layer3"],
        pre_trained=pre_trained,
        coreset_sampling_ratio=params.get("coreset_sampling_ratio", 0.1),
        num_neighbors=num_neighbors,
    )

    # neighbourhood_kernel_size → feature_pooler kernel 크기 override
    # anomalib 2.4.1: PatchcoreModel.feature_pooler = AvgPool2d(3,1,1) 하드코딩
    nks = params.get("neighbourhood_kernel_size", 3)
    torch_model = getattr(model, "model", None)
    if torch_model is not None and hasattr(torch_model, "feature_pooler"):
        import torch.nn as _nn
        padding = nks // 2
        torch_model.feature_pooler = _nn.AvgPool2d(nks, 1, padding)

    if not pre_trained and params.get("pretrained_path"):
        state_dict = torch.load(params["pretrained_path"], map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        # anomalib 2.4.x: 실제 backbone은 model.model.feature_extractor.feature_extractor
        torch_model = getattr(model, "model", None)
        fe_outer = getattr(torch_model, "feature_extractor", None) if torch_model else None
        fe_inner = getattr(fe_outer, "feature_extractor", None)
        target = fe_inner or fe_outer or getattr(model, "backbone", None)
        if target is None:
            raise AttributeError("PatchCore backbone을 찾을 수 없어 pretrained 가중치를 로드할 수 없습니다.")
        target.load_state_dict(state_dict, strict=False)

    return model


# ── Anomaly map helpers ────────────────────────────────────────────────────────

def _extract_patchcore_features(
    model: "Patchcore",
    images: torch.Tensor,
) -> torch.Tensor:
    """layer2, layer3 특징 추출 후 (B*H'*W', C2+C3) 반환.

    anomalib 2.4.x: model.model.feature_extractor(images) → {"layer2": ..., "layer3": ...} dict 직접 반환.
    구버전 anomalib: hook 방식 fallback.
    """
    torch_model = getattr(model, "model", None)
    fe = getattr(torch_model, "feature_extractor", None) if torch_model is not None else None

    if fe is not None:
        # anomalib 2.4.x — TimmFeatureExtractor.forward() 가 dict 반환
        with torch.no_grad():
            feat_dict = fe(images)
        if isinstance(feat_dict, dict) and len(feat_dict) >= 1:
            layer_names = sorted(feat_dict.keys())
            f2 = feat_dict[layer_names[0]]
            f3 = feat_dict[layer_names[-1]]
            f3_up = nn.functional.interpolate(
                f3, size=f2.shape[-2:], mode="bilinear", align_corners=False
            )
            combined = torch.cat([f2, f3_up], dim=1)
            B, C, H, W = combined.shape
            return combined.permute(0, 2, 3, 1).reshape(B * H * W, C)

    # Fallback: hook 방식 (구버전 anomalib)
    backbone = (
        getattr(model, "backbone", None)
        or getattr(getattr(model, "feature_extractor", None), "feature_extractor", None)
    )
    if backbone is None or not hasattr(backbone, "layer2"):
        raise AttributeError(
            f"PatchCore backbone에서 layer2/layer3를 찾을 수 없습니다. "
            f"model 속성: {[a for a in dir(model) if not a.startswith('_')]}"
        )

    feat_hooks: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(name: str):
        def hook(module, input, output):
            feat_hooks[name] = output
        return hook

    hooks.append(backbone.layer2.register_forward_hook(_make_hook("layer2")))
    hooks.append(backbone.layer3.register_forward_hook(_make_hook("layer3")))

    with torch.no_grad():
        _ = backbone(images)

    for h in hooks:
        h.remove()

    f2 = feat_hooks["layer2"]
    f3 = feat_hooks["layer3"]
    f3_up = nn.functional.interpolate(
        f3, size=f2.shape[-2:], mode="bilinear", align_corners=False
    )
    combined = torch.cat([f2, f3_up], dim=1)
    B, C, H, W = combined.shape
    return combined.permute(0, 2, 3, 1).reshape(B * H * W, C)


def _efficientad_training_step(
    model: "EfficientAd",
    images: torch.Tensor,
    penalty_images: torch.Tensor,
    alpha: float = 0.5,
) -> dict:
    """
    3단계 순서로 loss 계산 시도. 반환: {"loss_total": Tensor} 포함 dict.

    Path 1: model.training_step() — 2.4.x는 Batch 타입을 요구하므로 dict 전달 시 실패,
             except로 잡고 다음 경로로 진행.
    Path 2: EfficientAdModel.forward(batch, batch_imagenet) — anomalib 2.4.x 전용.
    Path 3: 컴포넌트별 수동 forward — 구버전 anomalib fallback.
    """
    _inner = getattr(model, "model", model)

    # Path 1: training_step (E-2: 반환 키 "loss" → "loss_total" 정규화)
    if hasattr(model, "training_step"):
        try:
            loss = model.training_step({"image": images, "penalty_images": penalty_images}, batch_idx=0)
            if isinstance(loss, dict):
                if "loss_total" not in loss and "loss" in loss:
                    loss["loss_total"] = loss["loss"]
                return loss
            return {"loss_total": loss}
        except Exception:
            pass

    # Path 2: EfficientAdModel.forward(batch, batch_imagenet) — anomalib 2.4.x (E-3)
    if hasattr(_inner, "compute_losses"):
        try:
            result = _inner(batch=images, batch_imagenet=penalty_images)
            if isinstance(result, tuple) and len(result) == 3:
                loss_st, loss_ae, loss_stae = result
                total = alpha * loss_ae + (1.0 - alpha) * loss_st + loss_stae
                return {"loss_st": loss_st, "loss_ae": loss_ae, "loss_total": total}
        except Exception:
            pass

    # Path 3: 컴포넌트별 수동 forward — 구버전 anomalib (E-1: ae alias 추가)
    teacher = getattr(model, "teacher", getattr(_inner, "teacher", None))
    student = getattr(model, "student", getattr(_inner, "student", None))
    autoencoder = (
        getattr(model, "autoencoder", None) or getattr(model, "ae", None)
        or getattr(_inner, "autoencoder", None) or getattr(_inner, "ae", None)
    )

    if teacher is None or student is None or autoencoder is None:
        raise AttributeError(
            f"EfficientAd: teacher/student/autoencoder를 찾을 수 없습니다. "
            f"model 속성: {[a for a in dir(model) if not a.startswith('_')]}"
        )

    # AutoEncoder.forward()가 버전에 따라 (batch,) 또는 (batch, image_size) 서명을 가짐.
    # anomalib 2.x: forward(batch, image_size) — image_size 필수.
    _h, _w = images.shape[-2], images.shape[-1]

    with torch.no_grad():
        teacher_out = teacher(images)
    student_out = student(images)

    try:
        ae_out = autoencoder(images, (_h, _w))      # anomalib 2.x signature
    except TypeError:
        ae_out = autoencoder(images)                 # anomalib 1.x signature

    # ── loss_stae: AE 재구성 이미지를 teacher/student에 다시 통과시켜 비교 ──────
    # EfficientAD 원본 알고리즘의 핵심 3번째 손실 항목.
    # AE가 재구성한 이미지(이상 없는 버전)에 대해 student가 teacher를 얼마나
    # 모방하는지 학습 → AE 경로를 통한 이상 감지 능력 훈련.
    # ae_out은 detach()하여 loss_stae의 gradient가 AE가 아닌 student만 업데이트.
    ae_out_detached = ae_out.detach()
    with torch.no_grad():
        teacher_ae_out = teacher(ae_out_detached)
    student_ae_out = student(ae_out_detached)
    loss_stae = torch.mean((teacher_ae_out - student_ae_out) ** 2)

    loss_st    = torch.mean((teacher_out - student_out) ** 2)
    loss_ae    = torch.mean((images - ae_out) ** 2)
    loss_total = alpha * loss_ae + (1.0 - alpha) * loss_st + loss_stae
    return {
        "loss_st":   loss_st,
        "loss_ae":   loss_ae,
        "loss_stae": loss_stae,
        "loss_total": loss_total,
    }


def _mask_padding_from_anomaly_map(
    amap: np.ndarray,
    image: torch.Tensor,
    padding_threshold: float = -1.7,
) -> np.ndarray:
    """
    resize_with_padding()로 추가된 zero-padding 영역을 anomaly map에서 제거한다.

    zero-padding은 ImageNet 정규화 후 매우 낮은 값(-2.1, -2.0, -1.8)을 가진다.
    모든 채널이 threshold 미만인 픽셀 = padding → anomaly score를 0으로 설정.

    padding 영역이 anomaly map에서 높게 나오면 heatmap 코너가 붉게 표시되는
    시각적 버그를 유발한다.
    """
    try:
        if image.dim() == 4 and image.shape[1] == 3:
            # (1, 3, H, W) → 모든 채널이 threshold 미만인 픽셀 = padding
            is_padding = (image[0] < padding_threshold).all(dim=0).cpu().numpy()  # (H, W)
            # 이미지 전체가 padding이거나 padding이 없으면 마스킹 불필요
            if is_padding.any() and not is_padding.all():
                amap = amap.copy()
                amap[is_padding] = 0.0
    except Exception:
        pass  # 마스킹 실패 시 원본 반환
    return amap


def _get_anomaly_map(
    model: object,
    image: torch.Tensor,
) -> np.ndarray:
    """
    단일 이미지 추론 → Anomaly Map (H, W) float32 반환.

    anomalib 2.4.x: model.model(image) → InferenceBatch(anomaly_map=...) 반환.
    model(image) 보다 model.model(image)를 먼저 시도해 LightningModule forward 미정의 문제를 회피.
    padding 영역은 자동으로 0으로 마스킹하여 heatmap 코너 아티팩트를 제거한다.
    """
    torch_model = getattr(model, "model", None)

    # model.model → model 순서로 시도 (2.4.x는 torch_model이 더 안정적)
    for m in ([torch_model, model] if torch_model is not None else [model]):
        if m is None:
            continue
        try:
            output = m(image)
            if isinstance(output, dict) and "anomaly_map" in output:
                amap = output["anomaly_map"]
            elif hasattr(output, "anomaly_map"):
                amap = output.anomaly_map
            else:
                amap = output
            if isinstance(amap, torch.Tensor):
                raw = amap.squeeze().cpu().numpy().astype(np.float32)
                return _mask_padding_from_anomaly_map(raw, image)
        except Exception:
            continue

    # Patchcore KNN fallback — model 또는 model.model에서 memory_bank 탐색
    mem_holder = (
        model if hasattr(model, "memory_bank")
        else (torch_model if (torch_model is not None and hasattr(torch_model, "memory_bank")) else None)
    )
    if mem_holder is not None:
        features = _extract_patchcore_features(model, image)
        mem = mem_holder.memory_bank
        if mem.device != features.device:
            mem = mem.to(features.device)
        dists = torch.cdist(features.unsqueeze(0), mem.unsqueeze(0), p=2).squeeze(0)
        k = min(getattr(mem_holder, "num_neighbors", 9), dists.shape[1])
        patch_scores, _ = torch.topk(dists, k, dim=1, largest=False)
        patch_scores = patch_scores.mean(dim=1)
        N = patch_scores.shape[0]
        spatial_size = int(N ** 0.5)
        if spatial_size * spatial_size != N:
            raise ValueError(
                f"PatchCore feature map 크기({N}개 패치)가 정방형이 아닙니다. "
                f"image_size를 8의 배수(예: 256)로 설정해 주세요."
            )
        patch_map = patch_scores.reshape(spatial_size, spatial_size).cpu().numpy()
        H = W = image.shape[-1]
        raw = cv2.resize(patch_map, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return _mask_padding_from_anomaly_map(raw, image)

    raise NotImplementedError(f"알 수 없는 모델 구조: {type(model)}")


# ── Public factory functions ───────────────────────────────────────────────────

def create_trainer(
    model_config: dict,
    preprocessing_config: dict,
    dataset_path: str,
    device: str,
    experiment_id: str,
    stop_event: threading.Event,
    result_queue: queue.Queue,
    pause_event: "threading.Event | None" = None,
    # EfficientAD resume
    start_step: int = 0,
    student_state_dict: "dict | None" = None,
    autoencoder_state_dict: "dict | None" = None,
    optimizer_st_state_dict: "dict | None" = None,
    optimizer_ae_state_dict: "dict | None" = None,
    scheduler_st_state_dict: "dict | None" = None,
    scheduler_ae_state_dict: "dict | None" = None,
    loss_history: "list | None" = None,
    # PatchCore resume
    start_batch_idx: int = 0,
    accumulated_features: "torch.Tensor | None" = None,
) -> "TrainingWorker":
    """TrainingWorker 생성 후 반환. 실행은 호출자가 worker.start()로 수행."""
    from utils.training_worker import TrainingWorker

    return TrainingWorker(
        experiment_id=experiment_id,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        dataset_path=dataset_path,
        device=device,
        stop_event=stop_event,
        result_queue=result_queue,
        pause_event=pause_event,
        start_step=start_step,
        student_state_dict=student_state_dict,
        autoencoder_state_dict=autoencoder_state_dict,
        optimizer_st_state_dict=optimizer_st_state_dict,
        optimizer_ae_state_dict=optimizer_ae_state_dict,
        scheduler_st_state_dict=scheduler_st_state_dict,
        scheduler_ae_state_dict=scheduler_ae_state_dict,
        loss_history=loss_history,
        start_batch_idx=start_batch_idx,
        accumulated_features=accumulated_features,
    )


def load_model_for_inference(
    exp_id: str,
    model_path: str,
    model_config: dict,
    device: str,
) -> object:
    """
    저장된 model_state_dict.pth 로드 후 추론 가능한 모델 반환 (eval 모드).

    Raises:
        RuntimeError: pth 파일 없거나 state_dict 불일치 시
    """
    pth_path = Path(model_path) / "model_state_dict.pth"
    if not pth_path.exists():
        raise RuntimeError(
            f"ERR_MODEL_FILE_NOT_FOUND: {pth_path} — "
            "모델 파일이 존재하지 않습니다."
        )

    model_type = model_config["model_type"]
    if model_type == "efficientad":
        model = _create_efficientad_model(model_config)
    elif model_type == "patchcore":
        model = _create_patchcore_model(model_config)
    else:
        raise RuntimeError(f"알 수 없는 model_type: {model_type}")

    try:
        state_dict = torch.load(str(pth_path), map_location=device)
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        raise RuntimeError(
            f"ERR_MODEL_LOAD_FAILED: state_dict 불일치. {e}"
        ) from e

    model.to(device)
    model.eval()
    return model


def run_inference(
    model: object,
    image_tensor: torch.Tensor,  # (1, C, H, W) — 이미 전처리된 텐서
) -> np.ndarray:
    """단일 이미지 추론. Anomaly Map (H, W) float32 반환."""
    device = next(model.parameters()).device
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        return _get_anomaly_map(model, image_tensor)
