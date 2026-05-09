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
    IMAGENET_PENALTY_DIR 존재 시 실제 이미지 사용, 없으면 FakeData.
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
    """PRD 08 B.4.1 매핑 테이블 기준 EfficientAd 생성."""
    if EfficientAd is None:
        raise ImportError(
            "anomalib이 설치되지 않았습니다. EfficientAD 모델을 생성할 수 없습니다."
        )
    params = model_config["params"]
    model_size_key = params.get("model_size", "medium")
    model_size_enum = SIZE_MAP.get(model_size_key, SIZE_MAP.get("medium"))

    return EfficientAd(
        teacher_out_channels=params.get("out_channels", 384),
        model_size=model_size_enum,
        lr=params.get("learning_rate", 1e-4),
        weight_decay=params.get("weight_decay", 1e-4),
        padding=params.get("padding", False),
        map_combination_alpha=params.get("ae_loss_weight", 0.5),
        autoencoder_lr=params.get("autoencoder_lr", params.get("learning_rate", 1e-4)),
        autoencoder_weight_decay=params.get("autoencoder_weight_decay", 1e-5),
        penalized_normalized=(params.get("imagenet_penalty_weight", 1.0) > 0),
    )


# ── PatchCore model creation ───────────────────────────────────────────────────

def _create_patchcore_model(model_config: dict) -> "Patchcore":
    """PRD 08 B.5.1 매핑 테이블 기준 Patchcore 생성."""
    if Patchcore is None:
        raise ImportError(
            "anomalib이 설치되지 않았습니다. PatchCore 모델을 생성할 수 없습니다."
        )
    params = model_config["params"]
    pre_trained = (params.get("pretrained_source", "torchvision") == "torchvision")
    num_neighbors = params.get("knn", params.get("neighbourhood_kernel_size", 3))

    model = Patchcore(
        backbone=params.get("backbone", "wide_resnet50_2"),
        layers=["layer2", "layer3"],
        pre_trained=pre_trained,
        coreset_sampling_ratio=params.get("coreset_sampling_ratio", 0.1),
        num_neighbors=num_neighbors,
    )

    if not pre_trained and params.get("pretrained_path"):
        state_dict = torch.load(params["pretrained_path"], map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.backbone.load_state_dict(state_dict, strict=False)

    return model


# ── Anomaly map helpers ────────────────────────────────────────────────────────

def _extract_patchcore_features(
    model: "Patchcore",
    images: torch.Tensor,
) -> torch.Tensor:
    """layer2, layer3 특징 추출 후 (B*H'*W', C2+C3) 반환."""
    features: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(name: str):
        def hook(module, input, output):
            features[name] = output
        return hook

    hooks.append(model.backbone.layer2.register_forward_hook(_make_hook("layer2")))
    hooks.append(model.backbone.layer3.register_forward_hook(_make_hook("layer3")))

    with torch.no_grad():
        _ = model.backbone(images)

    for h in hooks:
        h.remove()

    f2 = features["layer2"]
    f3 = features["layer3"]
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
) -> dict:
    """
    EfficientAd.training_step() 시도 후 실패 시 수동 fallback.
    반환: {"loss_total": Tensor} 포함 dict.
    """
    if hasattr(model, "training_step"):
        batch = {"image": images, "penalty_images": penalty_images}
        try:
            loss = model.training_step(batch, batch_idx=0)
            if isinstance(loss, dict):
                return loss
            return {"loss_total": loss}
        except Exception:
            pass

    # Fallback: manual student-teacher loss
    with torch.no_grad():
        teacher_out = model.teacher(images)
    student_out = model.student(images)
    ae_out = model.autoencoder(images)

    loss_st = torch.mean((teacher_out - student_out) ** 2)
    loss_ae = torch.mean((images - ae_out) ** 2)
    alpha = getattr(model, "map_combination_alpha", 0.5)
    loss_total = alpha * loss_ae + (1.0 - alpha) * loss_st
    return {"loss_st": loss_st, "loss_ae": loss_ae, "loss_total": loss_total}


def _get_anomaly_map(
    model: object,
    image: torch.Tensor,
) -> np.ndarray:
    """
    단일 이미지 추론 → Anomaly Map (H, W) float32 반환.
    EfficientAd / Patchcore 공통 경로 시도 후 Patchcore fallback.
    """
    if hasattr(model, "anomaly_map_generator") or hasattr(model, "forward"):
        try:
            output = model(image)
            if isinstance(output, dict) and "anomaly_map" in output:
                amap = output["anomaly_map"]
            elif hasattr(output, "anomaly_map"):
                amap = output.anomaly_map
            else:
                amap = output
            if isinstance(amap, torch.Tensor):
                return amap.squeeze().cpu().numpy().astype(np.float32)
        except Exception:
            pass

    # Patchcore KNN fallback
    if hasattr(model, "memory_bank"):
        features = _extract_patchcore_features(model, image)
        dists = torch.cdist(
            features.unsqueeze(0),
            model.memory_bank.unsqueeze(0),
            p=2,
        ).squeeze(0)
        k = min(
            getattr(model, "num_neighbors", 9),
            dists.shape[1],
        )
        patch_scores, _ = torch.topk(dists, k, dim=1, largest=False)
        patch_scores = patch_scores.mean(dim=1)
        spatial_size = int(patch_scores.shape[0] ** 0.5)
        patch_map = patch_scores.reshape(spatial_size, spatial_size).cpu().numpy()
        H = W = image.shape[-1]
        return cv2.resize(patch_map, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)

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
    image_tensor: torch.Tensor,
) -> np.ndarray:
    """단일 이미지 추론. Anomaly Map (H, W) float32 반환."""
    device = next(iter(model.parameters())).device
    tensor = image_tensor.to(device)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        return _get_anomaly_map(model, tensor)
