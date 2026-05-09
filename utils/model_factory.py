from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from anomalib.models.image.patchcore.lightning_model import Patchcore
except ImportError:
    Patchcore = None  # type: ignore[assignment,misc]

try:
    from anomalib.models.image.efficient_ad.lightning_model import (
        EfficientAd,
        EfficientAdModelSize,
    )
except ImportError:
    EfficientAd = None         # type: ignore[assignment,misc]
    EfficientAdModelSize = None  # type: ignore[assignment,misc]


# ──────────────────────────────────────────────────────────────
# 공개 팩토리 API
# ──────────────────────────────────────────────────────────────

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
    exp_id: str,             # Z.7: 로그 컨텍스트용 추가 파라미터
    model_path: str,
    model_config: dict,
    device: str,
) -> object:
    """
    저장된 state_dict 로드 후 추론용 모델 반환.
    model_config.model_type에 따라 Patchcore 또는 EfficientAd 초기화.
    """
    pth_path = Path(model_path) / "model_state_dict.pth"
    if not pth_path.exists():
        raise FileNotFoundError(
            f"[{exp_id}] 모델 파일이 없습니다: {pth_path}"
        )

    state_dict = torch.load(str(pth_path), map_location=device)
    model_type = model_config.get("model_type", "")

    if model_type == "patchcore":
        model = _create_patchcore_model(model_config)
    elif model_type == "efficientad":
        model = _create_efficientad_model(model_config)
    else:
        raise ValueError(f"[{exp_id}] 지원하지 않는 모델 타입: {model_type}")

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def run_inference(
    model: object,
    image_path: str,
    preprocessing_config: dict,
) -> np.ndarray:
    """
    단일 이미지 추론. Anomaly Map (H, W) float32 반환.
    학습과 동일한 preprocessing_config 사용 (DA-05).
    """
    from utils.image_utils import apply_preprocessing

    _, image_tensor = apply_preprocessing(image_path, preprocessing_config)
    image_tensor = image_tensor.unsqueeze(0)   # (1, C, H, W)
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        anomaly_map = _get_anomaly_map(model, image_tensor)

    return anomaly_map


# ──────────────────────────────────────────────────────────────
# PatchCore 모델 생성 (PRD 08 B.5.1)
# ──────────────────────────────────────────────────────────────

def _create_patchcore_model(model_config: dict) -> "Patchcore":
    if Patchcore is None:
        raise ImportError("anomalib 패키지가 설치되지 않았습니다.")

    params = model_config.get("params", {})
    pre_trained = (params.get("pretrained_source", "torchvision") == "torchvision")

    # knn 파라미터: knn 우선, 없으면 neighbourhood_kernel_size
    num_neighbors = params.get("knn") or params.get("neighbourhood_kernel_size", 9)

    model = Patchcore(
        backbone=params.get("backbone", "wide_resnet50_2"),
        layers=["layer2", "layer3"],          # E.2: layer2+layer3 고정
        pre_trained=pre_trained,
        coreset_sampling_ratio=params.get("coreset_sampling_ratio", 0.1),
        num_neighbors=int(num_neighbors),
    )

    # 로컬 가중치 로드 (pretrained_source == "local")
    if not pre_trained and params.get("pretrained_path"):
        state_dict = torch.load(params["pretrained_path"], map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.backbone.load_state_dict(state_dict, strict=False)

    return model


# ──────────────────────────────────────────────────────────────
# EfficientAD 모델 생성 (PRD 08 B.4.1) — stub
# ──────────────────────────────────────────────────────────────

def _create_efficientad_model(model_config: dict) -> "EfficientAd":
    if EfficientAd is None or EfficientAdModelSize is None:
        raise ImportError("anomalib 패키지가 설치되지 않았습니다.")

    params  = model_config.get("params", {})
    SIZE_MAP = {
        "small":  EfficientAdModelSize.S,
        "medium": EfficientAdModelSize.M,
    }
    return EfficientAd(
        teacher_out_channels=params.get("out_channels", 384),
        model_size=SIZE_MAP.get(params.get("model_size", "small"), EfficientAdModelSize.S),
        lr=params.get("learning_rate", 1e-4),
        weight_decay=params.get("weight_decay", 1e-5),
        padding=params.get("padding", False),
        map_combination_alpha=params.get("ae_loss_weight", 0.5),
    )


# ──────────────────────────────────────────────────────────────
# 공유 추론 헬퍼 (training_worker.py에서도 import)
# ──────────────────────────────────────────────────────────────

def _extract_patchcore_features(
    model: "Patchcore",
    images: torch.Tensor,   # (B, C, H, W)
) -> torch.Tensor:
    """
    layer2, layer3 forward hook으로 특징 추출 후
    동일 공간 크기로 bilinear upsample + concat.
    반환 shape: (B * H' * W', C2 + C3)
    """
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(name: str):
        def _hook(module, inp, out):
            captured[name] = out
        return _hook

    hooks.append(model.backbone.layer2.register_forward_hook(_make_hook("layer2")))
    hooks.append(model.backbone.layer3.register_forward_hook(_make_hook("layer3")))

    with torch.no_grad():
        _ = model.backbone(images)

    for h in hooks:
        h.remove()

    f2 = captured["layer2"]   # (B, C2, H2, W2)
    f3 = captured["layer3"]   # (B, C3, H3, W3)

    # f3를 f2의 공간 크기로 upsample
    f3_up = nn.functional.interpolate(
        f3, size=f2.shape[-2:], mode="bilinear", align_corners=False
    )
    combined = torch.cat([f2, f3_up], dim=1)   # (B, C2+C3, H2, W2)

    B, C, H, W = combined.shape
    patches = combined.permute(0, 2, 3, 1).reshape(B * H * W, C)
    return patches   # (B*H'*W', C_combined)


def _get_anomaly_map(model, image: torch.Tensor) -> np.ndarray:
    """
    단일 이미지 (1, C, H, W) → Anomaly Map (H, W) float32.

    1순위: model.anomaly_map_generator 존재 시 Anomalib 내장 경로 사용
    2순위: model.memory_bank 존재 시 수동 kNN 거리 계산
    """
    # Anomalib 내장 경로
    if hasattr(model, "anomaly_map_generator"):
        output = model(image)
        if isinstance(output, dict) and "anomaly_map" in output:
            amap = output["anomaly_map"]
        elif hasattr(output, "anomaly_map"):
            amap = output.anomaly_map
        else:
            amap = output
        return amap.squeeze().cpu().numpy().astype(np.float32)

    # PatchCore 수동 kNN fallback
    if hasattr(model, "memory_bank") and model.memory_bank is not None:
        features = _extract_patchcore_features(model, image)   # (H'*W', C)
        mem = model.memory_bank
        if mem.device != features.device:
            mem = mem.to(features.device)

        # 유클리드 거리 행렬
        dists = torch.cdist(
            features.unsqueeze(0),
            mem.unsqueeze(0),
            p=2,
        ).squeeze(0)   # (H'*W', M)

        k = min(getattr(model, "num_neighbors", 9), dists.shape[1])
        patch_scores, _ = torch.topk(dists, k, dim=1, largest=False)
        patch_scores = patch_scores.mean(dim=1)   # (H'*W',)

        spatial = int(round(patch_scores.shape[0] ** 0.5))
        patch_map = patch_scores.reshape(spatial, spatial).cpu().numpy()

        H = W = image.shape[-1]
        amap = cv2.resize(patch_map, (W, H), interpolation=cv2.INTER_LINEAR)
        return amap.astype(np.float32)

    raise NotImplementedError(f"알 수 없는 모델 구조: {type(model)}")
