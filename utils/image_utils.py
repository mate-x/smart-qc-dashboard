from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}


# ──────────────────────────────────────────────────────────────
# 저수준 독립 함수 (numpy 입출력) — 테스트에서 직접 import 가능
# ──────────────────────────────────────────────────────────────

def ensure_rgb(image: Image.Image) -> np.ndarray:
    """PIL Image를 RGB numpy 배열 (H, W, 3)으로 변환."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def apply_homomorphic(
    img: np.ndarray,
    sigma: float = 10.0,
    gamma_H: float = 1.5,
    gamma_L: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """
    Homomorphic Filter — GaussianBlur 기반 공간 도메인 근사.
    sigma: 저주파/고주파 분리 기준 (클수록 넓은 저주파 제거)
    gamma_H: 고주파(reflectance) 가중치
    gamma_L: 저주파(illumination) 가중치
    normalize: True 시 출력을 [0, 255]로 정규화
    """
    def _homo_channel(ch: np.ndarray) -> np.ndarray:
        ch_float = np.log(ch.astype(np.float32) + 1e-6)
        blur = cv2.GaussianBlur(ch_float, (0, 0), sigma)
        high = ch_float - blur
        out = gamma_H * high + gamma_L * blur
        out = np.exp(out)
        if normalize:
            out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
        return np.clip(out, 0, 255).astype(np.uint8)

    channels = [_homo_channel(img[:, :, c]) for c in range(3)]
    return np.stack(channels, axis=2)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    channels = [clahe.apply(img[:, :, c]) for c in range(3)]
    return np.stack(channels, axis=2)


def apply_he(img: np.ndarray) -> np.ndarray:
    """Histogram Equalization 적용 (채널별 독립 처리)."""
    channels = [cv2.equalizeHist(img[:, :, c]) for c in range(3)]
    return np.stack(channels, axis=2)


def resize_with_padding(
    image: Image.Image | np.ndarray,
    target_size: int,
) -> np.ndarray:
    """
    비율 유지 Resize 후 검정(0) 패딩으로 target_size×target_size 생성.
    PIL Image 또는 numpy 배열 입력 모두 허용. numpy 배열 반환.
    """
    if isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
    else:
        pil = image

    w, h = pil.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = pil.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return np.array(canvas)


def build_gt_mask_path(image_path: str, dataset_root: str) -> Path:
    """
    테스트 이미지 경로 → GT 마스크 경로 변환.
    test/{defect_class}/{stem}{ext} → {dataset_root}/ground_truth/{defect_class}/{stem}_mask{ext}
    """
    p = Path(image_path)
    defect_class = p.parent.name
    stem = p.stem
    ext = p.suffix
    return Path(dataset_root) / "ground_truth" / defect_class / f"{stem}_mask{ext}"


# ──────────────────────────────────────────────────────────────
# 고수준 파이프라인 함수 (PIL Image 기반)
# ──────────────────────────────────────────────────────────────

def load_image(path: str) -> Image.Image:
    """이미지 로드 + 모드 정규화 (L/RGBA → RGB)."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def apply_filter(
    image: Image.Image,
    method: str,
    params: dict | None,
) -> Image.Image:
    """전처리 필터 적용. 내부적으로 독립 함수를 호출."""
    if method == "none":
        return image

    # Grayscale(L) 또는 RGBA 등 → RGB 변환 (FR-T2-07)
    # apply_preprocessing 경로에서는 load_image가 먼저 변환하지만,
    # 직접 호출 시에도 안전하게 동작하도록 방어적으로 처리.
    if image.mode != "RGB":
        image = image.convert("RGB")

    arr = np.array(image)  # H x W x 3, uint8

    if method == "he":
        result = apply_he(arr)
    elif method == "clahe":
        clip_limit = (params or {}).get("clip_limit", 2.0)
        result = apply_clahe(arr, clip_limit=clip_limit)
    elif method == "homomorphic":
        p = params or {}
        result = apply_homomorphic(
            arr,
            sigma=p.get("sigma", 10.0),
            gamma_H=p.get("gamma_H", 1.5),
            gamma_L=p.get("gamma_L", 0.5),
            normalize=p.get("normalize", True),
        )
    else:
        return image

    return Image.fromarray(result.astype(np.uint8), mode="RGB")


def normalize_to_tensor(
    image: Image.Image,
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    """PIL Image → torch.Tensor (C, H, W), 정규화 적용."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform(image)


def apply_preprocessing(
    image_path: str,
    config: dict,
) -> tuple[Image.Image, torch.Tensor]:
    """
    load_image → apply_filter → resize_with_padding → normalize_to_tensor
    파이프라인 전체 실행.
    반환: (미리보기용 PIL.Image, 학습용 torch.Tensor)
    """
    image = load_image(image_path)
    method = config.get("method", "none")
    params = config.get("params")
    filtered = apply_filter(image, method, params)
    target_size = config.get("image_size", 256)

    padded_arr = resize_with_padding(filtered, target_size)
    padded_pil = Image.fromarray(padded_arr)

    mean = config.get("mean", [0.485, 0.456, 0.406])
    std = config.get("std", [0.229, 0.224, 0.225])
    tensor = normalize_to_tensor(padded_pil, mean, std)
    return padded_pil, tensor


def tensor_to_display_image(tensor: torch.Tensor) -> Image.Image:
    """역정규화 없이 단순 clamp → PIL Image (미리보기용)."""
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def anomaly_map_to_heatmap(
    anomaly_map: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """Anomaly Score 2D 배열 → jet colormap 히트맵 PIL Image."""
    normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(normalized, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_rgb, mode="RGB")


def make_anomaly_overlay(
    image_path: str,
    anomaly_map: np.ndarray,
    threshold: float,
    score_min: float,
    score_max: float,
    alpha: float = 0.45,
) -> Image.Image:
    """이상 영역(threshold 초과 픽셀)에 빨간 반투명 오버레이를 합성한 PIL Image 반환."""
    orig     = Image.open(image_path).convert("RGB")
    orig_arr = np.array(orig, dtype=np.float32)
    h, w     = orig_arr.shape[:2]

    if score_max > score_min:
        amap_norm = np.clip(
            (anomaly_map - score_min) / (score_max - score_min), 0.0, 1.0
        ).astype(np.float32)
    else:
        amap_norm = np.zeros_like(anomaly_map, dtype=np.float32)

    amap_resized    = cv2.resize(amap_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    mask            = amap_resized >= threshold

    result          = orig_arr.copy()
    result[mask, 0] = result[mask, 0] * (1 - alpha) + 255 * alpha
    result[mask, 1] = result[mask, 1] * (1 - alpha)
    result[mask, 2] = result[mask, 2] * (1 - alpha)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def pil_to_png_stream(pil_image: Image.Image) -> io.BytesIO:
    """PIL Image → PNG BytesIO 스트림."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    return buf


def create_triplet_image(
    original: Image.Image,
    gt_mask: Image.Image | None,
    heatmap: Image.Image,
) -> Image.Image:
    """원본 / GT 마스크 / Heatmap 3개를 가로로 이어 붙인 단일 PIL Image."""
    size = original.size
    if gt_mask is None:
        gt_mask = Image.new("RGB", size, (0, 0, 0))
    else:
        gt_mask = gt_mask.convert("RGB").resize(size, Image.NEAREST)
    heatmap = heatmap.resize(size, Image.LANCZOS)

    triplet = Image.new("RGB", (size[0] * 3, size[1]))
    triplet.paste(original, (0, 0))
    triplet.paste(gt_mask, (size[0], 0))
    triplet.paste(heatmap, (size[0] * 2, 0))
    return triplet
