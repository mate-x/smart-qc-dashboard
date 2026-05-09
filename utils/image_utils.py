from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode == "L":
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def apply_filter(
    image: Image.Image,
    method: str,
    params: dict | None,
) -> Image.Image:
    if method == "none":
        return image

    arr = np.array(image)  # H x W x 3, uint8

    if method == "he":
        channels = [cv2.equalizeHist(arr[:, :, c]) for c in range(3)]
        result = np.stack(channels, axis=2)

    elif method == "clahe":
        clip_limit = (params or {}).get("clip_limit", 2.0)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        channels = [clahe.apply(arr[:, :, c]) for c in range(3)]
        result = np.stack(channels, axis=2)

    elif method == "homomorphic":
        p = params or {}
        sigma = p.get("sigma", 10.0)
        gamma_h = p.get("gamma_H", 1.5)
        gamma_l = p.get("gamma_L", 0.5)
        normalize = p.get("normalize", True)

        def _homo_channel(ch: np.ndarray) -> np.ndarray:
            ch_float = ch.astype(np.float32) + 1e-6
            log_ch = np.log(ch_float)
            blur = cv2.GaussianBlur(log_ch, (0, 0), sigma)
            high = log_ch - blur
            out = gamma_h * high + gamma_l * blur
            out = np.exp(out)
            if normalize:
                out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
            return np.clip(out, 0, 255).astype(np.uint8)

        channels = [_homo_channel(arr[:, :, c]) for c in range(3)]
        result = np.stack(channels, axis=2)

    else:
        return image

    return Image.fromarray(result.astype(np.uint8), mode="RGB")


def resize_with_padding(
    image: Image.Image,
    target_size: int,
) -> Image.Image:
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def normalize_to_tensor(
    image: Image.Image,
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform(image)


def apply_preprocessing(
    image_path: str,
    config: dict,
) -> tuple[Image.Image, torch.Tensor]:
    image = load_image(image_path)
    method = config.get("method", "none")
    params = config.get("params")
    filtered = apply_filter(image, method, params)
    target_size = config.get("image_size", 256)
    padded = resize_with_padding(filtered, target_size)
    mean = config.get("mean", [0.485, 0.456, 0.406])
    std = config.get("std", [0.229, 0.224, 0.225])
    tensor = normalize_to_tensor(padded, mean, std)
    return padded, tensor


def tensor_to_display_image(tensor: torch.Tensor) -> Image.Image:
    # 역정규화 없이 단순 clamp → PIL 변환 (미리보기용)
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def anomaly_map_to_heatmap(
    anomaly_map: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(normalized, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_rgb, mode="RGB")


def create_triplet_image(
    original: Image.Image,
    gt_mask: Image.Image | None,
    heatmap: Image.Image,
) -> Image.Image:
    size = original.size  # (W, H)
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
