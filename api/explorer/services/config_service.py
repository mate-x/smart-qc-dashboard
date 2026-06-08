"""
api/explorer/services/config_service.py

탭2 · 설정:
    get_config()              현재 설정 + device_info 반환 (device_info 1회 캐싱)
    save_config(...)          preprocessing_config + model_config 서버 저장
    preview_threshold(...)    threshold 기준 정상/결함 비율 계산 (순수 함수)
    save_config_yaml()        서버 상태 → configs.yaml 저장
    load_config_yaml()        configs.yaml → 서버 상태 반영 후 반환
"""
from __future__ import annotations

from api.explorer.state import get_state
from utils.config_manager import ConfigLoadError, load_config, save_config_section


def get_config() -> dict:
    state = get_state()
    return {
        "preprocessing_config": state["preprocessing_config"],
        "model_cfg":            state["model_config"],
        "device_info":          _detect_device_once(),
    }


def save_config(preprocessing_config: dict, model_config: dict) -> None:
    state = get_state()
    state["preprocessing_config"] = preprocessing_config
    state["model_config"]         = model_config


def preview_threshold(threshold_method: str, threshold_value: float) -> dict:
    """percentile 방식일 때 정상/결함 비율 근사치 반환. absolute 방식은 None."""
    if threshold_method == "percentile":
        normal_ratio = round(threshold_value / 100.0, 6)
        defect_ratio = round(1.0 - normal_ratio, 6)
        return {"normal_ratio": normal_ratio, "defect_ratio": defect_ratio}
    return {"normal_ratio": None, "defect_ratio": None}


def preview_preprocessing_image(
    dataset_path: str,
    background_method: str,
    method: str,
    params: dict | None,
    image_size: int,
) -> dict:
    """전처리 전/후 이미지를 base64 PNG로 반환."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    from utils.dataset_converter import detect_ok_ng_dirs
    from utils.image_utils import SUPPORTED_FORMATS, apply_filter, resize_with_padding

    root = __import__("pathlib").Path(dataset_path)
    warning: str | None = None
    sample_path = None

    # SAM2 경로 우선 탐색
    if background_method == "sam2":
        bg_dir = root / "background_clean" / "train" / "good"
        if bg_dir.is_dir():
            imgs = sorted(f for f in bg_dir.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS)
            if imgs:
                sample_path = imgs[0]
        if sample_path is None:
            warning = "SAM2 전처리 이미지를 찾을 수 없어 원본으로 표시합니다."

    # MVTec: train/good/
    if sample_path is None:
        train_good = root / "train" / "good"
        if train_good.is_dir():
            imgs = sorted(f for f in train_good.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS)
            if imgs:
                sample_path = imgs[0]

    # OK/NG: OK 계열 폴더 자동 탐색
    if sample_path is None:
        ok_dir, _ = detect_ok_ng_dirs(root)
        if ok_dir is not None:
            imgs = sorted(f for f in ok_dir.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS)
            if imgs:
                sample_path = imgs[0]

    if sample_path is None:
        raise ValueError("샘플 이미지를 찾을 수 없습니다.")

    original = Image.open(sample_path).convert("RGB")
    processed = apply_filter(original, method, params)

    original_arr = resize_with_padding(original, image_size)
    processed_arr = resize_with_padding(processed, image_size)

    def _to_b64(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return {
        "original_b64":  _to_b64(original_arr),
        "processed_b64": _to_b64(processed_arr),
        "warning":       warning,
    }


def save_config_yaml() -> None:
    state = get_state()
    if not state["preprocessing_config"] and not state["model_config"]:
        raise ValueError("저장할 설정이 없습니다. 먼저 설정을 저장해 주세요.")
    if state["preprocessing_config"]:
        save_config_section("preprocessing", state["preprocessing_config"])
    if state["model_config"]:
        save_config_section("model", state["model_config"])


def load_config_yaml() -> dict:
    """
    ./configs.yaml 로드 후 서버 state에 반영.

    Raises:
        ValueError: 파싱 실패 또는 유효 섹션 없음
    """
    try:
        raw = load_config("./configs.yaml")
    except ConfigLoadError as e:
        raise ValueError(str(e))

    pre_section = raw.get("preprocessing")
    mdl_section = raw.get("model")

    if pre_section is None and mdl_section is None:
        raise ValueError("configs.yaml에 preprocessing / model 섹션이 없습니다.")

    state = get_state()
    if pre_section:
        state["preprocessing_config"] = pre_section
    if mdl_section:
        state["model_config"] = mdl_section

    return {
        "preprocessing_config": state["preprocessing_config"],
        "model_cfg":            state["model_config"],
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _detect_device_once() -> dict:
    """GPU/CPU 정보를 최초 1회만 감지하고 state에 캐싱."""
    state = get_state()
    if state["device_info"] is not None:
        return state["device_info"]

    try:
        import torch
        if torch.cuda.is_available():
            device_info = {
                "device":   "cuda",
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_gb":  round(
                    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
                ),
            }
        else:
            device_info = {"device": "cpu", "gpu_name": None, "vram_gb": None}
    except Exception:
        device_info = {"device": "cpu", "gpu_name": None, "vram_gb": None}

    state["device_info"] = device_info
    return device_info
