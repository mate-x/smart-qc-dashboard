from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from utils.config_manager import ConfigLoadError, load_config, save_config_section
from utils.image_utils import SUPPORTED_FORMATS, apply_filter, resize_with_padding
from utils.messages import MSG
from utils.storage import validate_imagenet_penalty_dir

_METHOD_LABELS = ["없음", "Homomorphic", "HE", "CLAHE"]
_METHOD_MAP: dict[str, str] = {
    "없음": "none",
    "Homomorphic": "homomorphic",
    "HE": "he",
    "CLAHE": "clahe",
}
_REVERSE_METHOD_MAP: dict[str, str] = {v: k for k, v in _METHOD_MAP.items()}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 순수 헬퍼 함수 (테스트 가능)
# ---------------------------------------------------------------------------

def _parse_float_list(text: str) -> list[float] | None:
    """쉼표 구분 문자열을 float 리스트로 파싱. 파싱 실패 시 None 반환."""
    try:
        return [float(x.strip()) for x in text.split(",")]
    except ValueError:
        return None


def _snap_image_size(size: int) -> int:
    """가장 가까운 32 배수로 반올림 후 [32, 1024] 범위로 클램프."""
    snapped = round(size / 32) * 32
    return max(32, min(1024, snapped))


def compute_threshold_ratio(
    threshold_method: str,
    threshold_value: float,
) -> tuple[float | None, float | None]:
    """
    FR-T2: Threshold 기준 정상/결함 비율 근사치 계산.

    percentile 방식: 정상 비율 = threshold_value / 100 (근사치)
    absolute 방식: 계산 불가 → (None, None) 반환
    """
    if threshold_method == "percentile":
        normal_ratio = round(threshold_value / 100.0, 6)
        defect_ratio = round(1.0 - normal_ratio, 6)
        return normal_ratio, defect_ratio
    return None, None


def build_efficientad_params(
    model_size: str,
    train_steps: int,
    optimizer: str,
    learning_rate: float,
    weight_decay: float,
    out_channels: int,
    padding: bool,
    ae_loss_weight: float,
    autoencoder_lr: float,
    autoencoder_weight_decay: float,
    lr_decay_epochs: int,
    lr_decay_factor: float,
    scheduler: str,
    use_imagenet_penalty: bool,
    penalty_batch_size: int,
) -> dict:
    """00_Global_Context 1.4절 EfficientAD model_params 오브젝트 생성."""
    return {
        "model_size": model_size,
        "train_steps": int(train_steps),
        "optimizer": optimizer,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "out_channels": int(out_channels),
        "padding": bool(padding),
        "ae_loss_weight": float(ae_loss_weight),
        "autoencoder_lr": float(autoencoder_lr),
        "autoencoder_weight_decay": float(autoencoder_weight_decay),
        "lr_decay_epochs": int(lr_decay_epochs),
        "lr_decay_factor": float(lr_decay_factor),
        "scheduler": scheduler,
        "use_imagenet_penalty": bool(use_imagenet_penalty),
        "penalty_batch_size": int(penalty_batch_size),
    }


def build_patchcore_params(
    backbone: str,
    pretrained_source: str,
    pretrained_path: str | None,
    coreset_sampling_ratio: float,
    neighbourhood_kernel_size: int,
    max_train: int,
    knn: int,
    top_k_ratio: float,
) -> dict:
    """00_Global_Context 1.4절 PatchCore model_params 오브젝트 생성."""
    return {
        "backbone": backbone,
        "pretrained_source": pretrained_source,
        "pretrained_path": pretrained_path,
        "coreset_sampling_ratio": float(coreset_sampling_ratio),
        "neighbourhood_kernel_size": int(neighbourhood_kernel_size),
        "max_train": int(max_train),
        "knn": int(knn),
        "top_k_ratio": float(top_k_ratio),
    }


def build_model_config(
    model_type: str,
    image_size: int,
    batch_size: int,
    random_seed: int,
    threshold_method: str,
    threshold_value: float,
    params: dict,
) -> dict:
    """00_Global_Context 1.7절 model_config 오브젝트 생성."""
    return {
        "model_type": model_type,
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "random_seed": int(random_seed),
        "threshold_method": threshold_method,
        "threshold_value": float(threshold_value),
        "params": params,
    }


def _build_preprocessing_config(
    method: str,
    params: dict | None,
    image_size: int,
    norm_label: str,
    mean: list[float],
    std: list[float],
) -> dict:
    """preprocessing_config 스키마 (00_Global_Context §1.6) 구성."""
    return {
        "method": method,
        "resize_mode": "padding",
        "image_size": image_size,
        "normalization": "imagenet" if norm_label == "ImageNet" else "custom",
        "mean": list(mean),
        "std": list(std),
        "params": params if params else None,
    }


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def render() -> None:
    st.header("탭2. 전처리 및 모델 설정")
    if not _guard():
        return
    _detect_device_once()

    # 최상단: 전처리 방식 radio + 모델 선택 radio 나란히
    col_prep, col_model = st.columns(2)
    with col_prep:
        method = _render_method_radio()
    with col_model:
        model_type = _render_model_radio()

    st.divider()

    # 전처리 영역
    st.markdown("### 전처리 영역")
    params = _render_method_params(method)
    image_size, norm_label, mean, std = _render_resize_norm_section()
    _render_preview(method, params, image_size)

    st.divider()

    # 모델 영역
    st.markdown("### 모델 영역")
    _render_device_info()
    batch_size, random_seed = _render_common_settings()
    st.divider()

    if model_type == "efficientad":
        model_params = _render_efficientad_params()
    else:
        model_params = _render_patchcore_params()

    st.divider()
    threshold_method, threshold_value = _render_threshold_section()
    _render_threshold_ratio_preview(threshold_method, threshold_value)

    st.divider()

    _render_save_area(
        method=method,
        params=params,
        image_size=image_size,
        norm_label=norm_label,
        mean=mean,
        std=std,
        model_type=model_type,
        batch_size=batch_size,
        random_seed=random_seed,
        threshold_method=threshold_method,
        threshold_value=threshold_value,
        model_params=model_params,
    )

    st.divider()
    _render_queue_section()


# ---------------------------------------------------------------------------
# Guard / 디바이스 감지
# ---------------------------------------------------------------------------

def _guard() -> bool:
    """dataset_meta 없으면 경고 후 렌더링 중단."""
    if st.session_state.get("dataset_meta") is None:
        st.warning(MSG["NO_DATASET"])
        return False
    return True


def _detect_device_once() -> None:
    """탭2 최초 진입 시 1회만 디바이스 감지 (idempotent)."""
    if st.session_state.get("device_info") is not None:
        return
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
            )
            device_info = {"device": "cuda", "gpu_name": gpu_name, "vram_gb": vram_gb}
        else:
            device_info = {"device": "cpu", "gpu_name": None, "vram_gb": None}
    except Exception:
        device_info = {"device": "cpu", "gpu_name": None, "vram_gb": None}
    st.session_state["device_info"] = device_info


def _render_device_info() -> None:
    device_info = st.session_state["device_info"]
    if device_info["device"] == "cuda":
        st.info(
            f"현재 디바이스: CUDA ({device_info['gpu_name']}), "
            f"VRAM: {device_info['vram_gb']:.1f} GB"
        )
    else:
        st.info("현재 디바이스: CPU")


# ---------------------------------------------------------------------------
# 전처리 영역 — method radio (최상단 column에 배치)
# ---------------------------------------------------------------------------

def _render_method_radio() -> str:
    """전처리 방식 radio. 선택된 내부 method 코드 반환."""
    label: str = st.radio(
        "전처리 방식 (Preprocessing Method)",
        options=_METHOD_LABELS,
        horizontal=True,
        key="tab2_method_label",
    )
    return _METHOD_MAP[label]


def _render_method_params(method: str) -> dict | None:
    """method에 따른 파라미터 위젯 렌더링. params dict 반환."""
    params: dict | None = None

    if method == "homomorphic":
        sigma = st.slider("sigma", 0.1, 50.0, 10.0, 0.1, key="tab2_sigma")
        gamma_h = st.slider("gamma_H", 1.0, 3.0, 1.5, 0.05, key="tab2_gamma_h")
        gamma_l = st.slider("gamma_L", 0.1, 1.0, 0.5, 0.05, key="tab2_gamma_l")
        normalize = st.checkbox("정규화 적용 (normalize)", value=True, key="tab2_normalize")
        params = {
            "sigma": sigma,
            "gamma_H": gamma_h,
            "gamma_L": gamma_l,
            "normalize": normalize,
        }
    elif method == "clahe":
        clip_limit = st.slider("클립 한계 (clipLimit)", 0.1, 40.0, 2.0, 0.1, key="tab2_clip_limit")
        params = {"clip_limit": clip_limit}
    elif method == "he":
        st.info("히스토그램 평탄화(HE)는 파라미터가 없습니다.")
        params = {}

    return params


# ---------------------------------------------------------------------------
# FR-T2-03: 이미지 크기 및 정규화 (전처리 영역 단독 소유)
# ---------------------------------------------------------------------------

def _render_resize_norm_section() -> tuple[int, str, list[float], list[float]]:
    st.subheader("이미지 크기 및 정규화")

    image_size = st.number_input(
        "이미지 크기 (image_size)",
        min_value=32,
        max_value=1024,
        value=256,
        step=32,
        key="tab2_image_size",
    )
    if int(image_size) % 32 != 0:
        st.error("32의 배수만 입력 가능합니다.")

    norm_label: str = st.radio(
        "정규화 방식 (Normalization)",
        ["ImageNet", "커스텀"],
        horizontal=True,
        key="tab2_norm_label",
    )

    if norm_label == "커스텀":
        mean_str = st.text_input(
            "mean (쉼표 구분, 예: 0.5,0.5,0.5)",
            value="0.5,0.5,0.5",
            key="tab2_mean",
        )
        std_str = st.text_input(
            "std (쉼표 구분, 예: 0.5,0.5,0.5)",
            value="0.5,0.5,0.5",
            key="tab2_std",
        )

        parsed_mean = _parse_float_list(mean_str)
        if parsed_mean is None:
            st.error("mean 값을 올바르게 입력해 주세요 (예: 0.5,0.5,0.5).")
            mean: list[float] = [0.5, 0.5, 0.5]
        elif len(parsed_mean) != 3:
            st.error("mean 값은 쉼표로 구분된 3개 숫자여야 합니다.")
            mean = [0.5, 0.5, 0.5]
        else:
            mean = parsed_mean

        parsed_std = _parse_float_list(std_str)
        if parsed_std is None:
            st.error("std 값을 올바르게 입력해 주세요 (예: 0.5,0.5,0.5).")
            std: list[float] = [0.5, 0.5, 0.5]
        elif len(parsed_std) != 3:
            st.error("std 값은 쉼표로 구분된 3개 숫자여야 합니다.")
            std = [0.5, 0.5, 0.5]
        else:
            std = parsed_std
    else:
        mean = list(_IMAGENET_MEAN)
        std = list(_IMAGENET_STD)

    return int(image_size), norm_label, mean, std


# ---------------------------------------------------------------------------
# FR-T2-04: 전처리 전후 미리보기
# ---------------------------------------------------------------------------

def _get_sample_image_path() -> Path | None:
    """train/good/ 알파벳 순 첫 번째 이미지 경로 반환."""
    dataset_path = st.session_state.get("dataset_path")
    if not dataset_path:
        return None
    train_good = Path(dataset_path) / "train" / "good"
    if not train_good.is_dir():
        return None
    images = sorted(f for f in train_good.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS)
    return images[0] if images else None


def _render_preview(method: str, params: dict | None, image_size: int) -> None:
    st.subheader("전처리 미리보기")
    if image_size % 32 != 0:
        st.warning("이미지 크기가 32의 배수가 아닙니다. 유효한 값을 입력하면 미리보기가 갱신됩니다.")
        return
    sample_path = _get_sample_image_path()
    if sample_path is None:
        st.info("미리보기: 샘플 이미지를 찾을 수 없습니다.")
        return

    try:
        original = Image.open(sample_path)
        if original.mode != "RGB":
            original = original.convert("RGB")
        processed = apply_filter(original, method, params)
        processed = resize_with_padding(processed, image_size)

        col1, col2 = st.columns(2)
        col1.image(original, caption="원본", use_container_width=True)
        col2.image(processed, caption=f"{method} 적용 후", use_container_width=True)
    except Exception as e:
        st.warning(f"미리보기 생성 실패: {e}")


# ---------------------------------------------------------------------------
# 모델 영역 — model radio (최상단 column에 배치)
# ---------------------------------------------------------------------------

def _render_model_radio() -> str:
    """모델 선택 radio. 내부 model_type 코드 반환."""
    label: str = st.radio(
        "모델 선택 (Model Type)",
        options=["EfficientAD", "PatchCore"],
        horizontal=True,
        key="tab2_model_label",
    )
    return "efficientad" if label == "EfficientAD" else "patchcore"


def _render_common_settings() -> tuple[int, int]:
    """공통 설정: batch_size + random_seed (image_size 제외)."""
    st.subheader("공통 설정")
    col1, col2 = st.columns(2)
    with col1:
        batch_size = int(
            st.number_input(
                "배치 크기 (batch_size)",
                min_value=1, max_value=128,
                value=16, step=1,
                key="tab2_batch_size",
            )
        )
    with col2:
        random_seed = int(
            st.number_input(
                "랜덤 시드 (random_seed)",
                min_value=0, max_value=2_147_483_647,
                value=42, step=1,
                key="tab2_random_seed",
            )
        )
    return batch_size, random_seed


# ---------------------------------------------------------------------------
# EfficientAD 파라미터 UI
# ---------------------------------------------------------------------------

def _render_efficientad_params() -> dict:
    st.subheader("EfficientAD 파라미터")

    col1, col2 = st.columns(2)
    with col1:
        model_size: str = st.radio(
            "모델 크기 (model_size)", ["small", "medium"],
            index=1, horizontal=True, key="tab2_ead_model_size",
        )
        train_steps = int(
            st.number_input(
                "학습 단계 수 (train_steps)", 1, 200_000, 70_000, 1,
                key="tab2_ead_train_steps",
            )
        )
        optimizer: str = st.selectbox(
            "옵티마이저 (optimizer)", ["adam", "adamw", "sgd"],
            index=0, key="tab2_ead_optimizer",
        )
        out_channels = int(
            st.selectbox(
                "출력 채널 수 (out_channels)", [128, 256, 384, 512],
                index=2, key="tab2_ead_out_channels",
            )
        )

    with col2:
        learning_rate = float(
            st.number_input(
                "학습률 (learning_rate)", 1e-6, 1e-1, 1e-4,
                format="%.6f", key="tab2_ead_lr",
            )
        )
        weight_decay = float(
            st.number_input(
                "가중치 감쇠 (weight_decay)", 0.0, 0.1, 1e-4,
                format="%.6f", key="tab2_ead_wd",
            )
        )
        padding = bool(st.checkbox("패딩 사용 (padding)", value=False, key="tab2_ead_padding"))

    st.markdown("**AE Loss 비중 (ae_loss_weight)**")
    ae_loss_weight = float(
        st.slider("ae_loss_weight (α)", 0.0, 1.0, 0.5, 0.01, key="tab2_ead_ae_weight")
    )

    with st.expander("고급 설정 (Advanced Settings)"):
        adv1, adv2 = st.columns(2)
        with adv1:
            autoencoder_lr = float(
                st.number_input(
                    "AE 학습률 (autoencoder_lr)", 1e-6, 1e-1, 1e-4,
                    format="%.6f", key="tab2_ead_ae_lr",
                )
            )
            autoencoder_wd = float(
                st.number_input(
                    "AE 가중치 감쇠 (autoencoder_weight_decay)", 0.0, 0.1, 1e-5,
                    format="%.6f", key="tab2_ead_ae_wd",
                )
            )
            lr_decay_epochs = int(
                st.number_input(
                    "LR 감쇠 에포크 (lr_decay_epochs)", 1000, 200_000, 50_000, 1000,
                    key="tab2_ead_decay_ep",
                )
            )
        with adv2:
            lr_decay_factor = float(
                st.slider("LR 감쇠 계수 (lr_decay_factor)", 0.01, 1.0, 0.1, 0.01,
                          key="tab2_ead_decay_f")
            )
            scheduler: str = st.selectbox(
                "스케줄러 (scheduler)", ["StepLR", "CosineAnnealingLR"],
                index=0, key="tab2_ead_sched",
            )
            use_imagenet_penalty = st.checkbox(
                "ImageNet Penalty 사용 (use_imagenet_penalty)",
                value=False, key="tab2_ead_use_penalty",
            )
            penalty_bs = int(
                st.number_input(
                    "패널티 배치 크기 (penalty_batch_size)", 1, 64, 8, 1,
                    key="tab2_ead_pen_bs",
                )
            )

        if use_imagenet_penalty:
            ok, count = validate_imagenet_penalty_dir()
            if not ok:
                st.warning(
                    "ImageNet penalty 디렉터리에 이미지가 없습니다. "
                    "use_imagenet_penalty가 True이면 EfficientAD 학습이 실패합니다. "
                    "이미지를 추가하거나 체크박스를 해제하세요."
                )
            elif count < 1000:
                st.caption(f"ImageNet 패널티 디렉터리: {count}장 확인됨 (1,000장 미만 — 품질 저하 가능)")
            else:
                st.caption(f"ImageNet 패널티 디렉터리: {count}장 확인됨")

    return build_efficientad_params(
        model_size=model_size,
        train_steps=train_steps,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        out_channels=out_channels,
        padding=padding,
        ae_loss_weight=ae_loss_weight,
        autoencoder_lr=autoencoder_lr,
        autoencoder_weight_decay=autoencoder_wd,
        lr_decay_epochs=lr_decay_epochs,
        lr_decay_factor=lr_decay_factor,
        scheduler=scheduler,
        use_imagenet_penalty=use_imagenet_penalty,
        penalty_batch_size=penalty_bs,
    )


# ---------------------------------------------------------------------------
# PatchCore 파라미터 UI
# ---------------------------------------------------------------------------

def _render_patchcore_params() -> dict:
    st.subheader("PatchCore 파라미터")

    col1, col2 = st.columns(2)
    with col1:
        backbone: str = st.selectbox(
            "백본 (backbone)",
            ["wide_resnet50_2", "resnet18", "resnet50"],
            index=0, key="tab2_pc_backbone",
        )
        pretrained_label: str = st.radio(
            "사전학습 가중치 출처 (pretrained_source)",
            ["torchvision", "로컬 경로"],
            horizontal=True, key="tab2_pc_pretrained",
        )
        pretrained_source = "torchvision" if pretrained_label == "torchvision" else "local"

        pretrained_path: str | None = None
        if pretrained_source == "local":
            pretrained_path = (
                st.text_input(
                    "로컬 가중치 경로 (pretrained_path)",
                    key="tab2_pc_pretrained_path",
                ) or None
            )

    with col2:
        coreset_sampling_ratio = float(
            st.slider(
                "코어셋 비율 (coreset_sampling_ratio)",
                0.01, 1.0, 0.1, 0.01, key="tab2_pc_coreset",
            )
        )
        neighbourhood_kernel_size = int(
            st.select_slider(
                "이웃 커널 크기 (neighbourhood_kernel_size)",
                options=[1, 3, 5, 7, 9],
                value=3, key="tab2_pc_kernel",
            )
        )

    with st.expander("고급 설정 (Advanced Settings)"):
        adv1, adv2 = st.columns(2)
        with adv1:
            max_train = int(
                st.number_input(
                    "최대 학습 샘플 수 (max_train)", 100, 10_000, 1000, 100,
                    key="tab2_pc_max_train",
                )
            )
        with adv2:
            knn = int(
                st.number_input(
                    "k-NN 이웃 수 (knn)", 1, 50, 9, 1,
                    key="tab2_pc_knn",
                )
            )
            top_k_ratio = float(
                st.slider("Top-k 비율 (top_k_ratio)", 0.0, 1.0, 0.1, 0.01, key="tab2_pc_top_k")
            )

    return build_patchcore_params(
        backbone=backbone,
        pretrained_source=pretrained_source,
        pretrained_path=pretrained_path,
        coreset_sampling_ratio=coreset_sampling_ratio,
        neighbourhood_kernel_size=neighbourhood_kernel_size,
        max_train=max_train,
        knn=knn,
        top_k_ratio=top_k_ratio,
    )


# ---------------------------------------------------------------------------
# Threshold 설정
# ---------------------------------------------------------------------------

def _render_threshold_section() -> tuple[str, float]:
    st.subheader("Threshold 설정")
    threshold_label: str = st.radio(
        "Threshold 방식",
        options=["Percentile (백분위)", "Absolute (절대값)"],
        horizontal=True,
        key="tab2_threshold_label",
    )
    threshold_method = "percentile" if "Percentile" in threshold_label else "absolute"

    if threshold_method == "percentile":
        threshold_value = float(
            st.slider("백분위 값", 0.0, 100.0, 95.0, 0.5, key="tab2_threshold_pct")
        )
    else:
        threshold_value = float(
            st.slider("절대값", 0.0, 1.0, 0.5, 0.01, key="tab2_threshold_abs")
        )

    return threshold_method, threshold_value


def _render_threshold_ratio_preview(threshold_method: str, threshold_value: float) -> None:
    """dataset_meta가 존재할 때만 threshold 기준 비율 예상치를 표시한다."""
    if st.session_state.get("dataset_meta") is None:
        return

    normal_ratio, defect_ratio = compute_threshold_ratio(threshold_method, threshold_value)
    col1, col2 = st.columns(2)
    with col1:
        if normal_ratio is not None:
            st.metric("예상 정상 판정 비율", f"{normal_ratio:.1%}")
        else:
            st.metric("예상 정상 판정 비율", "학습 후 확인 가능")
    with col2:
        if defect_ratio is not None:
            st.metric("예상 결함 판정 비율", f"{defect_ratio:.1%}")
        else:
            st.metric("예상 결함 판정 비율", "학습 후 확인 가능")


# ---------------------------------------------------------------------------
# 실험 대기열 — 순수 헬퍼 함수 (FR-T2-16~18)
# ---------------------------------------------------------------------------

# 상태별 행 배경색 (FR-T2-17 상태 색상 규칙)
_STATUS_COLORS: dict[str, str] = {
    "대기중": "#e8e8e8",   # 회색
    "진행중": "#cce5ff",   # 파란색
    "완료":   "#d4edda",   # 초록색
    "실패":   "#f8d7da",   # 빨간색
    "건너뜀": "#fde8c8",   # 주황색
}

_VALID_STATUSES = frozenset(_STATUS_COLORS)


def _make_queue_item(preprocessing_config: dict, model_config: dict) -> dict:
    """
    대기열 항목 dict 생성 (FR-T2-16).
    name 형식: {MODEL_TYPE}_{uuid4().hex[:4]}
    status 초기값: "대기중"
    preprocessing_config / model_config 는 얕은 복사본 저장.
    """
    model_type = model_config.get("model_type", "model")
    name = f"{model_type.upper()}_{uuid.uuid4().hex[:4]}"
    return {
        "name":                name,
        "preprocessing_config": dict(preprocessing_config),
        "model_config":         dict(model_config),
        "status":              "대기중",
    }


def _build_queue_df(queue_items: list[dict]) -> pd.DataFrame:
    """
    대기열 항목 목록을 테이블 DataFrame으로 변환 (FR-T2-17).
    컬럼: 순번 / 실험명 / 모델 / 상태
    efficientad → "EAD", patchcore → "PC"
    빈 목록이면 빈 DataFrame (컬럼만 존재).
    """
    if not queue_items:
        return pd.DataFrame(columns=["순번", "실험명", "모델", "상태"])

    rows = []
    for i, item in enumerate(queue_items):
        model_type = (item.get("model_config") or {}).get("model_type", "?")
        model_abbr = "EAD" if model_type == "efficientad" else "PC"
        rows.append({
            "순번":   i + 1,
            "실험명": item.get("name", f"exp_{i + 1}"),
            "모델":   model_abbr,
            "상태":   item.get("status", "대기중"),
        })
    return pd.DataFrame(rows)


def _style_queue_rows(row: pd.Series) -> list[str]:
    """pandas Styler용 행 배경색 적용 (FR-T2-17 상태 색상 규칙)."""
    color = _STATUS_COLORS.get(str(row.get("상태", "")), "")
    bg = f"background-color: {color}" if color else ""
    return [bg] * len(row)


# ---------------------------------------------------------------------------
# 설정 저장 버튼 영역 (preprocessing_config + model_config 동시 Write)
# ---------------------------------------------------------------------------

def _render_save_area(
    method: str,
    params: dict | None,
    image_size: int,
    norm_label: str,
    mean: list[float],
    std: list[float],
    model_type: str,
    batch_size: int,
    random_seed: int,
    threshold_method: str,
    threshold_value: float,
    model_params: dict,
) -> None:
    st.subheader("설정 저장")

    if image_size % 32 != 0:
        st.error("이미지 크기가 32의 배수가 아닙니다. 수정 후 저장해 주세요.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("설정 저장", type="primary", key="tab2_btn_save"):
            pre_cfg = _build_preprocessing_config(method, params, image_size, norm_label, mean, std)
            mdl_cfg = build_model_config(
                model_type, image_size, batch_size, random_seed,
                threshold_method, threshold_value, model_params,
            )
            st.session_state["preprocessing_config"] = pre_cfg
            st.session_state["model_config"] = mdl_cfg
            st.success("설정이 저장되었습니다.")

    with col2:
        if st.button("configs.yaml 저장", key="tab2_btn_yaml_save"):
            pre_cfg = _build_preprocessing_config(method, params, image_size, norm_label, mean, std)
            mdl_cfg = build_model_config(
                model_type, image_size, batch_size, random_seed,
                threshold_method, threshold_value, model_params,
            )
            try:
                save_config_section("preprocessing", pre_cfg)
                save_config_section("model", mdl_cfg)
                st.success("configs.yaml에 저장되었습니다.")
            except RuntimeError as e:
                st.error(str(e))

    with col3:
        if st.button("configs.yaml 불러오기", key="tab2_btn_yaml_load"):
            st.session_state["_tab2_show_load"] = True

    with col4:
        if st.button("📋 대기열에 추가", key="tab2_btn_enqueue", use_container_width=True):
            pre_cfg = _build_preprocessing_config(method, params, image_size, norm_label, mean, std)
            mdl_cfg = build_model_config(
                model_type, image_size, batch_size, random_seed,
                threshold_method, threshold_value, model_params,
            )
            item = _make_queue_item(pre_cfg, mdl_cfg)
            q = list(st.session_state.get("experiment_queue", []))
            q.append(item)
            st.session_state["experiment_queue"] = q
            st.success(f"'{item['name']}'이(가) 대기열에 추가되었습니다.")

    if st.session_state.get("_tab2_show_load"):
        _render_load_ui()


def _render_load_ui() -> None:
    """UC-09: configs.yaml에서 preprocessing + model 섹션 모두 불러와 위젯에 반영."""
    yaml_path: str = st.text_input(
        "configs.yaml 경로", value="./configs.yaml", key="tab2_load_path"
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("확인", key="tab2_btn_load_confirm"):
            try:
                raw = load_config(yaml_path)
            except ConfigLoadError as e:
                st.error(f"불러오기 실패: {e}")
                return
            pre_section = raw.get("preprocessing")
            mdl_section = raw.get("model")
            if pre_section is None and mdl_section is None:
                st.error("configs.yaml에 preprocessing / model 섹션이 없습니다.")
                return
            if pre_section:
                _apply_preprocessing_config_to_widgets(pre_section)
                st.session_state["preprocessing_config"] = pre_section
            if mdl_section:
                _apply_model_config_to_widgets(mdl_section)
                st.session_state["model_config"] = mdl_section
            st.session_state["_tab2_show_load"] = False
            st.success("설정을 불러왔습니다.")
            st.rerun()
    with c2:
        if st.button("취소", key="tab2_btn_load_cancel"):
            st.session_state["_tab2_show_load"] = False
            st.rerun()


# ---------------------------------------------------------------------------
# 위젯 프리필 — configs.yaml 불러오기 후 rerun 전 적용
# ---------------------------------------------------------------------------

def _apply_preprocessing_config_to_widgets(config: dict) -> None:
    """전처리 설정을 위젯 key에 주입 → 다음 rerun에 반영."""
    method = config.get("method", "none")
    st.session_state["tab2_method_label"] = _REVERSE_METHOD_MAP.get(method, "없음")

    params = config.get("params") or {}
    if method == "homomorphic":
        st.session_state["tab2_sigma"]     = max(0.1,  min(50.0, float(params.get("sigma",   10.0))))
        st.session_state["tab2_gamma_h"]   = max(1.0,  min(3.0,  float(params.get("gamma_H", 1.5))))
        st.session_state["tab2_gamma_l"]   = max(0.1,  min(1.0,  float(params.get("gamma_L", 0.5))))
        st.session_state["tab2_normalize"] = bool(params.get("normalize", True))
    elif method == "clahe":
        st.session_state["tab2_clip_limit"] = max(0.1, min(40.0, float(params.get("clip_limit", 2.0))))

    st.session_state["tab2_image_size"] = _snap_image_size(int(config.get("image_size", 256)))

    norm = config.get("normalization", "imagenet")
    st.session_state["tab2_norm_label"] = "ImageNet" if norm == "imagenet" else "커스텀"

    if norm != "imagenet":
        mean = config.get("mean", [0.5, 0.5, 0.5])
        std  = config.get("std",  [0.5, 0.5, 0.5])
        if not isinstance(mean, list) or len(mean) != 3:
            mean = [0.5, 0.5, 0.5]
        if not isinstance(std, list) or len(std) != 3:
            std = [0.5, 0.5, 0.5]
        st.session_state["tab2_mean"] = ",".join(str(v) for v in mean)
        st.session_state["tab2_std"]  = ",".join(str(v) for v in std)


def _apply_model_config_to_widgets(cfg: dict) -> None:
    """모델 설정을 위젯 key에 주입 → 다음 rerun에 반영. image_size는 전처리 영역 소유이므로 생략."""
    model_type = cfg.get("model_type", "patchcore")
    st.session_state["tab2_model_label"] = (
        "EfficientAD" if model_type == "efficientad" else "PatchCore"
    )
    if cfg.get("batch_size"):
        st.session_state["tab2_batch_size"] = int(cfg["batch_size"])
    if cfg.get("random_seed") is not None:
        st.session_state["tab2_random_seed"] = int(cfg["random_seed"])

    threshold_method = cfg.get("threshold_method", "percentile")
    st.session_state["tab2_threshold_label"] = (
        "Percentile (백분위)" if threshold_method == "percentile" else "Absolute (절대값)"
    )
    threshold_value = cfg.get("threshold_value", 95.0)
    if threshold_method == "percentile":
        st.session_state["tab2_threshold_pct"] = float(threshold_value)
    else:
        st.session_state["tab2_threshold_abs"] = float(threshold_value)

    params = cfg.get("params", {})
    if model_type == "patchcore":
        _apply_patchcore_widgets(params)
    elif model_type == "efficientad":
        _apply_efficientad_widgets(params)


def _apply_patchcore_widgets(params: dict) -> None:
    if params.get("backbone"):
        st.session_state["tab2_pc_backbone"] = params["backbone"]
    pretrained_source = params.get("pretrained_source", "torchvision")
    st.session_state["tab2_pc_pretrained"] = (
        "torchvision" if pretrained_source == "torchvision" else "로컬 경로"
    )
    if params.get("pretrained_path"):
        st.session_state["tab2_pc_pretrained_path"] = params["pretrained_path"]
    if params.get("coreset_sampling_ratio") is not None:
        st.session_state["tab2_pc_coreset"] = float(params["coreset_sampling_ratio"])
    if params.get("neighbourhood_kernel_size") is not None:
        val = int(params["neighbourhood_kernel_size"])
        if val in (1, 3, 5, 7, 9):
            st.session_state["tab2_pc_kernel"] = val
    if params.get("max_train") is not None:
        st.session_state["tab2_pc_max_train"] = int(params["max_train"])
    if params.get("knn") is not None:
        st.session_state["tab2_pc_knn"] = int(params["knn"])
    if params.get("top_k_ratio") is not None:
        st.session_state["tab2_pc_top_k"] = float(params["top_k_ratio"])


def _apply_efficientad_widgets(params: dict) -> None:
    if params.get("model_size"):
        st.session_state["tab2_ead_model_size"] = params["model_size"]
    if params.get("train_steps") is not None:
        st.session_state["tab2_ead_train_steps"] = int(params["train_steps"])
    if params.get("optimizer"):
        st.session_state["tab2_ead_optimizer"] = params["optimizer"]
    if params.get("learning_rate") is not None:
        st.session_state["tab2_ead_lr"] = float(params["learning_rate"])
    if params.get("weight_decay") is not None:
        st.session_state["tab2_ead_wd"] = float(params["weight_decay"])
    if params.get("out_channels") is not None:
        st.session_state["tab2_ead_out_channels"] = int(params["out_channels"])
    if params.get("padding") is not None:
        st.session_state["tab2_ead_padding"] = bool(params["padding"])
    if params.get("ae_loss_weight") is not None:
        st.session_state["tab2_ead_ae_weight"] = float(params["ae_loss_weight"])
    if params.get("autoencoder_lr") is not None:
        st.session_state["tab2_ead_ae_lr"] = float(params["autoencoder_lr"])
    if params.get("autoencoder_weight_decay") is not None:
        st.session_state["tab2_ead_ae_wd"] = float(params["autoencoder_weight_decay"])
    if params.get("lr_decay_epochs") is not None:
        st.session_state["tab2_ead_decay_ep"] = int(params["lr_decay_epochs"])
    if params.get("lr_decay_factor") is not None:
        st.session_state["tab2_ead_decay_f"] = float(params["lr_decay_factor"])
    if params.get("scheduler"):
        st.session_state["tab2_ead_sched"] = params["scheduler"]
    if params.get("use_imagenet_penalty") is not None:
        st.session_state["tab2_ead_use_penalty"] = bool(params["use_imagenet_penalty"])
    if params.get("penalty_batch_size") is not None:
        st.session_state["tab2_ead_pen_bs"] = int(params["penalty_batch_size"])


# ---------------------------------------------------------------------------
# 실험 대기열 UI (FR-T2-17, FR-T2-18)
# ---------------------------------------------------------------------------

def _render_queue_section() -> None:
    """FR-T2-17/18: 탭2 하단 실험 대기열 2분할 UI."""
    queue_items: list[dict] = st.session_state.get("experiment_queue", [])

    st.subheader("📋 실험 대기열")

    col_left, col_right = st.columns(2)

    with col_left:
        selected_idx = _render_queue_table(queue_items)

    with col_right:
        _render_queue_detail(queue_items, selected_idx)


def _render_queue_table(queue_items: list[dict]) -> int | None:
    """
    FR-T2-17: 대기열 테이블 + 순서 조정(▲▼) + 삭제 버튼.
    선택된 항목의 인덱스(int) 반환. 선택 없으면 None.

    ▲/▼/🗑 은 status == "대기중" 항목에만 활성화.
    """
    if not queue_items:
        st.info("대기열이 비어 있습니다. '대기열에 추가' 버튼으로 실험을 추가하세요.")
        return None

    df = _build_queue_df(queue_items)

    event = st.dataframe(
        df.style.apply(_style_queue_rows, axis=1),
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        key="tab2_queue_table",
        hide_index=True,
    )

    selected_rows = event.selection.rows if event else []
    selected_idx: int | None = selected_rows[0] if selected_rows else None

    # ▲▼🗑 활성화 조건: 선택된 항목이 "대기중"일 때만
    is_pending = (
        selected_idx is not None
        and selected_idx < len(queue_items)
        and queue_items[selected_idx].get("status") == "대기중"
    )

    col_up, col_down, col_del = st.columns(3)

    with col_up:
        if st.button(
            "▲ 위로",
            disabled=not (is_pending and selected_idx > 0),
            use_container_width=True,
            key="tab2_queue_up",
        ):
            q = list(st.session_state["experiment_queue"])
            q[selected_idx], q[selected_idx - 1] = q[selected_idx - 1], q[selected_idx]
            st.session_state["experiment_queue"] = q
            st.rerun()

    with col_down:
        if st.button(
            "▼ 아래로",
            disabled=not (is_pending and selected_idx < len(queue_items) - 1),
            use_container_width=True,
            key="tab2_queue_down",
        ):
            q = list(st.session_state["experiment_queue"])
            q[selected_idx], q[selected_idx + 1] = q[selected_idx + 1], q[selected_idx]
            st.session_state["experiment_queue"] = q
            st.rerun()

    with col_del:
        if st.button(
            "🗑 삭제",
            disabled=not is_pending,
            use_container_width=True,
            key="tab2_queue_delete",
        ):
            q = list(st.session_state["experiment_queue"])
            q.pop(selected_idx)
            st.session_state["experiment_queue"] = q
            st.rerun()

    return selected_idx


def _render_queue_detail(queue_items: list[dict], selected_idx: int | None) -> None:
    """FR-T2-18: 선택된 항목의 전처리 + 모델 파라미터 상세 표시."""
    if selected_idx is None or selected_idx >= len(queue_items):
        st.info("항목을 선택하면 상세 설정이 표시됩니다.")
        return

    item = queue_items[selected_idx]
    st.markdown(f"**{item.get('name', '?')}**")
    st.json({
        "preprocessing": item.get("preprocessing_config", {}),
        "model":         item.get("model_config", {}),
    })
