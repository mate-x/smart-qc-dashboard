from __future__ import annotations

import itertools
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
    background_method: str = "none",
) -> dict:
    """preprocessing_config 스키마 (00_Global_Context §1.6) 구성."""
    return {
        "method": method,
        "background_method": background_method,
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

    # 메인 2단: 전처리 영역(좌) + 모델 영역(우)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 전처리 영역")
        image_size = _render_image_size_section()
        method = _render_method_radio()
        params = _render_method_params(method)
        background_method = _render_background_radio()
        _render_preview(method, params, image_size, background_method)

    # 정규화는 ImageNet 고정
    norm_label = "ImageNet"
    mean: list[float] = list(_IMAGENET_MEAN)
    std: list[float]  = list(_IMAGENET_STD)

    with col_right:
        st.markdown("### 모델 영역")
        model_type = _render_model_radio()
        _render_device_info()
        batch_size, random_seed = _render_common_settings()
        st.divider()
        if model_type == "efficientad":
            model_params = _render_efficientad_params()
        else:
            model_params = _render_patchcore_params()

    st.divider()

    # 하단 3단: Threshold + 설정 저장(좌) / 대기열 테이블(중) / 개별 파라미터(우)
    col_thr, col_table, col_detail = st.columns(3)

    with col_thr:
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
            background_method=background_method,
            model_type=model_type,
            batch_size=batch_size,
            random_seed=random_seed,
            threshold_method=threshold_method,
            threshold_value=threshold_value,
            model_params=model_params,
        )

    queue_items: list[dict] = st.session_state.get("experiment_queue", [])

    with col_table:
        st.subheader("실험 대기열 테이블")
        selected_idx = _render_queue_table(queue_items)

    with col_detail:
        st.subheader("개별 실험 파라미터")
        _render_queue_detail(queue_items, selected_idx)

    st.divider()
    _render_auto_experiment_section()


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
    """필터 방식 radio. 선택된 내부 method 코드 반환."""
    label: str = st.radio(
        "필터 방식",
        options=_METHOD_LABELS,
        horizontal=True,
        key="tab2_method_label",
    )
    return _METHOD_MAP[label]


def _render_background_radio() -> str:
    """배경 분리 방식 radio. "none" 또는 "sam2" 반환."""
    label: str = st.radio(
        "배경 분리 방식",
        options=["없음", "SAM2 Segmentation"],
        horizontal=True,
        key="tab2_background_label",
    )
    bg_method = "none" if label == "없음" else "sam2"

    if bg_method == "sam2":
        dataset_path = st.session_state.get("dataset_path")
        if dataset_path:
            bg_clean = Path(dataset_path) / "background_clean"
            if bg_clean.is_dir():
                st.success("background_clean/ 폴더 확인됨.")
            else:
                st.warning("background_clean/ 폴더가 없습니다. SAM2 처리된 이미지를 먼저 준비해 주세요.")
        else:
            st.warning("데이터셋 경로를 먼저 설정해 주세요.")

    return bg_method


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
# FR-T2-03: 이미지 크기 (정규화는 ImageNet 고정)
# ---------------------------------------------------------------------------

def _render_image_size_section() -> int:
    """이미지 크기 입력 위젯. 정규화는 ImageNet으로 고정."""
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
    return int(image_size)


def _render_resize_norm_section() -> tuple[int, str, list[float], list[float]]:
    """하위 호환 유지 (configs.yaml 불러오기 등에서 직접 호출되는 경우 대비)."""
    image_size = _render_image_size_section()
    return image_size, "ImageNet", list(_IMAGENET_MEAN), list(_IMAGENET_STD)


# ---------------------------------------------------------------------------
# FR-T2-04: 전처리 전후 미리보기
# ---------------------------------------------------------------------------

def _get_sample_image_path(background_method: str = "none") -> Path | None:
    """train/good/ 알파벳 순 첫 번째 이미지 경로 반환. SAM2이면 background_clean/ 우선."""
    dataset_path = st.session_state.get("dataset_path")
    if not dataset_path:
        return None
    root = Path(dataset_path)
    if background_method == "sam2":
        bg_train_good = root / "background_clean" / "train" / "good"
        if bg_train_good.is_dir():
            images = sorted(
                f for f in bg_train_good.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS
            )
            if images:
                return images[0]
    train_good = root / "train" / "good"
    if not train_good.is_dir():
        return None
    images = sorted(f for f in train_good.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS)
    return images[0] if images else None


def _render_preview(method: str, params: dict | None, image_size: int, background_method: str = "none") -> None:
    st.subheader("전처리 미리보기")
    if image_size % 32 != 0:
        st.warning("이미지 크기가 32의 배수가 아닙니다. 유효한 값을 입력하면 미리보기가 갱신됩니다.")
        return
    sample_path = _get_sample_image_path(background_method)
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


def _make_queue_item(
    preprocessing_config: dict,
    model_config: dict,
    set_id: str | None = None,
) -> dict:
    """
    대기열 항목 dict 생성 (FR-T2-16).
    name 형식: {MODEL_TYPE}_{uuid4().hex[:4]}
    status 초기값: "대기중"
    preprocessing_config / model_config 는 얕은 복사본 저장.
    set_id: 자동 실험 세트 식별자 (배치 자동 생성 시에만 설정)
    """
    model_type = model_config.get("model_type", "model")
    name = f"{model_type.upper()}_{uuid.uuid4().hex[:4]}"
    item: dict = {
        "name":                name,
        "preprocessing_config": dict(preprocessing_config),
        "model_config":         dict(model_config),
        "status":              "대기중",
    }
    if set_id:
        item["set_id"] = set_id
    return item


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
    background_method: str,
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

    col1, col2 = st.columns(2)

    with col1:
        if st.button("설정 저장", type="primary", key="tab2_btn_save", use_container_width=True):
            pre_cfg = _build_preprocessing_config(
                method, params, image_size, norm_label, mean, std, background_method
            )
            mdl_cfg = build_model_config(
                model_type, image_size, batch_size, random_seed,
                threshold_method, threshold_value, model_params,
            )
            st.session_state["preprocessing_config"] = pre_cfg
            st.session_state["model_config"] = mdl_cfg
            st.success("설정이 저장되었습니다.")

    with col2:
        if st.button("📋 대기열에 추가", key="tab2_btn_enqueue", use_container_width=True):
            pre_cfg = _build_preprocessing_config(
                method, params, image_size, norm_label, mean, std, background_method
            )
            mdl_cfg = build_model_config(
                model_type, image_size, batch_size, random_seed,
                threshold_method, threshold_value, model_params,
            )
            item = _make_queue_item(pre_cfg, mdl_cfg)
            q = list(st.session_state.get("experiment_queue", []))
            q.append(item)
            st.session_state["experiment_queue"] = q
            st.success(f"'{item['name']}'이(가) 대기열에 추가되었습니다.")


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

    bg = config.get("background_method", "none")
    st.session_state["tab2_background_label"] = "SAM2 Segmentation" if bg == "sam2" else "없음"

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


# ---------------------------------------------------------------------------
# 자동 실험 설계 — 태그 입력 헬퍼
# ---------------------------------------------------------------------------

def _ae_render_tags(state_key: str, key: str, fmt_fn) -> None:
    """태그 목록을 버튼 형태로 표시. 버튼 클릭 시 해당 태그 제거."""
    tags = st.session_state.get(state_key, [])
    if not tags:
        st.caption("값을 입력하고 추가 버튼을 누르세요.")
        return
    n_cols = min(len(tags), 8)
    cols = st.columns(n_cols)
    to_remove: int | None = None
    for i, v in enumerate(tags):
        with cols[i % n_cols]:
            if st.button(
                f"{fmt_fn(v)} ✕",
                key=f"ae_tag_{key}_{i}",
                use_container_width=True,
                help="클릭하여 제거",
            ):
                to_remove = i
    if to_remove is not None:
        st.session_state[state_key].pop(to_remove)
        st.rerun()


def _ae_tag_int(
    label: str,
    key: str,
    default: int,
    min_val: int,
    max_val: int,
    step: int = 1,
) -> list[int]:
    """정수형 태그 입력 위젯. 현재 태그 목록(list[int]) 반환."""
    state_key = f"ae_tags_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = [default]
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        new_val = int(
            st.number_input(
                label,
                min_value=min_val, max_value=max_val,
                value=default, step=step,
                key=f"ae_in_{key}",
            )
        )
    with col_btn:
        st.write("")
        if st.button("추가", key=f"ae_add_{key}", use_container_width=True):
            if new_val not in st.session_state[state_key]:
                st.session_state[state_key].append(new_val)
                st.session_state[state_key].sort()
                st.rerun()
    _ae_render_tags(state_key, key, str)
    return list(st.session_state[state_key])


def _ae_tag_float(
    label: str,
    key: str,
    default: float,
    min_val: float,
    max_val: float,
    step: float = 0.01,
    fmt: str = "%.4f",
) -> list[float]:
    """실수형 태그 입력 위젯. 현재 태그 목록(list[float]) 반환."""
    state_key = f"ae_tags_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = [default]
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        new_val = float(
            st.number_input(
                label,
                min_value=min_val, max_value=max_val,
                value=default, step=step, format=fmt,
                key=f"ae_in_{key}",
            )
        )
    with col_btn:
        st.write("")
        if st.button("추가", key=f"ae_add_{key}", use_container_width=True):
            if new_val not in st.session_state[state_key]:
                st.session_state[state_key].append(new_val)
                st.session_state[state_key].sort()
                st.rerun()
    _ae_render_tags(state_key, key, lambda v: fmt % v)
    return list(st.session_state[state_key])


# ---------------------------------------------------------------------------
# 자동 실험 설계 — config 빌드 헬퍼
# ---------------------------------------------------------------------------

def _ae_get_filter_params(method: str) -> dict | None:
    """전처리 방식별 파라미터를 탭2 위젯 현재값에서 읽어 반환."""
    if method == "homomorphic":
        return {
            "sigma":     st.session_state.get("tab2_sigma",     10.0),
            "gamma_H":   st.session_state.get("tab2_gamma_h",   1.5),
            "gamma_L":   st.session_state.get("tab2_gamma_l",   0.5),
            "normalize": st.session_state.get("tab2_normalize", True),
        }
    if method == "clahe":
        return {"clip_limit": st.session_state.get("tab2_clip_limit", 2.0)}
    if method == "he":
        return {}
    return None


def _ae_norm_mean(norm_label: str) -> list[float]:
    if norm_label == "ImageNet":
        return list(_IMAGENET_MEAN)
    parsed = _parse_float_list(st.session_state.get("tab2_mean", "0.5,0.5,0.5"))
    return parsed if (parsed and len(parsed) == 3) else [0.5, 0.5, 0.5]


def _ae_norm_std(norm_label: str) -> list[float]:
    if norm_label == "ImageNet":
        return list(_IMAGENET_STD)
    parsed = _parse_float_list(st.session_state.get("tab2_std", "0.5,0.5,0.5"))
    return parsed if (parsed and len(parsed) == 3) else [0.5, 0.5, 0.5]


# ---------------------------------------------------------------------------
# 자동 실험 설계 — 조합 생성
# ---------------------------------------------------------------------------

def _ae_generate_combinations(param_grid: dict) -> list[dict]:
    """param_grid 의 카르테시안 곱으로 모든 조합 생성."""
    if not param_grid or any(not v for v in param_grid.values()):
        return []
    keys = list(param_grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]


def _ae_build_combo_df(combos: list[dict], model_type: str) -> pd.DataFrame:
    """조합 목록을 미리보기 DataFrame 으로 변환."""
    if not combos:
        return pd.DataFrame()
    rows = []
    for c in combos:
        row: dict = {
            "전처리":   _REVERSE_METHOD_MAP.get(c.get("prep_method", "none"), "없음"),
            "크기":     c.get("image_size", ""),
            "정규화":   c.get("norm", ""),
            "배치":     c.get("batch_size", ""),
            "시드":     c.get("random_seed", ""),
            "Th방식":   c.get("threshold_method", ""),
            "Th값":     c.get("threshold_value", ""),
        }
        if model_type == "efficientad":
            row.update({
                "모델크기":   c.get("model_size", ""),
                "단계수":     c.get("train_steps", ""),
                "옵티마이저": c.get("optimizer", ""),
                "LR":         c.get("learning_rate", ""),
                "WD":         c.get("weight_decay", ""),
                "채널":       c.get("out_channels", ""),
                "패딩":       c.get("padding", ""),
                "AE비중":     c.get("ae_loss_weight", ""),
                "Penalty":    c.get("use_imagenet_penalty", ""),
                "스케줄러":   c.get("scheduler", ""),
            })
        else:
            row.update({
                "백본":       c.get("backbone", ""),
                "코어셋":     c.get("coreset_sampling_ratio", ""),
                "커널":       c.get("neighbourhood_kernel_size", ""),
                "max_train":  c.get("max_train", ""),
                "knn":        c.get("knn", ""),
                "top_k":      c.get("top_k_ratio", ""),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _ae_build_queue_item_from_combo(
    combo: dict, model_type: str, set_id: str
) -> dict:
    """단일 파라미터 조합으로 queue item 생성."""
    method = combo["prep_method"]
    pre_cfg = _build_preprocessing_config(
        method=method,
        params=_ae_get_filter_params(method),
        image_size=combo["image_size"],
        norm_label=combo["norm"],
        mean=_ae_norm_mean(combo["norm"]),
        std=_ae_norm_std(combo["norm"]),
    )
    if model_type == "efficientad":
        model_params = build_efficientad_params(
            model_size=combo["model_size"],
            train_steps=int(combo["train_steps"]),
            optimizer=combo["optimizer"],
            learning_rate=float(combo["learning_rate"]),
            weight_decay=float(combo["weight_decay"]),
            out_channels=int(combo["out_channels"]),
            padding=bool(combo["padding"]),
            ae_loss_weight=float(combo["ae_loss_weight"]),
            autoencoder_lr=float(st.session_state.get("tab2_ead_ae_lr", 1e-4)),
            autoencoder_weight_decay=float(st.session_state.get("tab2_ead_ae_wd", 1e-5)),
            lr_decay_epochs=int(st.session_state.get("tab2_ead_decay_ep", 50000)),
            lr_decay_factor=float(st.session_state.get("tab2_ead_decay_f", 0.1)),
            scheduler=combo["scheduler"],
            use_imagenet_penalty=bool(combo["use_imagenet_penalty"]),
            penalty_batch_size=int(st.session_state.get("tab2_ead_pen_bs", 8)),
        )
    else:
        model_params = build_patchcore_params(
            backbone=combo["backbone"],
            pretrained_source="torchvision",
            pretrained_path=None,
            coreset_sampling_ratio=float(combo["coreset_sampling_ratio"]),
            neighbourhood_kernel_size=int(combo["neighbourhood_kernel_size"]),
            max_train=int(combo["max_train"]),
            knn=int(combo["knn"]),
            top_k_ratio=float(combo["top_k_ratio"]),
        )
    mdl_cfg = build_model_config(
        model_type=model_type,
        image_size=int(combo["image_size"]),
        batch_size=int(combo["batch_size"]),
        random_seed=int(combo["random_seed"]),
        threshold_method=combo["threshold_method"],
        threshold_value=float(combo["threshold_value"]),
        params=model_params,
    )
    return _make_queue_item(pre_cfg, mdl_cfg, set_id=set_id)


# ---------------------------------------------------------------------------
# 자동 실험 설계 — 모델별 파라미터 UI
# ---------------------------------------------------------------------------

def _ae_render_efficientad_params() -> dict:
    """EfficientAD 파라미터 멀티셀렉트/태그 위젯. param_grid용 dict 반환."""
    col1, col2 = st.columns(2)
    with col1:
        model_sizes = st.multiselect(
            "모델 크기", ["small", "medium"],
            default=["medium"], key="ae_ead_model_size",
        )
        optimizers = st.multiselect(
            "옵티마이저", ["adam", "adamw", "sgd"],
            default=["adam"], key="ae_ead_optimizer",
        )
    with col2:
        schedulers = st.multiselect(
            "스케줄러", ["StepLR", "CosineAnnealingLR"],
            default=["StepLR"], key="ae_ead_scheduler",
        )
        out_channels_opts = st.multiselect(
            "출력 채널 수", [128, 256, 384, 512],
            default=[384], key="ae_ead_out_channels",
        )

    train_steps_list = _ae_tag_int(
        "학습 단계 수 (train_steps)", "train_steps", 70000, 1000, 200_000, 1000,
    )
    lr_list = _ae_tag_float(
        "학습률 (learning_rate)", "learning_rate", 1e-4, 1e-6, 1e-1, 1e-5, "%.6f",
    )

    with st.expander("추가 EfficientAD 설정"):
        col1, col2 = st.columns(2)
        with col1:
            padding_opts = st.multiselect(
                "패딩 사용", [True, False], default=[False],
                key="ae_ead_padding",
                format_func=lambda x: "사용" if x else "미사용",
            )
            penalty_opts = st.multiselect(
                "ImageNet Penalty", [True, False], default=[False],
                key="ae_ead_use_penalty",
                format_func=lambda x: "사용" if x else "미사용",
            )
        with col2:
            pass
        wd_list = _ae_tag_float(
            "가중치 감쇠 (weight_decay)", "weight_decay", 1e-4, 0.0, 0.1, 1e-5, "%.6f",
        )
        ae_wt_list = _ae_tag_float(
            "AE Loss 비중 (ae_loss_weight)", "ae_loss_weight", 0.5, 0.0, 1.0, 0.05, "%.2f",
        )

    return {
        "model_size":          model_sizes or ["medium"],
        "train_steps":         train_steps_list or [70000],
        "optimizer":           optimizers or ["adam"],
        "learning_rate":       lr_list or [1e-4],
        "weight_decay":        wd_list or [1e-4],
        "out_channels":        out_channels_opts or [384],
        "padding":             padding_opts or [False],
        "ae_loss_weight":      ae_wt_list or [0.5],
        "use_imagenet_penalty": penalty_opts or [False],
        "scheduler":           schedulers or ["StepLR"],
    }


def _ae_render_patchcore_params() -> dict:
    """PatchCore 파라미터 멀티셀렉트/태그 위젯. param_grid용 dict 반환."""
    col1, col2 = st.columns(2)
    with col1:
        backbones = st.multiselect(
            "백본 (backbone)",
            ["wide_resnet50_2", "resnet18", "resnet50"],
            default=["wide_resnet50_2"], key="ae_pc_backbone",
        )
    with col2:
        kernel_sizes = st.multiselect(
            "이웃 커널 크기", [1, 3, 5, 7, 9],
            default=[3], key="ae_pc_kernel",
        )

    coreset_list = _ae_tag_float(
        "코어셋 비율 (coreset_sampling_ratio)", "coreset_ratio",
        0.1, 0.01, 1.0, 0.01, "%.2f",
    )

    with st.expander("추가 PatchCore 설정"):
        max_train_list = _ae_tag_int(
            "최대 학습 샘플 (max_train)", "max_train", 1000, 100, 10_000, 100,
        )
        knn_list = _ae_tag_int("k-NN 이웃 수 (knn)", "knn", 9, 1, 50)
        top_k_list = _ae_tag_float(
            "Top-k 비율 (top_k_ratio)", "top_k_ratio", 0.1, 0.0, 1.0, 0.01, "%.2f",
        )

    return {
        "backbone":                  backbones or ["wide_resnet50_2"],
        "coreset_sampling_ratio":    coreset_list or [0.1],
        "neighbourhood_kernel_size": kernel_sizes or [3],
        "max_train":                 max_train_list or [1000],
        "knn":                       knn_list or [9],
        "top_k_ratio":               top_k_list or [0.1],
    }


# ---------------------------------------------------------------------------
# 자동 실험 설계 — 메인 렌더러
# ---------------------------------------------------------------------------

def _render_auto_experiment_section() -> None:
    """탭2 하단: 파라미터 다중 선택(좌) → 조합 미리보기(우) → 대기열 일괄 추가."""
    st.subheader("🔬 자동 실험 설계")
    st.caption(
        "파라미터 선택지를 여러 개 지정하면 모든 조합으로 실험을 자동 생성합니다. "
        "생성된 조합은 기존 대기열에 일괄 추가됩니다."
    )

    # ── 모델 선택 (단일) — 전폭 ──────────────────────────────────────────────
    ae_model_label: str = st.radio(
        "모델 선택 (단일)",
        ["EfficientAD", "PatchCore"],
        horizontal=True,
        key="ae_model_type",
    )
    model_type = "efficientad" if ae_model_label == "EfficientAD" else "patchcore"

    # ── 3단: 공통 파라미터(좌) + 모델 파라미터(중) + 조합 미리보기(우) ─────
    col_common, col_model_params, col_preview = st.columns(3)

    with col_common:
        st.markdown("#### 공통 파라미터")

        prep_labels: list[str] = st.multiselect(
            "전처리 방식",
            options=["없음", "Homomorphic", "HE", "CLAHE"],
            default=["없음"],
            key="ae_prep_methods",
        )
        if "Homomorphic" in prep_labels or "CLAHE" in prep_labels:
            st.caption(
                "ℹ️ Homomorphic / CLAHE 파라미터는 위 '전처리 영역'의 현재 설정값을 사용합니다."
            )
        prep_methods = [_METHOD_MAP[lbl] for lbl in prep_labels]

        image_sizes = _ae_tag_int("이미지 크기 (image_size)", "image_size", 256, 32, 1024, 32)
        invalid_sizes = [s for s in image_sizes if s % 32 != 0]
        if invalid_sizes:
            st.warning(f"32의 배수가 아닌 크기가 포함되어 있습니다: {invalid_sizes}")

        norm_options: list[str] = st.multiselect(
            "정규화 방식",
            ["ImageNet", "커스텀"],
            default=["ImageNet"],
            key="ae_norm_options",
        )
        if "커스텀" in norm_options:
            st.caption("ℹ️ 커스텀 정규화는 위 '전처리 영역'의 mean / std 값을 사용합니다.")

        with st.expander("추가 공통 설정 (배치 크기 / 시드 / Threshold)"):
            batch_sizes   = _ae_tag_int("배치 크기 (batch_size)", "batch_size", 16, 1, 128)
            random_seeds  = _ae_tag_int("랜덤 시드 (random_seed)", "random_seed", 42, 0, 2_147_483_647)
            threshold_methods: list[str] = st.multiselect(
                "Threshold 방식",
                ["percentile", "absolute"],
                default=["percentile"],
                key="ae_threshold_methods",
            )
            threshold_values = _ae_tag_float(
                "Threshold 값", "threshold_value", 95.0, 0.0, 100.0, 0.5, "%.1f",
            )

    with col_model_params:
        st.markdown(f"#### {ae_model_label} 파라미터")
        if model_type == "efficientad":
            model_grid = _ae_render_efficientad_params()
        else:
            model_grid = _ae_render_patchcore_params()

    with col_preview:
        st.markdown("#### 조합 미리보기")

        # expander가 닫혀 있어도 session_state에 값이 보존됨
        _batch_sizes  = st.session_state.get("ae_tags_batch_size",       [16])
        _rand_seeds   = st.session_state.get("ae_tags_random_seed",      [42])
        _thresh_meths = st.session_state.get("ae_threshold_methods",     ["percentile"])
        _thresh_vals  = st.session_state.get("ae_tags_threshold_value",  [95.0])

        param_grid: dict = {
            "prep_method":      prep_methods   or ["none"],
            "image_size":       image_sizes    or [256],
            "norm":             norm_options   or ["ImageNet"],
            "batch_size":       _batch_sizes   or [16],
            "random_seed":      _rand_seeds    or [42],
            "threshold_method": _thresh_meths  or ["percentile"],
            "threshold_value":  _thresh_vals   or [95.0],
        }
        param_grid.update(model_grid)

        combos = _ae_generate_combinations(param_grid)
        n_combos = len(combos)

        if n_combos == 0:
            st.warning("유효한 조합이 없습니다. 각 파라미터에 1개 이상의 값을 지정해 주세요.")
            return

        if n_combos > 100:
            st.warning(
                f"총 {n_combos}개 조합이 생성됩니다. "
                "선택지를 줄여 조합 수를 낮추는 것을 권장합니다."
            )
        else:
            st.info(f"총 **{n_combos}**개 조합이 생성됩니다.")

        combo_df = _ae_build_combo_df(combos, model_type)
        combo_df.insert(0, "포함", True)

        edited_df = st.data_editor(
            combo_df,
            hide_index=True,
            use_container_width=True,
            key="ae_combo_editor",
            column_config={
                "포함": st.column_config.CheckboxColumn("포함", default=True, width="small"),
            },
        )

        selected_combos = [
            combos[i]
            for i, include in enumerate(edited_df["포함"])
            if include
        ]
        n_selected = len(selected_combos)
        st.caption(f"{n_selected} / {n_combos}개 조합 선택됨")

        if st.button(
            f"📋 선택된 {n_selected}개 조합 대기열에 추가",
            disabled=(n_selected == 0),
            type="primary",
            key="ae_add_to_queue",
            use_container_width=True,
        ):
            set_id = f"SET_{uuid.uuid4().hex[:8].upper()}"
            q = list(st.session_state.get("experiment_queue", []))
            for combo in selected_combos:
                q.append(_ae_build_queue_item_from_combo(combo, model_type, set_id))
            st.session_state["experiment_queue"] = q
            st.success(
                f"세트 [{set_id}]: {n_selected}개 실험이 대기열에 추가되었습니다. "
                "탭3에서 일괄 학습을 시작하세요."
            )
