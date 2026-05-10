"""
탭3 — 모델 파라미터 설정

FR-T3-01 ~ FR-T3-10 구현.
00_Global_Context 1.7절 model_config, 1.4절 model_params 스키마 준수.
"""

from __future__ import annotations

import streamlit as st

from utils.config_manager import ConfigLoadError, load_config, save_config_section
from utils.messages import MSG
from utils.storage import validate_imagenet_penalty_dir


# ─────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────

def render() -> None:
    st.header("탭3. 모델 파라미터 설정")
    _guard()
    _detect_device_once()
    _render_device_info()
    _render_body()


# ─────────────────────────────────────────────
# Guard / 디바이스 감지
# ─────────────────────────────────────────────

def _guard() -> None:
    """FR-CMN-03 / 06_API_Spec §7.1: preprocessing_config 없으면 st.stop()."""
    if st.session_state.get("preprocessing_config") is None:
        st.warning(MSG["NO_PREPROCESSING"])
        st.stop()


def _detect_device_once() -> None:
    """FR-T3-07: 탭3 최초 진입 시 1회만 디바이스 감지 (idempotent)."""
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


# ─────────────────────────────────────────────
# 메인 렌더링 흐름
# ─────────────────────────────────────────────

def _render_body() -> None:
    preprocessing_config: dict = st.session_state["preprocessing_config"]
    default_image_size: int = int(preprocessing_config.get("image_size", 256))

    # FR-T3-09 (S): 저장된 model_config와 preprocessing image_size 불일치 경고
    saved_cfg = st.session_state.get("model_config")
    if saved_cfg and int(saved_cfg.get("image_size", default_image_size)) != default_image_size:
        st.warning(
            f"image_size가 전처리 설정({default_image_size}px)과 다릅니다. 동기화를 권장합니다."
        )

    # FR-T3-01: 모델 라디오 (R-UI-02: 비선택 UI 미렌더링)
    model_label: str = st.radio(
        "모델 선택 (Model Type)",
        options=["EfficientAD", "PatchCore"],
        horizontal=True,
        key="tab3_model_label",
    )
    model_type = "efficientad" if model_label == "EfficientAD" else "patchcore"

    st.divider()

    # FR-T3-02: 공통 파라미터
    st.subheader("공통 설정")
    col1, col2, col3 = st.columns(3)
    with col1:
        image_size = int(
            st.number_input(
                "이미지 크기 (image_size)",
                min_value=32, max_value=1024,
                value=default_image_size, step=32,
                key="tab3_image_size",
            )
        )
        if image_size % 32 != 0:
            st.error("32의 배수만 입력 가능합니다.")
    with col2:
        batch_size = int(
            st.number_input(
                "배치 크기 (batch_size)",
                min_value=1, max_value=128,
                value=16, step=1,
                key="tab3_batch_size",
            )
        )
    with col3:
        random_seed = int(
            st.number_input(
                "랜덤 시드 (random_seed)",
                min_value=0, max_value=2_147_483_647,
                value=42, step=1,
                key="tab3_random_seed",
            )
        )

    st.divider()

    # 모델별 파라미터 (R-UI-02: 선택된 모델만 렌더링)
    if model_type == "efficientad":
        params = _render_efficientad_params()
    else:
        params = _render_patchcore_params()

    st.divider()

    # FR-T3-07: Threshold 설정
    st.subheader("Threshold 설정")
    threshold_label: str = st.radio(
        "Threshold 방식",
        options=["Percentile (백분위)", "Absolute (절대값)"],
        horizontal=True,
        key="tab3_threshold_label",
    )
    threshold_method = "percentile" if "Percentile" in threshold_label else "absolute"

    if threshold_method == "percentile":
        threshold_value = float(
            st.slider("백분위 값", 0.0, 100.0, 95.0, 0.5, key="tab3_threshold_pct")
        )
    else:
        threshold_value = float(
            st.slider("절대값", 0.0, 1.0, 0.5, 0.01, key="tab3_threshold_abs")
        )

    # FR-T3-10 (S): 정상/결함 비율 실시간 표시
    _render_threshold_ratio_preview(threshold_method, threshold_value)

    st.divider()

    # FR-T3-08 + (S) FR-T3-09 저장 버튼 영역
    _render_save_area(
        model_type=model_type,
        image_size=image_size,
        batch_size=batch_size,
        random_seed=random_seed,
        threshold_method=threshold_method,
        threshold_value=threshold_value,
        params=params,
    )


# ─────────────────────────────────────────────
# EfficientAD 파라미터 UI  (FR-T3-03, FR-T3-04)
# ─────────────────────────────────────────────

def _render_efficientad_params() -> dict:
    """EfficientAD 전용 파라미터 UI. 반환값은 model_config.params 오브젝트."""
    st.subheader("EfficientAD 파라미터")

    col1, col2 = st.columns(2)
    with col1:
        model_size: str = st.radio(
            "모델 크기 (model_size)", ["small", "medium"],
            index=1, horizontal=True, key="ead_model_size",
        )
        train_steps = int(
            st.number_input(
                "학습 단계 수 (train_steps)", 1000, 200_000, 70_000, 1000,
                key="ead_train_steps",
            )
        )
        optimizer: str = st.selectbox(
            "옵티마이저 (optimizer)", ["adam", "adamw", "sgd"],
            index=0, key="ead_optimizer",
        )
        out_channels = int(
            st.selectbox(
                "출력 채널 수 (out_channels)", [128, 256, 384, 512],
                index=2, key="ead_out_channels",
            )
        )

    with col2:
        learning_rate = float(
            st.number_input(
                "학습률 (learning_rate)", 1e-6, 1e-1, 1e-4,
                format="%.6f", key="ead_lr",
            )
        )
        weight_decay = float(
            st.number_input(
                "가중치 감쇠 (weight_decay)", 0.0, 0.1, 1e-4,
                format="%.6f", key="ead_wd",
            )
        )
        padding = bool(st.checkbox("패딩 사용 (padding)", value=False, key="ead_padding"))

    # FR-T3-03: ae/st loss weight 자동 보정 (R-03)
    st.markdown("**AE / ST Loss 비중** (합산 1.0 자동 보정)")
    ae_loss_weight = float(
        st.slider("AE Loss 비중 (ae_loss_weight)", 0.0, 1.0, 0.5, 0.01, key="ead_ae_weight")
    )
    st_loss_weight = compute_st_loss_weight(ae_loss_weight)

    col_ae, col_st = st.columns(2)
    with col_ae:
        st.metric("ae_loss_weight", f"{ae_loss_weight:.2f}")
    with col_st:
        st.metric("st_loss_weight", f"{st_loss_weight:.2f}")

    # FR-T3-04: 고급 설정 expander
    with st.expander("고급 설정 (Advanced Settings)"):
        adv1, adv2 = st.columns(2)
        with adv1:
            autoencoder_lr = float(
                st.number_input(
                    "AE 학습률 (autoencoder_lr)", 1e-6, 1e-1, 1e-4,
                    format="%.6f", key="ead_ae_lr",
                )
            )
            autoencoder_wd = float(
                st.number_input(
                    "AE 가중치 감쇠 (autoencoder_weight_decay)", 0.0, 0.1, 1e-5,
                    format="%.6f", key="ead_ae_wd",
                )
            )
            lr_decay_epochs = int(
                st.number_input(
                    "LR 감쇠 에포크 (lr_decay_epochs)", 1000, 200_000, 50_000, 1000,
                    key="ead_decay_ep",
                )
            )
        with adv2:
            lr_decay_factor = float(
                st.slider("LR 감쇠 계수 (lr_decay_factor)", 0.01, 1.0, 0.1, 0.01,
                          key="ead_decay_f")
            )
            scheduler: str = st.selectbox(
                "스케줄러 (scheduler)", ["StepLR", "CosineAnnealingLR"],
                index=0, key="ead_sched",
            )
            imagenet_pw = float(
                st.slider(
                    "ImageNet 패널티 가중치 (imagenet_penalty_weight)",
                    0.0, 10.0, 1.0, 0.1, key="ead_img_pw",
                )
            )
            penalty_bs = int(
                st.number_input(
                    "패널티 배치 크기 (penalty_batch_size)", 1, 64, 8, 1,
                    key="ead_pen_bs",
                )
            )

        # §Z.1: imagenet_penalty_weight > 0 시 디렉터리 검증 피드백
        if imagenet_pw > 0:
            try:
                validate_imagenet_penalty_dir()
                st.caption("ImageNet 패널티 디렉터리: 이미지 확인됨")
            except ValueError as e:
                st.warning(
                    f"{e} — imagenet_penalty_weight > 0이면 EfficientAD 학습이 실패합니다. "
                    "이미지를 추가하거나 weight를 0으로 설정하세요."
                )

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
        imagenet_penalty_weight=imagenet_pw,
        penalty_batch_size=penalty_bs,
    )


# ─────────────────────────────────────────────
# PatchCore 파라미터 UI  (FR-T3-05, FR-T3-06)
# ─────────────────────────────────────────────

def _render_patchcore_params() -> dict:
    """PatchCore 전용 파라미터 UI. 반환값은 model_config.params 오브젝트."""
    st.subheader("PatchCore 파라미터")

    col1, col2 = st.columns(2)
    with col1:
        backbone: str = st.selectbox(
            "백본 (backbone)",
            ["wide_resnet50_2", "resnet18", "resnet50"],
            index=0, key="pc_backbone",
        )
        pretrained_label: str = st.radio(
            "사전학습 가중치 출처 (pretrained_source)",
            ["torchvision", "로컬 경로"],
            horizontal=True, key="pc_pretrained_label",
        )
        pretrained_source = "torchvision" if pretrained_label == "torchvision" else "local"

        pretrained_path: str | None = None
        if pretrained_source == "local":
            pretrained_path = (
                st.text_input(
                    "로컬 가중치 경로 (pretrained_path)",
                    key="pc_pretrained_path",
                ) or None
            )

    with col2:
        coreset_sampling_ratio = float(
            st.slider(
                "코어셋 비율 (coreset_sampling_ratio)",
                0.01, 1.0, 0.1, 0.01, key="pc_coreset",
            )
        )
        neighbourhood_kernel_size = int(
            st.select_slider(
                "이웃 커널 크기 (neighbourhood_kernel_size)",
                options=[1, 3, 5, 7, 9],
                value=3, key="pc_kernel",
            )
        )

    # FR-T3-06: PatchCore 고급 설정 expander
    with st.expander("고급 설정 (Advanced Settings)"):
        adv1, adv2 = st.columns(2)
        with adv1:
            max_train = int(
                st.number_input(
                    "최대 학습 샘플 수 (max_train)", 100, 10_000, 1000, 100,
                    key="pc_max_train",
                )
            )
        with adv2:
            knn = int(
                st.number_input(
                    "k-NN 이웃 수 (knn)", 1, 50, 9, 1,
                    key="pc_knn",
                )
            )
            top_k_ratio = float(
                st.slider("Top-k 비율 (top_k_ratio)", 0.0, 1.0, 0.1, 0.01, key="pc_top_k")
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


# ─────────────────────────────────────────────
# FR-T3-10 (S): 정상/결함 비율 실시간 표시
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# 저장 버튼 영역  (FR-T3-08)
# ─────────────────────────────────────────────

def _render_save_area(
    model_type: str,
    image_size: int,
    batch_size: int,
    random_seed: int,
    threshold_method: str,
    threshold_value: float,
    params: dict,
) -> None:
    col_save, col_yaml, col_load = st.columns(3)

    with col_save:
        if st.button("모델 설정 저장", type="primary", key="tab3_btn_save"):
            if image_size % 32 != 0:
                st.error("32의 배수 오류를 먼저 수정해 주세요.")
                return
            cfg = {
                "model_type": model_type,
                "image_size": image_size,
                "batch_size": batch_size,
                "random_seed": random_seed,
                "threshold_method": threshold_method,
                "threshold_value": threshold_value,
                "params": params,
            }
            st.session_state["model_config"] = cfg
            st.success("모델 설정이 저장되었습니다.")

    with col_yaml:
        if st.button("configs.yaml 저장", key="tab3_btn_yaml_save"):
            cfg = st.session_state.get("model_config")
            if cfg is None:
                st.warning("먼저 [모델 설정 저장]을 클릭해 주세요.")
            else:
                save_config_section("model", cfg)
                st.success("configs.yaml model 섹션이 저장되었습니다.")

    with col_load:
        if st.button("configs.yaml 불러오기", key="tab3_btn_yaml_load"):
            st.session_state["_tab3_show_load"] = True

    if st.session_state.get("_tab3_show_load"):
        _render_load_ui()


def _render_load_ui() -> None:
    """UC-09: configs.yaml에서 model 섹션 불러와 위젯에 반영."""
    yaml_path: str = st.text_input(
        "configs.yaml 경로", value="./configs.yaml", key="tab3_load_path"
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("확인", key="tab3_btn_load_confirm"):
            try:
                raw = load_config(yaml_path)
            except ConfigLoadError as e:
                st.error(f"불러오기 실패: {e}")
                return
            model_section = raw.get("model")
            if model_section is None:
                st.error("configs.yaml에 model 섹션이 없습니다.")
                return
            _apply_model_config_to_widgets(model_section)
            st.session_state["model_config"] = model_section
            st.session_state["_tab3_show_load"] = False
            st.success("model 설정을 불러왔습니다.")
            st.rerun()
    with c2:
        if st.button("취소", key="tab3_btn_load_cancel"):
            st.session_state["_tab3_show_load"] = False
            st.rerun()


# ─────────────────────────────────────────────
# 위젯 프리필 (configs.yaml 불러오기 후 rerun 전 적용)
# ─────────────────────────────────────────────

def _apply_model_config_to_widgets(cfg: dict) -> None:
    """저장된 model_config 값을 위젯 key에 주입 → 다음 rerun에 반영."""
    model_type = cfg.get("model_type", "patchcore")
    st.session_state["tab3_model_label"] = (
        "EfficientAD" if model_type == "efficientad" else "PatchCore"
    )
    if cfg.get("image_size"):
        st.session_state["tab3_image_size"] = int(cfg["image_size"])
    if cfg.get("batch_size"):
        st.session_state["tab3_batch_size"] = int(cfg["batch_size"])
    if cfg.get("random_seed") is not None:
        st.session_state["tab3_random_seed"] = int(cfg["random_seed"])

    threshold_method = cfg.get("threshold_method", "percentile")
    st.session_state["tab3_threshold_label"] = (
        "Percentile (백분위)" if threshold_method == "percentile" else "Absolute (절대값)"
    )
    threshold_value = cfg.get("threshold_value", 95.0)
    if threshold_method == "percentile":
        st.session_state["tab3_threshold_pct"] = float(threshold_value)
    else:
        st.session_state["tab3_threshold_abs"] = float(threshold_value)

    params = cfg.get("params", {})
    if model_type == "patchcore":
        _apply_patchcore_widgets(params)
    elif model_type == "efficientad":
        _apply_efficientad_widgets(params)


def _apply_patchcore_widgets(params: dict) -> None:
    if params.get("backbone"):
        st.session_state["pc_backbone"] = params["backbone"]
    pretrained_source = params.get("pretrained_source", "torchvision")
    st.session_state["pc_pretrained_label"] = (
        "torchvision" if pretrained_source == "torchvision" else "로컬 경로"
    )
    if params.get("pretrained_path"):
        st.session_state["pc_pretrained_path"] = params["pretrained_path"]
    if params.get("coreset_sampling_ratio") is not None:
        st.session_state["pc_coreset"] = float(params["coreset_sampling_ratio"])
    if params.get("neighbourhood_kernel_size") is not None:
        val = int(params["neighbourhood_kernel_size"])
        # select_slider 값은 options 목록 내여야 함
        if val in (1, 3, 5, 7, 9):
            st.session_state["pc_kernel"] = val
    if params.get("max_train") is not None:
        st.session_state["pc_max_train"] = int(params["max_train"])
    if params.get("knn") is not None:
        st.session_state["pc_knn"] = int(params["knn"])
    if params.get("top_k_ratio") is not None:
        st.session_state["pc_top_k"] = float(params["top_k_ratio"])


def _apply_efficientad_widgets(params: dict) -> None:
    if params.get("model_size"):
        st.session_state["ead_model_size"] = params["model_size"]
    if params.get("train_steps") is not None:
        st.session_state["ead_train_steps"] = int(params["train_steps"])
    if params.get("optimizer"):
        st.session_state["ead_optimizer"] = params["optimizer"]
    if params.get("learning_rate") is not None:
        st.session_state["ead_lr"] = float(params["learning_rate"])
    if params.get("weight_decay") is not None:
        st.session_state["ead_wd"] = float(params["weight_decay"])
    if params.get("out_channels") is not None:
        st.session_state["ead_out_channels"] = int(params["out_channels"])
    if params.get("padding") is not None:
        st.session_state["ead_padding"] = bool(params["padding"])
    if params.get("ae_loss_weight") is not None:
        st.session_state["ead_ae_weight"] = float(params["ae_loss_weight"])
    if params.get("autoencoder_lr") is not None:
        st.session_state["ead_ae_lr"] = float(params["autoencoder_lr"])
    if params.get("autoencoder_weight_decay") is not None:
        st.session_state["ead_ae_wd"] = float(params["autoencoder_weight_decay"])
    if params.get("lr_decay_epochs") is not None:
        st.session_state["ead_decay_ep"] = int(params["lr_decay_epochs"])
    if params.get("lr_decay_factor") is not None:
        st.session_state["ead_decay_f"] = float(params["lr_decay_factor"])
    if params.get("scheduler"):
        st.session_state["ead_sched"] = params["scheduler"]
    if params.get("imagenet_penalty_weight") is not None:
        st.session_state["ead_img_pw"] = float(params["imagenet_penalty_weight"])
    if params.get("penalty_batch_size") is not None:
        st.session_state["ead_pen_bs"] = int(params["penalty_batch_size"])


# ─────────────────────────────────────────────
# 순수 함수 (테스트 가능)
# ─────────────────────────────────────────────

def compute_threshold_ratio(
    threshold_method: str,
    threshold_value: float,
) -> tuple[float | None, float | None]:
    """
    FR-T3-10: Threshold 기준 정상/결함 비율 근사치 계산.

    percentile 방식: 정상 비율 = threshold_value / 100 (근사치)
    absolute 방식: 계산 불가 → (None, None) 반환
    """
    if threshold_method == "percentile":
        normal_ratio = round(threshold_value / 100.0, 6)
        defect_ratio = round(1.0 - normal_ratio, 6)
        return normal_ratio, defect_ratio
    return None, None


def compute_st_loss_weight(ae_loss_weight: float) -> float:
    """R-03: st_loss_weight = round(1.0 - ae_loss_weight, 6)."""
    return round(1.0 - ae_loss_weight, 6)


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
    imagenet_penalty_weight: float,
    penalty_batch_size: int,
) -> dict:
    """00_Global_Context 1.4절 EfficientAD model_params 오브젝트 생성."""
    st_loss_weight = compute_st_loss_weight(ae_loss_weight)
    return {
        "model_size": model_size,
        "train_steps": int(train_steps),
        "optimizer": optimizer,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "out_channels": int(out_channels),
        "padding": bool(padding),
        "ae_loss_weight": float(ae_loss_weight),
        "st_loss_weight": st_loss_weight,
        "autoencoder_lr": float(autoencoder_lr),
        "autoencoder_weight_decay": float(autoencoder_weight_decay),
        "lr_decay_epochs": int(lr_decay_epochs),
        "lr_decay_factor": float(lr_decay_factor),
        "scheduler": scheduler,
        "imagenet_penalty_weight": float(imagenet_penalty_weight),
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
