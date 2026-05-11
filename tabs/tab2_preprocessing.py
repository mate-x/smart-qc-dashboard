from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from utils.config_manager import get_preprocessing_config, save_config_section
from utils.image_utils import SUPPORTED_FORMATS, apply_filter, resize_with_padding
from utils.messages import MSG

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


def render() -> None:
    st.header("탭2. 전처리 파라미터 설정")
    if not _guard():
        return

    method, params = _render_method_section()
    image_size, norm_label, mean, std = _render_resize_norm_section()
    _render_preview(method, params, image_size)
    _render_save_buttons(method, params, image_size, norm_label, mean, std)


def _guard() -> bool:
    if st.session_state.get("dataset_path") is None:
        st.warning(MSG["NO_DATASET"])
        return False
    return True


# ---------------------------------------------------------------------------
# FR-T2-01 + FR-T2-02: 전처리 방식 선택 및 파라미터 UI
# ---------------------------------------------------------------------------

def _render_method_section() -> tuple[str, dict | None]:
    label: str = st.radio(
        "전처리 방식 (Preprocessing Method)",
        options=_METHOD_LABELS,
        horizontal=True,
        key="t2_method_label",
    )
    method = _METHOD_MAP[label]
    params: dict | None = None

    if method == "homomorphic":
        sigma = st.slider("sigma", 0.1, 50.0, 10.0, 0.1, key="t2_sigma")
        gamma_h = st.slider("gamma_H", 1.0, 3.0, 1.5, 0.05, key="t2_gamma_h")
        gamma_l = st.slider("gamma_L", 0.1, 1.0, 0.5, 0.05, key="t2_gamma_l")
        normalize = st.checkbox("정규화 적용 (normalize)", value=True, key="t2_normalize")
        params = {
            "sigma": sigma,
            "gamma_H": gamma_h,
            "gamma_L": gamma_l,
            "normalize": normalize,
        }
    elif method == "clahe":
        clip_limit = st.slider("클립 한계 (clipLimit)", 0.1, 40.0, 2.0, 0.1, key="t2_clip_limit")
        params = {"clip_limit": clip_limit}
    elif method == "he":
        st.info("히스토그램 평탄화(HE)는 파라미터가 없습니다.")
        params = {}

    return method, params


# ---------------------------------------------------------------------------
# FR-T2-03: 이미지 크기 및 정규화 설정
# ---------------------------------------------------------------------------

def _render_resize_norm_section() -> tuple[int, str, list[float], list[float]]:
    st.subheader("이미지 크기 및 정규화")

    image_size = st.number_input(
        "이미지 크기 (image_size)",
        min_value=32,
        max_value=1024,
        value=256,
        step=32,
        key="t2_image_size",
    )
    if int(image_size) % 32 != 0:
        st.error("32의 배수만 입력 가능합니다.")

    norm_label: str = st.radio(
        "정규화 방식 (Normalization)",
        ["ImageNet", "커스텀"],
        horizontal=True,
        key="t2_norm_label",
    )

    if norm_label == "커스텀":
        mean_str = st.text_input(
            "mean (쉼표 구분, 예: 0.5,0.5,0.5)",
            value="0.5,0.5,0.5",
            key="t2_mean",
        )
        std_str = st.text_input(
            "std (쉼표 구분, 예: 0.5,0.5,0.5)",
            value="0.5,0.5,0.5",
            key="t2_std",
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
# 순수 함수: preprocessing_config 스키마 (§1.6) 구성
# ---------------------------------------------------------------------------

def _build_config(
    method: str,
    params: dict | None,
    image_size: int,
    norm_label: str,
    mean: list[float],
    std: list[float],
) -> dict:
    return {
        "method": method,
        "resize_mode": "padding",
        "image_size": image_size,
        "normalization": "imagenet" if norm_label == "ImageNet" else "custom",
        "mean": list(mean),   # 호출자 리스트 변경으로부터 독립
        "std": list(std),
        "params": params if params else None,
    }


# ---------------------------------------------------------------------------
# 불러오기: 위젯 세션 키에 config 값 반영
# ---------------------------------------------------------------------------

def _apply_config_to_widgets(config: dict) -> None:
    """configs.yaml에서 불러온 config를 위젯 키에 반영 (다음 rerun 시 UI 반영).

    슬라이더 범위를 벗어나는 값은 클램프, image_size는 32 배수로 snap.
    """
    method = config.get("method", "none")
    st.session_state["t2_method_label"] = _REVERSE_METHOD_MAP.get(method, "없음")

    params = config.get("params") or {}
    if method == "homomorphic":
        st.session_state["t2_sigma"]     = max(0.1,  min(50.0, float(params.get("sigma",   10.0))))
        st.session_state["t2_gamma_h"]   = max(1.0,  min(3.0,  float(params.get("gamma_H", 1.5))))
        st.session_state["t2_gamma_l"]   = max(0.1,  min(1.0,  float(params.get("gamma_L", 0.5))))
        st.session_state["t2_normalize"] = bool(params.get("normalize", True))
    elif method == "clahe":
        st.session_state["t2_clip_limit"] = max(0.1, min(40.0, float(params.get("clip_limit", 2.0))))

    st.session_state["t2_image_size"] = _snap_image_size(int(config.get("image_size", 256)))

    norm = config.get("normalization", "imagenet")
    st.session_state["t2_norm_label"] = "ImageNet" if norm == "imagenet" else "커스텀"

    if norm != "imagenet":
        mean = config.get("mean", [0.5, 0.5, 0.5])
        std  = config.get("std",  [0.5, 0.5, 0.5])
        if not isinstance(mean, list) or len(mean) != 3:
            mean = [0.5, 0.5, 0.5]
        if not isinstance(std, list) or len(std) != 3:
            std = [0.5, 0.5, 0.5]
        st.session_state["t2_mean"] = ",".join(str(v) for v in mean)
        st.session_state["t2_std"]  = ",".join(str(v) for v in std)


# ---------------------------------------------------------------------------
# FR-T2-05: 설정 저장 버튼
# ---------------------------------------------------------------------------

def _render_save_buttons(
    method: str,
    params: dict | None,
    image_size: int,
    norm_label: str,
    mean: list[float],
    std: list[float],
) -> None:
    st.subheader("설정 저장")

    if image_size % 32 != 0:
        st.error("이미지 크기가 32의 배수가 아닙니다. 수정 후 저장해 주세요.")
        return

    col1, col2, col3 = st.columns(3)

    if col1.button("전처리 설정 저장", type="primary", key="t2_save_session"):
        config = _build_config(method, params, image_size, norm_label, mean, std)
        st.session_state["preprocessing_config"] = config
        st.success("전처리 설정이 저장되었습니다.")

    if col2.button("configs.yaml 저장", key="t2_save_yaml"):
        config = _build_config(method, params, image_size, norm_label, mean, std)
        try:
            save_config_section("preprocessing", config)
            st.success("configs.yaml에 저장되었습니다.")
        except RuntimeError as e:
            st.error(str(e))

    if col3.button("configs.yaml 불러오기", key="t2_load_yaml"):
        loaded = get_preprocessing_config()
        if loaded is None:
            st.warning("configs.yaml에 전처리 설정이 없습니다.")
        else:
            _apply_config_to_widgets(loaded)
            st.session_state["preprocessing_config"] = loaded
            st.success("configs.yaml에서 전처리 설정을 불러왔습니다.")
            st.rerun()
