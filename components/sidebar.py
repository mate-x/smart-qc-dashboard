import streamlit as st


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Smart QC Dashboard")
        st.markdown("---")

        _render_dataset_section()
        _render_device_section()
        _render_current_config_section()


def _render_dataset_section() -> None:
    dataset_meta = st.session_state.get("dataset_meta")
    if dataset_meta is None:
        return

    st.subheader("데이터셋")
    st.caption(st.session_state.get("dataset_path", ""))
    cols = st.columns(2)
    cols[0].metric("학습 (good)", dataset_meta.get("train_good_count", "-"))
    cols[1].metric("테스트 전체", dataset_meta.get("total_test_count", "-"))

    defect_classes = dataset_meta.get("defect_classes", [])
    if defect_classes:
        st.caption(f"결함 클래스: {', '.join(defect_classes)}")


def _render_device_section() -> None:
    device_info = st.session_state.get("device_info")
    if device_info is None:
        return

    st.markdown("---")
    st.subheader("디바이스")
    device = device_info.get("device", "cpu")
    if device == "cuda":
        gpu_name = device_info.get("gpu_name", "Unknown GPU")
        vram_gb = device_info.get("vram_gb", 0.0)
        st.success(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
    else:
        st.info("CPU 모드")


def _render_current_config_section() -> None:
    model_config = st.session_state.get("model_config")
    if model_config is None:
        return

    st.markdown("---")
    st.subheader("현재 설정")
    st.caption(f"모델: {model_config.get('model_type', '-')}")
    st.caption(f"이미지 크기: {model_config.get('image_size', '-')}px")
    st.caption(f"Threshold: {model_config.get('threshold_method', '-')} / {model_config.get('threshold_value', '-')}")

    run_status = st.session_state.get("current_run_status", "idle")
    if run_status == "running":
        st.warning("학습 진행 중...")
