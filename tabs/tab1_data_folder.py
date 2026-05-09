import streamlit as st

from utils.messages import ERR, MSG


def render() -> None:
    st.header("탭1. 데이터 폴더 구조")
    _render_ui()
    _handle_events()


def _render_ui() -> None:
    st.text_input(
        "데이터셋 경로",
        key="_tab1_dataset_path_input",
        placeholder="예: C:/datasets/mvtec/screw",
        help="MVTec AD 형식의 데이터셋 루트 경로를 입력하세요.",
    )
    st.button("경로 확인", key="_tab1_validate_btn")


def _handle_events() -> None:
    if not st.session_state.get("_tab1_validate_btn"):
        return

    path_input: str = st.session_state.get("_tab1_dataset_path_input", "").strip()
    if not path_input:
        st.error(ERR["ERR_DATASET_NOT_FOUND"])
        return

    # TODO: 탭1 구현 (02_Functional_Requirements / 05_Data_Model 참조)
    st.info("탭1 구현 예정입니다.")
