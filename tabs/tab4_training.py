import streamlit as st

from utils.messages import MSG


def render() -> None:
    st.header("탭4. 학습 시작 + 학습 로그")
    if not _guard():
        return
    _render_ui()
    _handle_events()


def _guard() -> bool:
    if st.session_state.get("dataset_path") is None:
        st.warning(MSG["NO_DATASET"])
        return False
    if st.session_state.get("preprocessing_config") is None:
        st.warning(MSG["NO_PREPROCESSING"])
        return False
    if st.session_state.get("model_config") is None:
        st.warning(MSG["NO_MODEL_CONFIG"])
        return False
    return True


def _render_ui() -> None:
    st.info(
        "학습 중 새로고침 시 학습 상태를 확인할 수 없습니다.",
        icon="⚠️",
    )
    # TODO: 탭4 구현 (08_AI_ML_Integration / 04_System_Architecture B.5 참조)
    st.info("탭4 구현 예정입니다.")


def _handle_events() -> None:
    pass
