import streamlit as st

from utils.messages import MSG


def render() -> None:
    st.header("탭3. 모델 파라미터 설정")
    if not _guard():
        return
    _render_ui()
    _handle_events()


def _guard() -> bool:
    if st.session_state.get("preprocessing_config") is None:
        st.warning(MSG["NO_PREPROCESSING"])
        return False
    return True


def _render_ui() -> None:
    # TODO: 탭3 구현 (07_Tab3_ModelParams 참조)
    st.info("탭3 구현 예정입니다.")


def _handle_events() -> None:
    pass
