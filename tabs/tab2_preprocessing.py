import streamlit as st

from utils.messages import MSG


def render() -> None:
    st.header("탭2. 전처리 파라미터 설정")
    if not _guard():
        return
    _render_ui()
    _handle_events()


def _guard() -> bool:
    if st.session_state.get("dataset_path") is None:
        st.warning(MSG["NO_DATASET"])
        return False
    return True


def _render_ui() -> None:
    # TODO: 탭2 구현 (06_Tab2_Preprocessing 참조)
    st.info("탭2 구현 예정입니다.")


def _handle_events() -> None:
    pass
