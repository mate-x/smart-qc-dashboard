import streamlit as st

from utils.messages import MSG


def render() -> None:
    st.header("탭6. 이상 영역 시각화")
    if not _guard():
        return
    _render_ui()
    _handle_events()


def _guard() -> bool:
    if st.session_state.get("selected_experiment_id") is None:
        st.info(MSG["NO_SELECTED_EXP"])
        return False
    return True


def _render_ui() -> None:
    # TODO: 탭6 구현 (11_Tab6_AnomalyMap 참조)
    st.info("탭6 구현 예정입니다.")


def _handle_events() -> None:
    pass
