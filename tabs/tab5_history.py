import streamlit as st

from utils.messages import MSG


def render() -> None:
    st.header("탭5. 실험 히스토리 + 결과 상세 + 모델 저장")
    if not _guard():
        return
    _render_ui()
    _handle_events()


def _guard() -> bool:
    if not st.session_state.get("experiments"):
        st.info(MSG["NO_EXPERIMENTS"])
        return False
    return True


def _render_ui() -> None:
    # TODO: 탭5 구현 (10_Tab5_History 참조)
    st.info("탭5 구현 예정입니다.")


def _handle_events() -> None:
    pass
