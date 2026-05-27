import streamlit as st

from inspection.tabs import insp_tab1_realtime, insp_tab2_history, insp_tab3_model


def render() -> None:
    tab1, tab2, tab3 = st.tabs([
        "🔍 탭1. 실시간 검사",
        "📋 탭2. 검사 이력 및 통계",
        "🔄 탭3. 딥러닝 모델 교체",
    ])

    with tab1:
        insp_tab1_realtime.render()

    with tab2:
        insp_tab2_history.render()

    with tab3:
        insp_tab3_model.render()
