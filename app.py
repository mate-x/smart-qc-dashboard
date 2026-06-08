import streamlit as st

from utils.env_init import ensure_required_dirs
from utils.session_state_init import init_session_state
from components.sidebar import render_sidebar
from tabs import (
    tab1_data_folder,
    tab2_config,
    tab3_training,
    tab4_history,
    tab5_anomaly_map,
)

ensure_required_dirs()

st.set_page_config(
    page_title="Smart QC Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_sidebar()

# TODO: [삭제 예정] 비전검사 대시보드가 React 앱으로 이전됨 — inspection 라우팅 전체 제거
if st.session_state.active_dashboard == "inspection":
    from inspection.inspection_app import render as render_inspection
    render_inspection()
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 탭1. 데이터 폴더",
        "⚙️ 탭2. 전처리 및 모델 설정",
        "🚀 탭3. 학습",
        "📊 탭4. 실험 히스토리",
        "🗺️ 탭5. 이상 영역 시각화",
    ])

    with tab1:
        tab1_data_folder.render()

    with tab2:
        tab2_config.render()

    with tab3:
        tab3_training.render()

    with tab4:
        tab4_history.render()

    with tab5:
        tab5_anomaly_map.render()
