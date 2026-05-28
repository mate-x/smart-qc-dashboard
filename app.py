import streamlit as st

from utils.env_init import ensure_required_dirs
from utils.session_state_init import init_session_state
from components.sidebar import render_sidebar
from tabs import (
    tab1_data_folder,
    tab2_preprocessing,
    tab3_model_params,
    tab4_training,
    tab5_history,
    tab6_anomaly_map,
)

ensure_required_dirs()

st.set_page_config(
    page_title="Smart QC Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()
render_sidebar()

if st.session_state.active_dashboard == "inspection":
    from inspection.inspection_app import render as render_inspection
    render_inspection()
else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 탭1. 데이터 폴더",
        "⚙️ 탭2. 전처리 설정",
        "🧠 탭3. 모델 파라미터",
        "🚀 탭4. 학습",
        "📊 탭5. 실험 히스토리",
        "🗺️ 탭6. 이상 영역 시각화",
    ])

    with tab1:
        tab1_data_folder.render()

    with tab2:
        tab2_preprocessing.render()

    with tab3:
        tab3_model_params.render()

    with tab4:
        tab4_training.render()

    with tab5:
        tab5_history.render()

    with tab6:
        tab6_anomaly_map.render()
