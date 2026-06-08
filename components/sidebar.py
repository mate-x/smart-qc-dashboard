import streamlit as st


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Smart QC Platform")
        st.markdown("---")

        explorer_type = (
            "primary" if st.session_state.active_dashboard == "explorer" else "secondary"
        )
        inspection_type = (
            "primary" if st.session_state.active_dashboard == "inspection" else "secondary"
        )

        if st.button(
            "🔬 모델 탐색 시스템",
            use_container_width=True,
            type=explorer_type,
            key="sidebar_btn_explorer",
        ):
            st.session_state.active_dashboard = "explorer"
            st.rerun()

        # TODO: [삭제 예정] 비전검사 대시보드가 React 앱으로 이전됨 — 버튼 제거
        if st.button(
            "🏭 비전검사 시스템",
            use_container_width=True,
            type=inspection_type,
            key="sidebar_btn_inspection",
        ):
            st.session_state.active_dashboard = "inspection"
            st.rerun()
