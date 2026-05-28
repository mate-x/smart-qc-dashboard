"""
inspection/tabs/insp_tab2_history.py — 검사 이력 및 통계 탭

FR-INSP-T2-01: 5열 이력 테이블 (seq 역순, 판정 행 색상)
FR-INSP-T2-02: KPI 카드 4개 — 총검사/양품/불량/불량률 (항상 표시)
FR-INSP-T2-03: CSV 내보내기 (비어 있으면 disabled)
FR-INSP-T2-04: 이력 초기화 2단계 확인 → reset_inspection_state()
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st

from inspection.utils.insp_session_init import reset_inspection_state
from utils.logger import log_warning
from utils.messages import INSP_MSG

KST = timezone(timedelta(hours=9))

_CSV_COLUMNS = ["번호", "시각", "이미지명", "판정결과", "Anomaly Score"]
_KEY_MAP: dict[str, str] = {
    "번호":          "seq",
    "시각":          "inspected_at",
    "이미지명":      "image_name",
    "판정결과":      "verdict",
    "Anomaly Score": "anomaly_score",
}
_VERDICT_DISPLAY = {"양품": "🟢 양품", "불량": "🔴 불량"}
_ROW_BG = {"🔴 불량": "#FFDDD6", "🟢 양품": "#D6F5DD"}


def _style_rows(row: pd.Series) -> list[str]:
    color = _ROW_BG.get(str(row["판정결과"]), "")
    return [f"background-color: {color}" if color else "" for _ in row]


def render() -> None:
    st.subheader("검사 이력 및 통계")

    # Guard: 모델 미선택 (FR-INSP-CMN-03 / 15_UI §C.2)
    if st.session_state.get("insp_active_model") is None:
        st.info("검사에 사용할 모델이 선택되지 않았습니다. 탭3에서 모델을 먼저 선택해 주세요.")
        return

    records: list[dict] = st.session_state.get("insp_records", [])

    # 헤더 행: 섹션 제목 + CSV 다운로드 버튼 (15_UI §E.1, E.4)
    col_title, col_csv = st.columns([3, 1])
    with col_title:
        st.markdown("#### 검사 이력 테이블")
    with col_csv:
        filename = f"inspection_history_{datetime.now(tz=KST).strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button(
            label="⬇ CSV 내보내기",
            data=_build_csv(records),
            file_name=filename,
            mime="text/csv",
            disabled=len(records) == 0,
            use_container_width=True,
        )

    # 필터 라디오 (00_Global §5.4)
    filter_opt = st.radio(
        "필터",
        options=["전체", "양품만", "불량만"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # 이력 테이블 (FR-INSP-T2-01)
    df = _build_dataframe(records, filter_opt)
    st.dataframe(
        df.style.apply(_style_rows, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # KPI 카드 — Guard 없음, 기록 없으면 0 표시 (FR-INSP-T2-02)
    _render_kpi(records)

    # 이력 초기화 섹션 (FR-INSP-T2-04)
    st.divider()
    _render_clear_section()


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────


def _build_csv(records: list[dict]) -> bytes:
    """판정결과는 원본 텍스트("양품"/"불량")로 출력 (이모지 없음)."""
    if not records:
        return (",".join(_CSV_COLUMNS) + "\n").encode("utf-8-sig")
    rows = [{col: records[i].get(_KEY_MAP[col], "") for col in _CSV_COLUMNS}
            for i in range(len(records))]
    return pd.DataFrame(rows)[_CSV_COLUMNS].to_csv(index=False).encode("utf-8-sig")


def _build_dataframe(records: list[dict], filter_opt: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=_CSV_COLUMNS)

    rows = []
    for r in records:
        rows.append({
            "번호":          r.get("seq", ""),
            "시각":          r.get("inspected_at", ""),
            "이미지명":      r.get("image_name", ""),
            "판정결과":      _VERDICT_DISPLAY.get(r.get("verdict", ""), r.get("verdict", "")),
            "Anomaly Score": f"{r['anomaly_score']:.4f}" if "anomaly_score" in r else "",
        })

    df = pd.DataFrame(rows, columns=_CSV_COLUMNS)

    if filter_opt == "양품만":
        df = df[df["판정결과"] == "🟢 양품"]
    elif filter_opt == "불량만":
        df = df[df["판정결과"] == "🔴 불량"]

    return df.sort_values("번호", ascending=False).reset_index(drop=True)


def _render_kpi(records: list[dict]) -> None:
    total = len(records)
    good  = sum(1 for r in records if r.get("verdict") == "양품")
    bad   = total - good
    rate  = f"{bad / total * 100:.1f}%" if total > 0 else "-"   # A-20

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 검사", total)
    col2.metric("양품",    good)
    col3.metric("불량",    bad)
    col4.metric("불량률",  rate)


def _render_clear_section() -> None:
    if not st.session_state.get("_tab2_confirm_clear", False):
        if st.button("🗑 이력 초기화", type="secondary"):
            st.session_state["_tab2_confirm_clear"] = True
            st.rerun()
    else:
        st.warning("정말로 검사 이력을 초기화하시겠습니까? 이 작업은 되돌릴 수 없습니다.")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✅ 초기화 확인", type="primary", use_container_width=True):
                st.session_state["_tab2_confirm_clear"] = False
                reset_inspection_state()
                log_warning(
                    "insp_history_cleared",
                    "검사 이력 초기화",
                    tab="insp_tab2",
                )
                st.success(INSP_MSG["HISTORY_CLEARED"])
                st.rerun()
        with col_no:
            if st.button("❌ 취소", type="secondary", use_container_width=True):
                st.session_state["_tab2_confirm_clear"] = False
                st.rerun()
