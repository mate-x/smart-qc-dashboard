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
import plotly.graph_objects as go
import streamlit as st

from inspection.utils.insp_session_init import (
    reset_inspection_state,
    get_sim_group_count,
    get_sim_all_group_labels,
    get_sim_group_label,
    get_sim_group_index,
)
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

    # 실시간 통계 차트 3분할 (FR-INSP-T2-05~07)
    _render_chart_section()

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


# ── 실시간 통계 차트 (FR-INSP-T2-05) ─────────────────────────────────────────

def _get_group_records(
    records: list[dict], group_idx: int, unit: int
) -> list[dict]:
    """
    seq 번호 기준으로 특정 그룹에 속하는 검사 기록을 필터링한다 (FR-INSP-T2-06).
    그룹 group_idx(0-indexed): seq (group_idx*unit + 1) ~ (group_idx + 1)*unit
    """
    return [
        r for r in records
        if get_sim_group_index(r.get("seq", 0), unit) == group_idx
    ]


def _build_histogram_fig(
    group_records: list[dict],
    threshold: float,
    group_label: str,
    unit: int,
) -> go.Figure:
    """
    FR-INSP-T2-06: Anomaly Score 히스토그램 Plotly 피겨 생성.

    파란색 bar: 정상 판정 (anomaly_score < threshold)
    빨간색 bar: 불량 판정 (anomaly_score >= threshold)
    x축: 0~1 고정 / y축: 동적 (기본 0~10, 데이터에 따라 확장)
    threshold 수직 점선 표시
    """
    normal_scores = [
        r["anomaly_score"] for r in group_records
        if r.get("anomaly_score", 0.0) < threshold
    ]
    defect_scores = [
        r["anomaly_score"] for r in group_records
        if r.get("anomaly_score", 0.0) >= threshold
    ]

    fig = go.Figure()

    _bin_cfg = dict(start=0.0, end=1.001, size=0.05)  # 20 bins, 0~1 fixed

    if normal_scores:
        fig.add_trace(go.Histogram(
            x=normal_scores,
            name="정상",
            marker_color="#4e9af1",
            opacity=0.75,
            xbins=_bin_cfg,
        ))

    if defect_scores:
        fig.add_trace(go.Histogram(
            x=defect_scores,
            name="불량",
            marker_color="#e05555",
            opacity=0.75,
            xbins=_bin_cfg,
        ))

    # threshold 수직 점선
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"thr={threshold:.3f}",
        annotation_position="top right",
    )

    # y축 기본 0~10, 데이터에 따라 동적 확장
    max_count = max(len(normal_scores), len(defect_scores), 0)
    y_max = max(10, max_count + 2)

    fig.update_layout(
        title=f"Anomaly Score 분포 — {group_label}",
        xaxis=dict(range=[-0.2, 1.2], title="Anomaly Score"),  # #02: 여유 공간 확보
        yaxis=dict(range=[0, y_max], title="검사 수"),
        barmode="overlay",
        height=300,
        margin=dict(t=50, b=40, l=45, r=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def _build_group_table_df(total_seqs: int, unit: int) -> pd.DataFrame:
    """
    단위 크기 기준 시간 범위 레이블 DataFrame 반환 (FR-INSP-T2-05).
    컬럼: '시간 범위' (단일 컬럼).
    빈 기록이면 빈 DataFrame (컬럼만 존재).
    """
    labels = get_sim_all_group_labels(total_seqs, unit)
    if not labels:
        return pd.DataFrame(columns=["시간 범위"])
    return pd.DataFrame({"시간 범위": labels})


def _render_unit_buttons(current_unit: int) -> None:
    """
    단위 선택 버튼 3개 렌더링 (FR-INSP-T2-05).
    col_left 안에서 호출되므로 3분할 왼쪽 열 너비에 자동 맞춰짐.
    """
    b20, b40, b100 = st.columns(3)
    with b20:
        if st.button(
            "20개",
            type="primary" if current_unit == 20 else "secondary",
            use_container_width=True,
            key="insp_chart_unit_btn_20",
        ):
            st.session_state["insp_chart_unit"] = 20
            st.session_state["insp_chart_selected_group"] = None
            st.rerun()
    with b40:
        if st.button(
            "40개",
            type="primary" if current_unit == 40 else "secondary",
            use_container_width=True,
            key="insp_chart_unit_btn_40",
        ):
            st.session_state["insp_chart_unit"] = 40
            st.session_state["insp_chart_selected_group"] = None
            st.rerun()
    with b100:
        if st.button(
            "100개",
            type="primary" if current_unit == 100 else "secondary",
            use_container_width=True,
            key="insp_chart_unit_btn_100",
        ):
            st.session_state["insp_chart_unit"] = 100
            st.session_state["insp_chart_selected_group"] = None
            st.rerun()


def _render_group_table(total_seqs: int, unit: int) -> int | None:
    """
    FR-INSP-T2-05: 시간 범위 테이블 렌더링.
    행 클릭 시 insp_chart_selected_group 갱신.
    기본 선택: 가장 최근(마지막) 그룹.
    반환: 현재 선택된 그룹 인덱스(0-indexed) 또는 None.
    """
    df = _build_group_table_df(total_seqs, unit)

    if df.empty:
        st.caption("검사 기록이 없습니다.")
        return None

    group_count = get_sim_group_count(total_seqs, unit)

    # 기본 선택: 가장 최근 그룹 (마지막 인덱스)
    current = st.session_state.get("insp_chart_selected_group")
    if current is None or current >= group_count:
        current = group_count - 1
        st.session_state["insp_chart_selected_group"] = current

    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="insp_group_table",
    )

    selected_rows = event.selection.rows if event else []
    if selected_rows:
        new_sel = selected_rows[0]
        if 0 <= new_sel < group_count:
            st.session_state["insp_chart_selected_group"] = new_sel
            current = new_sel

    return current


def _render_histogram(records: list[dict], unit: int) -> None:
    """FR-INSP-T2-06: 선택된 그룹의 Anomaly Score 히스토그램 렌더링."""
    selected_group = st.session_state.get("insp_chart_selected_group")

    if not records or selected_group is None:
        st.caption("Anomaly Score 히스토그램")
        return

    group_count = get_sim_group_count(len(records), unit)
    if selected_group >= group_count:
        st.caption("Anomaly Score 히스토그램")
        return

    threshold   = float(
        (st.session_state.get("insp_active_model") or {}).get("threshold", 0.5)
    )
    group_recs  = _get_group_records(records, selected_group, unit)
    group_label = get_sim_group_label(selected_group, unit)

    if not group_recs:
        st.caption("선택된 그룹에 검사 기록이 없습니다.")
        return

    fig = _build_histogram_fig(group_recs, threshold, group_label, unit)
    st.plotly_chart(fig, use_container_width=True)


_INSP_INTERVAL_SEC = 3  # 검사 1건 = 3초 (자동 검사 간격 기준)


def _build_scatter_fig(
    group_records: list[dict],
    threshold: float,
    group_label: str,
    unit: int,
) -> go.Figure:
    """
    FR-INSP-T2-07: Anomaly Score 산점도(Control Chart 스타일) Plotly 피겨 생성.

    x축: 시간(sec), 0 ~ unit×3 고정 (검사 1건 = 3초 기준)
         tick 6개 등간격 (unit=20 → 0,12,24,36,48,60 / unit=40 → 0,24,..,120 / unit=100 → 0,60,..,300)
    y축: [-0.2, 1.2] (정규화 score 시각적 여백 포함)
    파란 점: 정상 (score < threshold) / 빨간 점: 불량 (score >= threshold)
    앞뒤 점 선 연결 (control chart 스타일)
    threshold 수평 빨간 점선 표시
    """
    sorted_recs = sorted(group_records, key=lambda r: r.get("seq", 0))

    if sorted_recs:
        first_seq = sorted_recs[0].get("seq", 1)
        group_idx = (first_seq - 1) // unit
        # x = 그룹 내 0-indexed 위치 × 검사 간격(초)
        # 순번 p(1-indexed): time = (p-1) × 3sec
        x_values = [
            (r.get("seq", 0) - group_idx * unit - 1) * _INSP_INTERVAL_SEC
            for r in sorted_recs
        ]
    else:
        x_values = []

    y_values = [r.get("anomaly_score", 0.0) for r in sorted_recs]
    colors   = [
        "#4e9af1" if r.get("anomaly_score", 0.0) < threshold else "#e05555"
        for r in sorted_recs
    ]

    fig = go.Figure()

    # 데이터 산점도 (lines + markers)
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines+markers",
        marker=dict(color=colors, size=8),
        line=dict(color="lightgray", width=1),
        showlegend=False,
    ))

    # threshold 수평 빨간 점선
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"thr={threshold:.3f}",
        annotation_position="top right",
    )

    # x축: 0 ~ unit×3(sec), tick 6개 등간격
    max_sec       = unit * _INSP_INTERVAL_SEC
    tick_interval = max_sec // 5
    tick_vals     = list(range(0, max_sec + 1, tick_interval))

    fig.update_layout(
        title=f"Anomaly Score 추이 — {group_label}",
        xaxis=dict(
            range=[0, max_sec],
            title="시간 (sec)",
            tickvals=tick_vals,
            ticktext=[str(v) for v in tick_vals],
        ),
        yaxis=dict(range=[-0.2, 1.2], title="Anomaly Score"),  # #02: 여유 공간 확보
        height=300,
        margin=dict(t=50, b=40, l=45, r=15),
    )

    return fig


def _render_scatter(records: list[dict], unit: int) -> None:
    """FR-INSP-T2-07: 선택된 그룹의 Anomaly Score 산점도 렌더링."""
    selected_group = st.session_state.get("insp_chart_selected_group")

    if not records or selected_group is None:
        st.caption("Anomaly Score 산점도")
        return

    group_count = get_sim_group_count(len(records), unit)
    if selected_group >= group_count:
        st.caption("Anomaly Score 산점도")
        return

    threshold   = float(
        (st.session_state.get("insp_active_model") or {}).get("threshold", 0.5)
    )
    group_recs  = _get_group_records(records, selected_group, unit)
    group_label = get_sim_group_label(selected_group, unit)

    if not group_recs:
        st.caption("선택된 그룹에 검사 기록이 없습니다.")
        return

    fig = _build_scatter_fig(group_recs, threshold, group_label, unit)
    st.plotly_chart(fig, use_container_width=True)


def _render_chart_section() -> None:
    """
    FR-INSP-T2-05~07: KPI 카드 아래 3분할 실시간 통계 차트 영역.

    [좌] 단위 선택 버튼(20/40/100개) + 시간 범위 테이블  ← 버튼이 col_left 안에 위치
    [중] Anomaly Score 히스토그램    (FR-INSP-T2-06)
    [우] Anomaly Score 산점도        (FR-INSP-T2-07)

    버튼을 col_left 안에 배치하는 이유:
      1. (#01) 버튼 크기가 왼쪽 열 너비에 자동 맞춰짐 (1/3 페이지 폭 = 기존 대비 절반 이하)
      2. (#02) 버튼 컬럼이 별도의 외부 st.columns()를 만들지 않으므로
               col_left→col_mid 간 session_state 전파가 보장됨
    """
    records: list[dict] = st.session_state.get("insp_records", [])
    unit: int           = st.session_state.get("insp_chart_unit", 20)
    total_seqs          = len(records)

    st.markdown("---")
    st.markdown("##### 실시간 통계 차트")

    # 3분할 레이아웃 (단위 버튼도 col_left 안에 배치)
    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        # 단위 선택 버튼 — col_left 폭에 맞춰 렌더 (#01 수정)
        _render_unit_buttons(unit)
        _render_group_table(total_seqs, unit)

    with col_mid:
        _render_histogram(records, unit)

    with col_right:
        _render_scatter(records, unit)


# ── 이력 초기화 ───────────────────────────────────────────────────────────────

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
