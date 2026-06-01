from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

from utils.messages import MSG
from utils.storage import delete_experiment, load_history

_logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def render() -> None:
    st.header("탭4. 실험 히스토리 + 결과 상세 + 모델 저장")

    # FR-T4-06: 탭 진입 시마다 history.json 재로드
    experiments = load_history()
    st.session_state["experiments"] = {
        r["experiment_id"]: r for r in experiments if "experiment_id" in r
    }

    # FR-T4-05: Guard
    if not experiments:
        st.warning(MSG["NO_EXPERIMENTS"])
        return

    _render_experiment_table(experiments)

    selected_id = st.session_state.get("selected_experiment_id")
    selected = next(
        (r for r in experiments if r.get("experiment_id") == selected_id), None
    )

    # FR-T4-03: 삭제 버튼 (선택 없으면 disabled)
    st.divider()
    if st.session_state.get("_confirm_delete") and selected is not None:
        _render_delete_confirm(selected)
    else:
        if st.button(
            "🗑 실험 삭제",
            type="secondary",
            disabled=selected is None,
            key="t5_delete_btn",
        ):
            st.session_state["_confirm_delete"] = True
            st.rerun()

    # FR-T4-02, FR-T4-04: 상세 결과 + 모델 저장 (completed 실험만)
    if selected is not None and selected.get("status") == "completed":
        st.divider()
        _render_detail(selected)
        st.divider()
        _render_model_save(selected)

    # FR-T4-07: 다중 실험 비교 차트
    completed = [r for r in experiments if r.get("status") == "completed"]
    if len(completed) >= 2:
        st.divider()
        _render_comparison_section(completed)


# ──────────────────────────────────────────────────────────────────────────────
# 실험 목록 테이블 (FR-T4-01, FR-T4-09)
# ──────────────────────────────────────────────────────────────────────────────

def _render_experiment_table(experiments: list[dict]) -> None:
    sorted_exps = sorted(
        experiments, key=lambda r: r.get("created_at", ""), reverse=True
    )
    df = _build_table_df(sorted_exps)

    # FR-T4-09: 중단 실험 회색 텍스트
    def _row_style(row: pd.Series) -> list[str]:
        if row["상태"] == "중단":
            return ["color: gray"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_row_style, axis=1)

    selection = st.dataframe(
        styled,
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        key="t5_table",
    )

    rows = selection.selection.rows if selection else []
    if rows:
        idx = rows[0]
        if 0 <= idx < len(sorted_exps):
            st.session_state["selected_experiment_id"] = sorted_exps[idx].get(
                "experiment_id"
            )


def _build_table_df(sorted_exps: list[dict]) -> pd.DataFrame:
    rows = []
    for r in sorted_exps:
        metrics = r.get("metrics") or {}
        is_completed = r.get("status") == "completed"

        def _m(key: str) -> str:
            return f"{metrics.get(key, 0):.4f}" if is_completed and key in metrics else "—"

        rows.append(
            {
                "실험명": r.get("name", ""),
                "모델": r.get("model_type", ""),
                "파라미터 요약": _param_summary(r),
                "Accuracy": _m("accuracy"),
                "Precision": _m("precision"),
                "Recall": _m("recall"),
                "F1": _m("f1_score"),
                "F2": _m("f2_score"),
                "AUC": _m("auc"),
                "실행 시각": r.get("created_at", "")[:19].replace("T", " "),
                "상태": r.get("status", ""),
            }
        )
    return pd.DataFrame(rows)


_BACKBONE_ABBREV = {
    "wide_resnet50_2": "wrn50",
    "resnet50": "r50",
    "resnet18": "r18",
}


def _param_summary(r: dict) -> str:
    model_type = r.get("model_type", "")
    params = r.get("model_params") or {}
    if model_type == "efficientad":
        size = params.get("model_size", "?")
        steps = params.get("train_steps", "?")
        opt = params.get("optimizer", "?")
        steps_str = f"{steps // 1000}k" if isinstance(steps, int) else str(steps)
        return f"{size}/{steps_str}/{opt}"
    if model_type == "patchcore":
        backbone = params.get("backbone", "?")
        ratio = params.get("coreset_sampling_ratio", "?")
        backbone_abbr = _BACKBONE_ABBREV.get(backbone, backbone)
        return f"{backbone_abbr}/{ratio}"
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# 실험 삭제 (FR-T4-03)
# ──────────────────────────────────────────────────────────────────────────────

def _render_delete_confirm(record: dict) -> None:
    exp_id = record.get("experiment_id", "")
    st.warning("삭제 후 복구할 수 없습니다. 계속하시겠습니까?")
    col1, col2, _ = st.columns([1, 1, 6])
    with col1:
        if st.button("확인", type="primary", key="t5_delete_confirm"):
            try:
                delete_experiment(exp_id, record.get("model_path"))
                _logger.warning("experiment_deleted: id=%s", exp_id)
            except Exception as e:
                st.error(f"삭제 실패: {e}")
                return
            st.session_state.pop("selected_experiment_id", None)
            st.session_state.pop("_confirm_delete", None)
            st.success("실험이 삭제되었습니다.")
            st.rerun()
    with col2:
        if st.button("취소", key="t5_delete_cancel"):
            st.session_state.pop("_confirm_delete", None)
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# 실험 상세 결과 (FR-T4-02)
# ──────────────────────────────────────────────────────────────────────────────

def _render_detail(record: dict) -> None:
    metrics = record.get("metrics") or {}
    st.subheader(f"상세 결과 — {record.get('name', '')}")

    # 지표 카드
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    with c2:
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    with c3:
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    with c4:
        st.metric("F1", f"{metrics.get('f1_score', 0):.4f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(_confusion_matrix_fig(metrics), use_container_width=True)
    with col2:
        st.plotly_chart(_roc_curve_fig(metrics), use_container_width=True)
    with col3:
        threshold_value = record.get("threshold_value")
        st.plotly_chart(
            _score_dist_fig(metrics, threshold_value=threshold_value),
            use_container_width=True,
        )


def _confusion_matrix_fig(metrics: dict) -> go.Figure:
    cm = metrics.get("confusion_matrix") or {}
    tn, fp, fn, tp = cm.get("tn", 0), cm.get("fp", 0), cm.get("fn", 0), cm.get("tp", 0)

    fig = go.Figure(
        go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["예측 정상", "예측 결함"],
            y=["실제 정상", "실제 결함"],
            text=[[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(title="Confusion Matrix", height=320)
    return fig


def _roc_curve_fig(metrics: dict) -> go.Figure:
    scores = metrics.get("anomaly_scores") or []
    labels = metrics.get("image_labels") or []

    fig = go.Figure()
    if scores and labels and len(set(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        fig.add_trace(
            go.Scatter(
                x=fpr.tolist(),
                y=tpr.tolist(),
                mode="lines",
                name=f"ROC (AUC={auc:.4f})",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Random",
            )
        )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="FPR",
        yaxis_title="TPR",
        height=320,
    )
    return fig


def _score_dist_fig(metrics: dict, threshold_value: float | None = None) -> go.Figure:
    scores = metrics.get("anomaly_scores") or []
    labels = metrics.get("image_labels") or []

    fig = go.Figure()
    if scores and labels:
        arr = np.array(scores, dtype=np.float64)
        s_min, s_max = arr.min(), arr.max()
        if s_max > s_min:
            norm = (arr - s_min) / (s_max - s_min)
        else:
            norm = np.zeros_like(arr)

        norm_threshold = (
            (threshold_value - s_min) / (s_max - s_min)
            if threshold_value is not None and s_max > s_min
            else threshold_value
        )

        normal = [float(v) for v, l in zip(norm, labels) if l == 0]
        defect = [float(v) for v, l in zip(norm, labels) if l == 1]
        if normal:
            fig.add_trace(go.Histogram(x=normal, name="정상", opacity=0.7, nbinsx=30))
        if defect:
            fig.add_trace(go.Histogram(x=defect, name="결함", opacity=0.7, nbinsx=30))
        if norm_threshold is not None:
            fig.add_vline(
                x=norm_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"threshold={norm_threshold:.4f}",
            )
    fig.update_layout(
        title="Anomaly Score 분포 (Min-Max 정규화)",
        xaxis_title="Anomaly Score (0 ~ 1)",
        yaxis_title="이미지 수",
        xaxis=dict(range=[0, 1]),
        barmode="overlay",
        height=320,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 모델 저장 (FR-T4-04, FR-T4-08)
# ──────────────────────────────────────────────────────────────────────────────

def _render_model_save(record: dict) -> None:
    st.subheader("모델 저장")
    exp_id = record.get("experiment_id", "")
    default_path = record.get("model_path") or f"./models/{exp_id}/"
    save_path = st.text_input("저장 경로", value=default_path, key="t5_save_path")
    if st.button("💾 모델 저장", type="primary", key="t5_save_btn"):
        _do_model_save(record, save_path)


def _do_model_save(record: dict, save_path: str) -> None:
    src_dir = Path(record.get("model_path", ""))
    dst_dir = Path(save_path)

    try:
        check_target = str(dst_dir.parent) if dst_dir.parent.exists() else "."
        usage = shutil.disk_usage(check_target)
        free_mb = usage.free / (1024 ** 2)
        if free_mb < 100:
            st.error(
                f"디스크 여유 공간이 부족합니다 ({free_mb:.0f} MB). "
                "100 MB 이상 필요합니다."
            )
            return
        model_type = record.get("model_config", {}).get("model_type", "")
        warn_mb = 1024 if model_type == "patchcore" else 500
        if free_mb < warn_mb:
            st.warning(
                f"디스크 여유 공간이 {free_mb:.0f} MB입니다. "
                f"저장은 허용되지만 {warn_mb} MB 이상 권장합니다."
            )
    except OSError:
        pass

    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        pth_src = src_dir / "model_state_dict.pth"
        yaml_src = src_dir / "configs.yaml"
        total_bytes = 0

        if pth_src.exists():
            pth_dst = dst_dir / "model_state_dict.pth"
            if dst_dir.resolve() != src_dir.resolve():
                shutil.copy2(pth_src, pth_dst)
            total_bytes += pth_dst.stat().st_size if pth_dst.exists() else pth_src.stat().st_size

        if yaml_src.exists():
            yaml_dst = dst_dir / "configs.yaml"
            if dst_dir.resolve() != src_dir.resolve():
                shutil.copy2(yaml_src, yaml_dst)
            total_bytes += yaml_dst.stat().st_size if yaml_dst.exists() else yaml_src.stat().st_size

        size_mb = total_bytes / (1024 ** 2)
        _logger.info("model_saved: id=%s path=%s", record.get("experiment_id"), str(dst_dir))
        st.success(
            f"저장 완료\n"
            f"경로: {dst_dir}\n"
            f"파일: model_state_dict.pth, configs.yaml\n"
            f"용량: {size_mb:.1f} MB"
        )
    except Exception as e:
        st.error(f"모델 저장 실패: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 다중 실험 비교 차트 (FR-T4-07)
# ──────────────────────────────────────────────────────────────────────────────

_METRIC_MAP = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1_score",
    "F2": "f2_score",
}


def _render_comparison_section(completed: list[dict]) -> None:
    with st.expander("다중 실험 비교 차트", expanded=False):
        st.write("비교할 실험을 선택하세요 (최대 10개)")

        n_cols = min(len(completed), 4)
        cols = st.columns(n_cols)
        selected_exps = []
        for i, r in enumerate(completed):
            name = r.get("name") or r.get("experiment_id", f"exp_{i}")
            with cols[i % n_cols]:
                if st.checkbox(name, key=f"t5_cmp_{r.get('experiment_id', i)}"):
                    selected_exps.append(r)

        if len(selected_exps) > 10:
            st.warning("최대 10개까지 선택 가능합니다.")
            selected_exps = selected_exps[:10]

        if len(selected_exps) < 2:
            st.info("비교 차트를 보려면 실험을 2개 이상 선택하세요.")
            return

        selected_metrics = st.multiselect(
            "비교 메트릭",
            list(_METRIC_MAP.keys()),
            default=["Accuracy", "F1"],
            key="t5_cmp_metrics",
        )
        chart_type = st.radio(
            "차트 유형",
            ["막대 차트", "레이더 차트"],
            horizontal=True,
            key="t5_cmp_type",
        )

        if not selected_metrics:
            return

        if chart_type == "막대 차트":
            _render_bar_comparison(selected_exps, selected_metrics)
        else:
            _render_radar_comparison(selected_exps, selected_metrics)


def _render_bar_comparison(exps: list[dict], selected_metrics: list[str]) -> None:
    exp_names = [r.get("name") or r.get("experiment_id", "") for r in exps]
    fig = go.Figure()
    for display_key in selected_metrics:
        key = _METRIC_MAP[display_key]
        values = [r.get("metrics", {}).get(key, 0) for r in exps]
        fig.add_trace(go.Bar(name=display_key, x=exp_names, y=values))
    fig.update_layout(
        barmode="group",
        title="실험 비교",
        xaxis_title="실험",
        yaxis_title="값",
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_radar_comparison(exps: list[dict], selected_metrics: list[str]) -> None:
    if len(selected_metrics) < 3:
        st.info("레이더 차트는 메트릭을 3개 이상 선택해야 합니다.")
        return

    fig = go.Figure()
    cats = selected_metrics + [selected_metrics[0]]
    for r in exps:
        m = r.get("metrics") or {}
        vals = [m.get(_METRIC_MAP[k], 0) for k in selected_metrics]
        vals = vals + [vals[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=cats,
                fill="toself",
                name=r.get("name") or r.get("experiment_id", ""),
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="실험 비교 (레이더 차트)",
    )
    st.plotly_chart(fig, use_container_width=True)
