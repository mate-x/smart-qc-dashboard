"""
inspection/tabs/insp_tab3_model.py — 딥러닝 모델 교체 탭

FR-INSP-T3-01: status=="completed" 실험 목록 (F1 내림차순)
FR-INSP-T3-02: [이 모델로 검사 시작] 버튼 — R-INSP-05 순서 준수
FR-INSP-T3-03: Guard 없음. 완료 실험 없으면 안내 메시지

session_state 쓰기 권한: insp_active_model (탭3), 나머지 insp_* 는 reset_inspection_state() 위임.
history.json 쓰기 금지 (R-INSP-04).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.logger import log_info
from utils.storage import load_history

_NO_COMPLETED_EXP = "사용 가능한 완료된 실험이 없습니다. 탭4에서 학습을 먼저 완료해 주세요."


def render() -> None:
    st.subheader("딥러닝 모델 교체")

    # F.4: 경고 메시지는 항상 표시 (이력 유무 무관)
    st.warning("⚠️ 모델을 교체하면 현재 세션의 모든 검사 이력이 삭제됩니다.")

    # FR-INSP-T3-01: completed 실험 로드 (history.json 기준 — session_state보다 최신)
    experiments = _load_completed_experiments()

    if not experiments:
        st.info(_NO_COMPLETED_EXP)
        return

    # 실험 목록 테이블 (F.2 스펙)
    df = _build_experiment_df(experiments)

    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="insp_tab3_exp_table",
    )

    selected_rows = event.selection.rows
    selected_experiment = experiments[selected_rows[0]] if selected_rows else None

    # F.3: 적용 버튼 — 행 미선택 시 disabled
    if st.button(
        "✅ 이 모델로 검사 시작",
        type="primary",
        disabled=(selected_experiment is None),
    ):
        _apply_model(selected_experiment)  # type: ignore[arg-type]


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────


def _load_completed_experiments() -> list[dict]:
    """history.json에서 status=="completed" 실험 로드 후 F1 내림차순 정렬."""
    all_records = load_history()
    completed = [r for r in all_records if r.get("status") == "completed"]
    completed.sort(
        key=lambda r: (r.get("metrics") or {}).get("f1_score", 0.0),
        reverse=True,
    )
    return completed


def _build_experiment_df(experiments: list[dict]) -> pd.DataFrame:
    """실험 목록을 F.2 스펙 DataFrame으로 변환. 현재 적용 모델에 ✅현재 표시."""
    current_id = (st.session_state.get("insp_active_model") or {}).get("experiment_id")

    rows = []
    for exp in experiments:
        exp_id = exp.get("experiment_id", "")
        name = exp.get("name") or exp_id
        if exp_id == current_id:
            name = f"{name} ✅현재"

        metrics = exp.get("metrics") or {}
        f1 = metrics.get("f1_score")
        auc = metrics.get("auc")

        created_at = exp.get("created_at", "")
        if "T" in created_at:
            # "2026-05-26T14:02:31+09:00" → "2026-05-26 14:02"
            created_at = created_at.replace("T", " ").split("+")[0][:16]

        product = exp.get("product_name", "") or "(입력 없음)"
        rows.append({
            "실험명":    name,
            "검사 제품": product,
            "모델 타입": exp.get("model_type", ""),
            "F1":       f"{f1:.4f}" if f1 is not None else "-",
            "AUC":      f"{auc:.4f}" if auc is not None else "-",
            "실행 시각": created_at,
        })

    return pd.DataFrame(rows)


def _apply_model(selected_experiment: dict) -> None:
    """
    FR-INSP-T3-02: 모델 교체 처리.

    R-INSP-05 초기화 순서:
      Step 1. _load_insp_model.clear()     — 캐시 무효화 (A-19)
      Step 2. reset_inspection_state()     — insp_active_model 제외 전체 초기화
      Step 3. insp_active_model 갱신
      Step 4. build_test_pool() → insp_test_pool/index 설정
      Step 5. st.rerun()
    """
    from inspection.utils.insp_session_init import _load_insp_model, reset_inspection_state
    from inspection.utils.test_sampler import build_test_pool

    # Step 1
    _load_insp_model.clear()

    # Step 2
    reset_inspection_state()

    # Step 3
    preprocessing_config = {
        "method":     selected_experiment.get("preprocessing_method", "none"),
        "params":     selected_experiment.get("preprocessing_params") or {},
        "image_size": selected_experiment.get("image_size", 256),
    }
    # 정규화 기준값 계산 (학습 테스트셋 anomaly_score의 min/max)
    from inspection.utils.insp_session_init import normalize_anomaly_score
    _metrics    = selected_experiment.get("metrics") or {}
    _all_scores = _metrics.get("anomaly_scores") or []
    score_min   = float(min(_all_scores)) if _all_scores else 0.0
    score_max   = float(max(_all_scores)) if _all_scores else 1.0

    # threshold도 동일 기준으로 정규화 → insp_tab1 판정과 차트 모두 [0,1] 공간
    _raw_threshold        = _resolve_threshold(selected_experiment)
    _threshold_normalized = normalize_anomaly_score(_raw_threshold, score_min, score_max)

    st.session_state["insp_active_model"] = {
        "experiment_id":        selected_experiment["experiment_id"],
        "model_path":           selected_experiment["model_path"],
        "model_type":           selected_experiment["model_type"],
        "threshold":            _threshold_normalized,  # [0, 1] 정규화됨
        "dataset_path":         selected_experiment["dataset_path"],
        "preprocessing_config": preprocessing_config,
        "score_min":            score_min,   # 정규화 기준: 학습 테스트셋 최솟값
        "score_max":            score_max,   # 정규화 기준: 학습 테스트셋 최댓값
    }

    # Step 4
    bg_method = selected_experiment.get("background_method", "none")
    try:
        pool = build_test_pool(selected_experiment["dataset_path"], background_method=bg_method)
    except FileNotFoundError as e:
        st.error(f"테스트 풀 구성 실패: {e}")
        st.session_state["insp_active_model"] = None
        return

    if not pool:
        st.error(
            "ERR_INSP_TEST_POOL_EMPTY: 테스트 이미지를 찾을 수 없습니다. "
            f"데이터셋 경로를 확인해 주세요: {selected_experiment['dataset_path']}/test/"
        )
        st.session_state["insp_active_model"] = None
        return

    st.session_state["insp_test_pool"] = pool
    st.session_state["insp_pool_index"] = 0

    # Step 5
    log_info(
        "insp_model_applied",
        f"모델 적용 완료: {selected_experiment['experiment_id']}",
        tab="insp_tab3",
        data={
            "experiment_id": selected_experiment["experiment_id"],
            "model_type":    selected_experiment.get("model_type", "unknown"),
        },
    )
    st.rerun()


def _resolve_threshold(experiment: dict) -> float:
    """
    실험 레코드에서 추론용 threshold 추출.

    absolute 메서드: threshold_value 직접 사용.
    percentile 메서드: metrics.anomaly_scores 중 정상(label==0) 분포의
                       percentile로 재계산 (훈련 정상 스코어 미저장 문제 보완).
                       정상 스코어가 없으면 threshold_value fallback.
    """
    method = experiment.get("threshold_method", "absolute")
    value = float(experiment.get("threshold_value", 0.5))

    if method == "absolute":
        return value

    # percentile: 테스트 정상 스코어로 근사
    metrics = experiment.get("metrics") or {}
    scores = metrics.get("anomaly_scores", [])
    labels = metrics.get("image_labels", [])

    if scores and labels and len(scores) == len(labels):
        normal_scores = [s for s, l in zip(scores, labels) if l == 0]
        if normal_scores:
            return float(np.percentile(normal_scores, value))

    return value
