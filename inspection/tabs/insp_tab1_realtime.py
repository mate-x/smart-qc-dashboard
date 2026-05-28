"""
inspection/tabs/insp_tab1_realtime.py — 실시간 검사 탭

FR-INSP-T1-01: 수동 검사 (1개 검사)
FR-INSP-T1-02: 자동 검사 (3초마다 1개)
FR-INSP-T1-03: 3열 결과 패널 [1, 2, 2] — 판정카드 / 원본이미지 / Anomaly Map
FR-INSP-T1-04: 불량 감지 팝업
FR-INSP-T1-06: test_pool 소진 시 재셔플 알림 + 빈 풀 guard

session_state 쓰기 권한: insp_records, insp_seq_counter, insp_last_result,
                         insp_last_anomaly_map, insp_auto_active, insp_defect_popup
history.json 쓰기 금지 (R-INSP-04).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import streamlit as st

from inspection.utils.insp_session_init import get_insp_model
from utils.logger import log_info, log_warning
from utils.messages import INSP_MSG

KST = timezone(timedelta(hours=9))
_GUARD_MSG = "검사에 사용할 모델이 선택되지 않았습니다. 탭3에서 모델을 먼저 선택해 주세요."


def render() -> None:
    st.subheader("실시간 검사")

    # Guard: 모델 미선택 (FR-INSP-CMN-03)
    if st.session_state.get("insp_active_model") is None:
        st.info(_GUARD_MSG)
        return

    # Guard: 빈 test_pool — 모델은 있으나 테스트 이미지 없음 (ERR_INSP_TEST_POOL_EMPTY)
    if not st.session_state.get("insp_test_pool"):
        st.error(INSP_MSG["POOL_EMPTY"])
        return

    # 불량 감지 팝업 우선 렌더링 (06_API §7.1.1)
    if st.session_state.get("insp_defect_popup"):
        _render_defect_popup()
        return

    # 자동 검사 중 배너 (G.5)
    is_auto = bool(st.session_state.get("insp_auto_active", False))
    if is_auto:
        st.warning("🔄 자동 검사 진행 중...")

    # 버튼 행 (D.2)
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        if st.button(
            "🔍 수동 검사 (1개 검사)",
            type="primary",
            disabled=is_auto,
            use_container_width=True,
        ):
            log_info(
                "insp_inspection_started_manual",
                "수동 검사 1회 실행",
                tab="insp_tab1",
            )
            ok = _run_single_inspection()
            if ok:
                st.rerun()
    with col_b2:
        if st.button(
            "▶ 자동 검사 (3초마다 1개)",
            type="secondary",
            disabled=is_auto,
            use_container_width=True,
        ):
            log_info(
                "insp_inspection_started_auto",
                "자동 검사 시작",
                tab="insp_tab1",
                data={"interval_s": 3.0},
            )
            st.session_state["insp_auto_active"] = True
            st.rerun()
    with col_b3:
        if st.button(
            "⏹ 자동 검사 중지",
            type="secondary",
            disabled=not is_auto,
            use_container_width=True,
        ):
            log_info(
                "insp_inspection_stopped_auto",
                "자동 검사 수동 중지",
                tab="insp_tab1",
                data={"total_inspected": st.session_state.get("insp_seq_counter", 0)},
            )
            st.session_state["insp_auto_active"] = False
            st.rerun()

    # 3열 결과 패널 (D.3)
    _render_result_panel()

    # 자동 검사 루프 — ADR-INSP-04: sleep + rerun 패턴 (스레드 없음)
    if st.session_state.get("insp_auto_active") and not st.session_state.get("insp_defect_popup"):
        ok = _run_single_inspection()
        if ok and not st.session_state.get("insp_defect_popup"):
            time.sleep(3)   # NFR-INSP-02: 자동 검사 타이밍 ≤ 0.5초 오차
        st.rerun()


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────


def _render_defect_popup() -> None:
    """FR-INSP-T1-05: 불량 감지 팝업 — 팝업만 렌더링하고 return (D.4 스펙)."""
    st.error(f"❌ {INSP_MSG['DEFECT_DETECTED']}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 확인 및 재개", type="primary", use_container_width=True):
            st.session_state["insp_auto_active"] = True
            st.session_state["insp_defect_popup"] = False
            st.rerun()
    with col2:
        if st.button("🛑 검사 종료", type="secondary", use_container_width=True):
            st.session_state["insp_auto_active"] = False
            st.session_state["insp_defect_popup"] = False
            st.rerun()


def _render_result_panel() -> None:
    """FR-INSP-T1-03: [1, 2, 2] 3열 패널 — 판정카드 / 원본이미지 / Anomaly Map.

    초기 상태(insp_last_result is None):
      col1: 안내 텍스트
      col2: 이미지 placeholder
      col3: Anomaly Map placeholder
    """
    from utils.image_utils import anomaly_map_to_heatmap

    last_result: dict | None = st.session_state.get("insp_last_result")
    last_anomaly_map: np.ndarray | None = st.session_state.get("insp_last_anomaly_map")

    col1, col2, col3 = st.columns([1, 2, 2])

    with col1:
        if last_result is None:
            st.info("검사 버튼을 눌러 시작하세요.")
        else:
            verdict = last_result["verdict"]
            score = last_result["anomaly_score"]
            if verdict == "양품":
                st.success("✅ 양품")
            else:
                st.error("❌ 불량")
            st.metric("Anomaly Score", f"{score:.4f}")

    with col2:
        if last_result is not None:
            st.image(
                last_result["image_path"],
                caption=last_result["image_name"],
                use_container_width=True,
            )
        else:
            st.caption("원본 이미지가 여기에 표시됩니다.")

    with col3:
        if last_anomaly_map is not None:
            heatmap = anomaly_map_to_heatmap(last_anomaly_map)
            st.image(heatmap, caption="Anomaly Map", use_container_width=True)
        else:
            st.caption("Anomaly Map이 여기에 표시됩니다.")


def _run_single_inspection() -> bool:
    """
    단일 이미지 추론 흐름 (07_Backend §12.3).
    수동/자동 공용. session_state 갱신 후 반환 (st.rerun 호출 없음).

    Returns:
        True: 정상 완료. False: 오류 발생 (st.error 표시 후 반환).
    """
    from inspection.utils.test_sampler import sample_from_pool
    from utils.model_factory import run_inference
    from utils.image_utils import apply_preprocessing

    active = st.session_state["insp_active_model"]
    threshold = active["threshold"]

    # 1. test_pool에서 이미지 샘플링 (FR-INSP-T1-06)
    try:
        image_path, _gt_label, was_reshuffled = sample_from_pool()
    except RuntimeError as e:
        st.error(str(e))
        st.session_state["insp_auto_active"] = False
        return False

    # pool 소진 후 재셔플 알림 (FR-INSP-T1-06 — insp_pool_reshuffled 이벤트)
    if was_reshuffled:
        st.toast(INSP_MSG["POOL_RESHUFFLED"], icon="🔄")

    # 2. 모델 로드
    model = get_insp_model()
    if model is None:
        st.error("모델 로드 실패.")
        st.session_state["insp_auto_active"] = False
        return False

    # 3. 전처리 + 추론
    preprocessing_config = active.get("preprocessing_config") or {"method": "none", "params": {}}
    _, image_tensor = apply_preprocessing(image_path, preprocessing_config)
    image_tensor = image_tensor.unsqueeze(0)           # (1, C, H, W)
    anomaly_map = run_inference(model, image_tensor)   # (H, W) float32

    # 4. 이미지 레벨 Score
    anomaly_score = float(np.max(anomaly_map))

    # 5. 판정
    verdict = "불량" if anomaly_score >= threshold else "양품"

    # 6. inspection_record 구성 (00_Global §1.10 스키마)
    seq = st.session_state["insp_seq_counter"] + 1
    record = {
        "seq":           seq,
        "inspected_at":  datetime.now(tz=KST).strftime("%Y-%m-%d %H:%M:%S"),
        "image_name":    Path(image_path).name,
        "image_path":    image_path,
        "verdict":       verdict,
        "anomaly_score": round(anomaly_score, 6),
    }

    # 7. session_state 갱신
    st.session_state["insp_records"].append(record)
    st.session_state["insp_seq_counter"] = seq
    st.session_state["insp_last_result"] = record
    st.session_state["insp_last_anomaly_map"] = anomaly_map

    # 8. 불량 감지 로그 (자동·수동 공통)
    if verdict == "불량":
        log_warning(
            "insp_defect_detected",
            f"불량 감지: {Path(image_path).name} (score={anomaly_score:.4f})",
            tab="insp_tab1",
            data={
                "image_name":    Path(image_path).name,
                "anomaly_score": anomaly_score,
                "threshold":     threshold,
            },
        )

    # 9. 자동 검사 중 불량 감지 → 루프 중지 + 팝업 설정
    if verdict == "불량" and st.session_state.get("insp_auto_active"):
        log_info(
            "insp_inspection_stopped_auto",
            "불량 감지로 자동 검사 중지",
            tab="insp_tab1",
            data={"total_inspected": seq},
        )
        st.session_state["insp_auto_active"] = False
        st.session_state["insp_defect_popup"] = True

    return True
