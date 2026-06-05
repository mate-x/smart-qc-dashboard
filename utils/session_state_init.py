import streamlit as st

SESSION_STATE_SCHEMA: dict = {
    # 대시보드 라우팅
    "active_dashboard": "explorer",  # "explorer" | "inspection"

    # 탭1 Write
    "dataset_path": None,           # str | None
    "dataset_meta": None,           # dict | None  (00_Global_Context 1.5절)

    # 탭2 Write
    "preprocessing_config": None,   # dict | None  (00_Global_Context 1.6절)

    # 탭3 Write
    "model_config": None,           # dict | None  (00_Global_Context 1.7절)
    "device_info": None,            # dict | None  (00_Global_Context 1.8절)

    # 실험 대기열 (탭2 Write · 탭3 Read/Write — FR-T2-16~18, FR-T3-14~15)
    "experiment_queue":    [],      # list[dict]  항목: {name, preprocessing_config, model_config, status}
                                    # status: "대기중" | "진행중" | "완료" | "실패" | "건너뜀"

    # 탭4 Write
    "experiments": {},              # dict[str, dict]  key: experiment_id
    "current_run_status": "idle",   # "idle" | "running" | "paused" | "stopped" | "completed"
    "current_exp_id": None,         # str | None — 현재 실행 중인 실험 ID

    # 탭4 내부 상태 (접두사 _ = 탭4 전용)
    "_stop_event": None,            # threading.Event | None
    "_pause_event": None,           # threading.Event | None — 일시정지 제어
    "_result_queue": None,          # queue.Queue | None
    "_worker": None,                # TrainingWorker | None  — 단일 워커 보장(07 PRD §8.2)
    "_progress": None,              # dict | None  {"step", "total", "loss", "elapsed"}
    "_log_lines": [],               # list[str]  최대 100줄
    "_loss_history": [],            # list[dict]  {"step": int, "loss": float}
    "_last_ckpt_path": None,        # str | None — 가장 최근 저장된 체크포인트 경로
    "_current_stage_idx": None,     # int | None — 현재 학습 단계 인덱스 (FR-T3-11/12)
    "_current_stage_name": None,    # str | None — 현재 학습 단계 이름
    "_batch_queue_mode":     False,  # bool — 일괄 학습 실행 중 여부 (FR-T3-15)
    "_batch_total_count":    0,     # int  — 일괄 학습 전체 항목 수 (배너 표시용)
    "_batch_skip_current":   False, # bool — 현재 학습 건너뛰기 신호 (⏭ 클릭 시 True)
    "_batch_advance_pending": False, # bool — 완료/실패/건너뜀 후 다음 항목 자동 시작 대기

    # 탭5 Write
    "selected_experiment_id": None, # str | None

    # 탭6 Write
    "anomaly_map_threshold": None,  # float | None

    # TODO: [삭제 예정] 비전검사 대시보드가 React 앱으로 이전됨 — 아래 insp_* 키 전체 제거
    # 비전검사 대시보드 (insp_ 네임스페이스, R-INSP-01)
    "insp_active_model":   None,    # dict | None — history.json experiment 레코드 전체
    "insp_records":        [],      # list[dict] — inspection_record 배열 (1.10절)
    "insp_seq_counter":    0,       # int — 다음 seq 값
    "insp_auto_active":    False,   # bool — 자동 검사 실행 중 여부
    "insp_last_result":    None,    # dict | None — 직전 추론 결과
    "insp_last_anomaly_map": None,  # np.ndarray | None — 직전 이상 맵 (H, W) float32
    "insp_defect_popup":   False,   # bool — 불량 감지 팝업 표시 여부
    "insp_test_pool":      [],      # list[tuple[str, str]] — (절대경로, "양품"|"불량")
    "insp_pool_index":     0,       # int — 현재 샘플링 위치

    # 비전검사 통계 차트 (FR-INSP-T2-05)
    "insp_chart_unit":           20,   # int — 현재 그룹 단위 (20/40/100)
    "insp_chart_selected_group": None, # int | None — 선택된 그룹 인덱스(0-indexed)
}


def init_session_state() -> None:
    for key, default in SESSION_STATE_SCHEMA.items():
        if key not in st.session_state:
            st.session_state[key] = default
