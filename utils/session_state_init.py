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

    # 탭5 Write
    "selected_experiment_id": None, # str | None

    # 탭6 Write
    "anomaly_map_threshold": None,  # float | None

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
}


def init_session_state() -> None:
    for key, default in SESSION_STATE_SCHEMA.items():
        if key not in st.session_state:
            st.session_state[key] = default
