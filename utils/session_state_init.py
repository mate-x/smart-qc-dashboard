import streamlit as st

SESSION_STATE_SCHEMA: dict = {
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
    "current_run_status": "idle",   # "idle" | "running" | "stopped" | "completed"
    "current_exp_id": None,         # str | None — 현재 실행 중인 실험 ID

    # 탭4 내부 상태 (접두사 _ = 탭4 전용)
    "_stop_event": None,            # threading.Event | None
    "_result_queue": None,          # queue.Queue | None
    "_progress": None,              # dict | None  {"step", "total", "loss"}
    "_log_lines": [],               # list[str]  최대 100줄
    "_loss_history": [],            # list[dict]  {"step": int, "loss": float}

    # 탭5 Write
    "selected_experiment_id": None, # str | None

    # 탭6 Write
    "anomaly_map_threshold": None,  # float | None
}


def init_session_state() -> None:
    for key, default in SESSION_STATE_SCHEMA.items():
        if key not in st.session_state:
            st.session_state[key] = default
