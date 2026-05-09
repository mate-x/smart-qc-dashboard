from __future__ import annotations

import queue
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st

from utils.cache_manager import set_anomaly_map_cache
from utils.messages import MSG
from utils.storage import save_completed_experiment

KST = timezone(timedelta(hours=9))
_MAX_LOG_LINES = 100


# ──────────────────────────────────────────────────────────────
# 진입점
# ──────────────────────────────────────────────────────────────

def render() -> None:
    st.header("탭4. 학습 시작 + 학습 로그")
    if not _guard():
        return
    _render_ui()
    _handle_events()


# ──────────────────────────────────────────────────────────────
# 진입 조건 검사
# ──────────────────────────────────────────────────────────────

def _guard() -> bool:
    if st.session_state.get("dataset_path") is None:
        st.warning(MSG["NO_DATASET"])
        return False
    if st.session_state.get("preprocessing_config") is None:
        st.warning(MSG["NO_PREPROCESSING"])
        return False
    if st.session_state.get("model_config") is None:
        st.warning(MSG["NO_MODEL_CONFIG"])
        return False
    return True


# ──────────────────────────────────────────────────────────────
# UI 렌더링 (상태별 분기)
# ──────────────────────────────────────────────────────────────

def _render_ui() -> None:
    status = st.session_state.get("current_run_status", "idle")
    if status == "running":
        _render_running()
    elif status == "completed":
        _render_completed()
    elif status == "stopped":
        _render_stopped()
    elif status == "error":
        _render_error()
    else:
        _render_idle()


def _render_idle() -> None:
    st.info("학습 중 새로고침 시 학습 상태를 확인할 수 없습니다.", icon="⚠️")

    model_config = st.session_state.get("model_config", {})
    model_type   = model_config.get("model_type", "?")
    dataset_path = st.session_state.get("dataset_path", "?")

    st.write(f"**모델 타입**: `{model_type}`")
    st.write(f"**데이터셋**: `{dataset_path}`")

    if st.button("▶ 학습 시작", type="primary", key="btn_start"):
        _handle_start_training()
        st.rerun()


def _render_running() -> None:
    exp_id = st.session_state.get("current_exp_id", "?")
    st.info("학습 중 새로고침 시 학습 상태를 확인할 수 없습니다.", icon="⚠️")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.write(f"**실험 ID**: `{exp_id}`")
    with col2:
        if st.button("■ 중단", type="secondary", key="btn_stop"):
            _handle_stop_training()

    progress_data = st.session_state.get("_progress")
    if progress_data:
        step    = progress_data.get("step", 0)
        total   = progress_data.get("total", 1)
        loss    = progress_data.get("loss", 0.0)
        elapsed = progress_data.get("elapsed", 0.0)
        pct     = step / max(total, 1)
        st.progress(
            pct,
            text=f"배치 {step}/{total} | Loss: {loss:.4f} | 경과: {elapsed:.1f}s",
        )
    else:
        st.progress(0.0, text="초기화 중...")

    log_lines = st.session_state.get("_log_lines", [])
    if log_lines:
        st.text_area(
            "학습 로그 (최근 30줄)",
            value="\n".join(log_lines[-30:]),
            height=200,
            disabled=True,
            key="log_area_running",
        )


def _render_completed() -> None:
    exp_id = st.session_state.get("current_exp_id", "?")
    st.success(f"학습 완료: `{exp_id}`")

    experiments = st.session_state.get("experiments", {})
    record      = experiments.get(exp_id, {})
    metrics     = record.get("metrics", {})

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC",      f"{metrics.get('auc', 0):.4f}")
        c2.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
        c3.metric("Recall",   f"{metrics.get('recall', 0):.4f}")
        c4.metric("정확도",    f"{metrics.get('accuracy', 0):.4f}")

        st.write(f"소요 시간: **{record.get('duration_seconds', 0)}초** | "
                 f"Threshold: **{record.get('threshold', 0):.4f}**")

    log_lines = st.session_state.get("_log_lines", [])
    if log_lines:
        with st.expander("학습 로그 전체 보기"):
            st.text("\n".join(log_lines))

    if st.button("▶ 새 학습 시작", key="btn_new_run"):
        st.session_state["current_run_status"] = "idle"
        st.rerun()


def _render_stopped() -> None:
    st.warning(MSG["TRAIN_STOPPED"])
    step = st.session_state.get("_stopped_step", 0)
    st.write(f"중단 시점 배치: **{step}**")

    log_lines = st.session_state.get("_log_lines", [])
    if log_lines:
        with st.expander("로그 보기"):
            st.text("\n".join(log_lines))

    if st.button("▶ 새 학습 시작", key="btn_restart_stopped"):
        st.session_state["current_run_status"] = "idle"
        st.rerun()


def _render_error() -> None:
    error_info = st.session_state.get("_last_error") or {}
    st.error(f"학습 오류: {error_info.get('exception', '알 수 없는 오류')}")
    tb = error_info.get("traceback", "")
    if tb:
        with st.expander("오류 상세 (traceback)"):
            st.code(tb)

    if st.button("▶ 새 학습 시작", key="btn_restart_error"):
        st.session_state["current_run_status"] = "idle"
        st.session_state["_last_error"] = None
        st.rerun()


# ──────────────────────────────────────────────────────────────
# 이벤트 핸들러 (폴링 루프)
# ──────────────────────────────────────────────────────────────

def _handle_events() -> None:
    """
    running 상태에서만 큐를 드레인하고 1초 뒤 rerun (R-THREAD-05).
    상태가 바뀌었으면 즉시 rerun.
    """
    if st.session_state.get("current_run_status") != "running":
        return

    _drain_queue()

    new_status = st.session_state.get("current_run_status")
    if new_status == "running":
        time.sleep(1.0)
    st.rerun()


def _drain_queue() -> None:
    """큐에서 꺼낼 수 있는 메시지를 전부 처리한다."""
    q: queue.Queue | None = st.session_state.get("_result_queue")
    if q is None:
        return

    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            break

        msg_type = msg.get("type")

        if msg_type == "progress":
            st.session_state["_progress"] = {
                "step":    msg["step"],
                "total":   msg["total"],
                "loss":    msg.get("loss", 0.0),
                "elapsed": msg.get("elapsed", 0.0),
            }
            # loss_history: EfficientAD 차트용 (PatchCore는 loss=0.0이므로 무시)
            if msg.get("loss", 0.0) > 0.0:
                history = st.session_state.get("_loss_history", [])
                history.append({"step": msg["step"], "loss": msg["loss"]})
                st.session_state["_loss_history"] = history

        elif msg_type == "log":
            lines = st.session_state.get("_log_lines", [])
            lines.append(msg["message"])
            if len(lines) > _MAX_LOG_LINES:
                lines = lines[-_MAX_LOG_LINES:]
            st.session_state["_log_lines"] = lines

        elif msg_type == "completed":
            _handle_completed(msg)
            break   # 이후 메시지는 다음 rerun에서 없을 것이므로 종료

        elif msg_type == "stopped":
            _handle_stopped(msg)
            break

        elif msg_type == "error":
            _handle_error(msg)
            break


# ──────────────────────────────────────────────────────────────
# 터미널 메시지 핸들러
# ──────────────────────────────────────────────────────────────

def _handle_completed(msg: dict) -> None:
    """
    completed 메시지 처리:
      1. save_completed_experiment (3단계 저장 프로토콜)
      2. set_anomaly_map_cache (Z.6)
      3. session_state 업데이트
    """
    exp_id               = st.session_state.get("current_exp_id", "")
    model_config         = st.session_state.get("model_config", {})
    preprocessing_config = st.session_state.get("preprocessing_config", {})
    dataset_path         = st.session_state.get("dataset_path", "")

    record = {
        "experiment_id":       exp_id,
        "name":                exp_id,
        "model_type":          model_config.get("model_type", ""),
        "created_at":          datetime.now(KST).isoformat(),
        "dataset_path":        dataset_path,
        "model_config":        model_config,
        "preprocessing_config": preprocessing_config,
        "threshold":           msg.get("threshold", 0.0),
        "metrics":             msg.get("metrics", {}),
        "duration_seconds":    msg.get("duration_seconds", 0),
        "status":              "completed",
    }

    # Stage 1~3: 모델 저장 → configs.yaml → history.json
    try:
        save_completed_experiment(exp_id, msg["model"], record)
    except Exception as exc:
        st.error(f"모델 저장 실패: {exc}")

    # Anomaly Map LRU 캐시 (Z.6)
    set_anomaly_map_cache(
        exp_id=exp_id,
        data={
            "anomaly_maps": msg.get("anomaly_maps", {}),
            "image_paths":  msg.get("image_paths", []),
        },
    )

    experiments            = st.session_state.get("experiments", {})
    experiments[exp_id]    = record
    st.session_state["experiments"]         = experiments
    st.session_state["current_run_status"]  = "completed"
    st.session_state["_stop_event"]         = None
    st.session_state["_result_queue"]       = None


def _handle_stopped(msg: dict) -> None:
    step = msg.get("step", 0)
    st.session_state["current_run_status"] = "stopped"
    st.session_state["_stopped_step"]      = step
    st.session_state["_stop_event"]        = None
    st.session_state["_result_queue"]      = None


def _handle_error(msg: dict) -> None:
    st.session_state["_last_error"] = {
        "exception": str(msg.get("exception", "알 수 없는 오류")),
        "traceback": msg.get("traceback", ""),
    }
    st.session_state["current_run_status"] = "error"
    st.session_state["_stop_event"]        = None
    st.session_state["_result_queue"]      = None


# ──────────────────────────────────────────────────────────────
# 학습 시작 / 중단
# ──────────────────────────────────────────────────────────────

def _handle_start_training() -> None:
    from utils.model_factory import create_trainer

    model_config         = st.session_state.get("model_config", {})
    preprocessing_config = st.session_state.get("preprocessing_config", {})
    dataset_path         = st.session_state.get("dataset_path", "")
    device_info          = st.session_state.get("device_info") or {}
    device               = device_info.get("device", "cpu") if isinstance(device_info, dict) else "cpu"

    exp_id      = _generate_exp_id(model_config.get("model_type", "model"))
    stop_event  = threading.Event()
    result_queue = queue.Queue()

    worker = create_trainer(
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        dataset_path=dataset_path,
        device=device,
        experiment_id=exp_id,
        stop_event=stop_event,
        result_queue=result_queue,
    )
    worker.start()

    st.session_state["current_run_status"] = "running"
    st.session_state["current_exp_id"]     = exp_id
    st.session_state["_stop_event"]        = stop_event
    st.session_state["_result_queue"]      = result_queue
    st.session_state["_progress"]          = None
    st.session_state["_log_lines"]         = []
    st.session_state["_loss_history"]      = []


def _handle_stop_training() -> None:
    stop_event: threading.Event | None = st.session_state.get("_stop_event")
    if stop_event is not None:
        stop_event.set()


# ──────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────

def _generate_exp_id(model_type: str) -> str:
    """R-NAMING-03: {model_type}_{YYYYMMDD}_{HHMMSS}_{4자리 uuid4 hex}"""
    now      = datetime.now(KST)
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    uid      = uuid.uuid4().hex[:4]
    return f"{model_type}_{date_str}_{time_str}_{uid}"
