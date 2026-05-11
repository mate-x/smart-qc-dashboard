from __future__ import annotations

import queue
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from utils.cache_manager import set_anomaly_map_cache
from utils.config_manager import load_config, save_config_section
from utils.messages import MSG
from utils.metrics import compute_metrics, compute_threshold
from utils.storage import (
    append_experiment,
    check_disk_before_save,
    save_completed_experiment,
    validate_imagenet_penalty_dir,
    IMAGENET_PENALTY_DIR,
)
from utils.training_worker import TrainingWorker

KST = timezone(timedelta(hours=9))


# ── ID / 타임스탬프 생성 ───────────────────────────────────────────────────────

def generate_experiment_id(model_type: str) -> str:
    """R-NAMING-03: {model_type}_{YYYYMMDD}_{HHMMSS}_{4자리_소문자_16진수}"""
    now = datetime.now(tz=KST)
    return (
        f"{model_type}_{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}"
        f"_{uuid.uuid4().hex[:4]}"
    )


def generate_created_at() -> str:
    return datetime.now(tz=KST).isoformat()


# ── 탭 진입점 ─────────────────────────────────────────────────────────────────

def render() -> None:
    st.header("탭4. 학습 시작 + 학습 로그")

    if not _guard():
        return

    status = st.session_state.get("current_run_status", "idle")

    st.info(
        "학습 중 새로고침 시 학습 상태를 확인할 수 없습니다.",
        icon="⚠️",
    )

    if status == "running":
        finished = _drain_queue()
        _render_running_ui()
        if not finished:
            time.sleep(0.3)
            st.rerun()
    else:
        _show_last_result()
        _render_idle_ui()


def _show_last_result() -> None:
    """직전 학습 결과 메시지를 idle UI 위에 표시 (1회만)."""
    result = st.session_state.pop("_last_result", None)
    if not result:
        return
    level, text = result.get("level", "info"), result.get("text", "")
    if level == "success":
        st.success(text)
    elif level == "warning":
        st.warning(text)
    elif level == "error":
        st.error(text)


# ── Guard ──────────────────────────────────────────────────────────────────────

def _guard() -> bool:
    """3개 선행 조건 확인. 미아 스레드(orphan thread) 감지 포함."""
    worker = st.session_state.get("_worker")
    if worker is not None and worker.is_alive():
        q = st.session_state.get("_result_queue")
        if q is None:
            _reset_run_state()
            st.info(
                "새로고침으로 인해 학습 상태를 확인할 수 없습니다. "
                "새로 학습을 시작하거나 탭5에서 히스토리를 확인하세요."
            )

    missing = False
    if st.session_state.get("dataset_path") is None:
        st.warning(MSG["NO_DATASET"])
        missing = True
    if st.session_state.get("preprocessing_config") is None:
        st.warning(MSG["NO_PREPROCESSING"])
        missing = True
    if st.session_state.get("model_config") is None:
        st.warning(MSG["NO_MODEL_CONFIG"])
        missing = True
    return not missing


# ── Idle UI ────────────────────────────────────────────────────────────────────

def _render_idle_ui() -> None:
    """current_run_status == "idle" 상태 UI."""
    _render_pretrain_summary()

    experiment_name = st.text_input(
        "실험명 (비워두면 자동 생성)",
        max_chars=64,
        placeholder="예: EfficientAD CLAHE clip2.0 실험",
        key="tab4_experiment_name",
    )

    if st.button("학습 시작", type="primary"):
        _handle_start_training(experiment_name)


def _render_pretrain_summary() -> None:
    """학습 전 설정 요약 표시."""
    model_config: dict | None = st.session_state.get("model_config")
    preprocessing_config: dict | None = st.session_state.get("preprocessing_config")
    if not model_config or not preprocessing_config:
        return

    with st.expander("현재 학습 설정 요약", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**모델**: {model_config.get('model_type', '?').upper()}")
            st.markdown(f"**이미지 크기**: {model_config.get('image_size', '?')}")
            st.markdown(f"**배치 크기**: {model_config.get('batch_size', '?')}")
            st.markdown(f"**랜덤 시드**: {model_config.get('random_seed', '?')}")
        with col2:
            st.markdown(f"**전처리**: {preprocessing_config.get('method', 'none')}")
            st.markdown(
                f"**Threshold 방식**: "
                f"{model_config.get('threshold_method', '?')} "
                f"({model_config.get('threshold_value', '?')})"
            )
            device_info = st.session_state.get("device_info") or {}
            st.markdown(f"**디바이스**: {device_info.get('device', 'cpu').upper()}")


# ── Running UI ─────────────────────────────────────────────────────────────────

def _render_running_ui() -> None:
    """current_run_status == "running" 상태 UI."""
    st.info("🔄 학습이 진행 중입니다. 탭을 전환해도 학습은 계속됩니다.")

    progress = st.session_state.get("_progress") or {}
    step = progress.get("step", 0)
    total = progress.get("total", 1)
    loss = progress.get("loss")

    start_t = st.session_state.get("_training_start_time") or time.time()
    elapsed = time.time() - start_t  # 항상 실시간 벽시계 기준

    pct = step / total if total > 0 else 0.0
    label = f"Step {step:,} / {total:,} ({pct*100:.1f}%)"
    if loss is not None:
        label += f" | Loss: {loss:.4f}"
    label += f" | 경과: {elapsed:.0f}s"
    st.progress(pct, text=label)

    loss_history = st.session_state.get("_loss_history") or []
    if loss_history:
        df = pd.DataFrame(loss_history)
        fig = px.line(df, x="step", y="loss", title="학습 Loss 곡선")
        fig.update_layout(height=250, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    log_lines = st.session_state.get("_log_lines") or []
    log_text = "\n".join(log_lines[-50:])
    st.text_area("학습 로그", value=log_text, height=200, disabled=True, key="tab4_log_area")

    if st.button("학습 중지", type="secondary"):
        stop_event: threading.Event | None = st.session_state.get("_stop_event")
        if stop_event:
            stop_event.set()
        st.info("중지 신호를 전송했습니다. 현재 스텝 완료 후 중단됩니다.")


# ── 학습 시작 핸들러 ──────────────────────────────────────────────────────────

def _handle_start_training(experiment_name: str) -> None:
    """[학습 시작] 버튼 클릭 시 호출."""
    if st.session_state.get("current_run_status") != "idle":
        st.warning("이미 학습이 진행 중입니다.")
        st.stop()

    model_config: dict = st.session_state["model_config"]
    preprocessing_config: dict = st.session_state["preprocessing_config"]
    dataset_path: str = st.session_state["dataset_path"]
    device_info: dict = st.session_state.get("device_info") or {"device": "cpu"}

    try:
        check_disk_before_save(model_config["model_type"])
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    if model_config.get("model_type") == "efficientad":
        penalty_weight = model_config.get("params", {}).get("imagenet_penalty_weight", 1.0)
        if penalty_weight > 0:
            ok, count = validate_imagenet_penalty_dir()
            if not ok:
                st.error(
                    f"EfficientAD 학습에 필요한 ImageNet penalty 데이터가 없습니다. "
                    f"`{IMAGENET_PENALTY_DIR}` 경로에 이미지를 추가해 주세요."
                )
                st.stop()
            elif count < 1000:
                st.warning(f"ImageNet penalty 이미지가 {count}장입니다. 1,000장 이상 권장합니다.")

    exp_id = generate_experiment_id(model_config["model_type"])
    created_at = generate_created_at()

    if not experiment_name.strip():
        experiment_name = f"{model_config['model_type'].upper()} {exp_id[-4:]}"

    save_config_section(
        section="experiment",
        data={"name": experiment_name, "created_at": created_at},
        path="./configs.yaml",
    )

    stop_event = threading.Event()
    result_queue: queue.Queue = queue.Queue()

    worker = TrainingWorker(
        experiment_id=exp_id,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        dataset_path=dataset_path,
        device=device_info.get("device", "cpu"),
        stop_event=stop_event,
        result_queue=result_queue,
    )
    worker.daemon = True
    worker.start()

    st.session_state["current_run_status"] = "running"
    st.session_state["current_exp_id"] = exp_id
    st.session_state["_stop_event"] = stop_event
    st.session_state["_result_queue"] = result_queue
    st.session_state["_worker"] = worker
    st.session_state["_progress"] = {
        "step": 0,
        "total": _get_total_steps(model_config),
        "loss": None,
        "elapsed": 0.0,
    }
    st.session_state["_log_lines"] = []
    st.session_state["_loss_history"] = []
    st.session_state["_training_start_time"] = time.time()

    st.rerun()


def _get_total_steps(model_config: dict) -> int:
    if model_config.get("model_type") == "efficientad":
        return model_config["params"].get("train_steps", 70000)
    return 1  # PatchCore


# ── Queue 드레인 ───────────────────────────────────────────────────────────────

def _drain_queue() -> bool:
    """Queue에 쌓인 메시지를 모두 소비. 종료 메시지 수신 시 True 반환."""
    q: queue.Queue | None = st.session_state.get("_result_queue")
    if q is None:
        return False

    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            break

        msg_type = msg.get("type")

        if msg_type == "progress":
            _handle_progress(msg)
        elif msg_type == "log":
            _handle_log(msg)
        elif msg_type == "completed":
            _handle_completed(msg)
            return True
        elif msg_type == "error":
            _handle_error(msg)
            return True
        elif msg_type == "stopped":
            _handle_stopped(msg)
            return True
    return False


# ── 메시지 핸들러 ──────────────────────────────────────────────────────────────

def _handle_progress(msg: dict) -> None:
    st.session_state["_progress"] = {
        "step":    msg["step"],
        "total":   msg["total"],
        "loss":    msg["loss"],
        "elapsed": msg["elapsed"],
    }
    loss_history: list = st.session_state["_loss_history"]
    loss_history.append({"step": msg["step"], "loss": msg["loss"]})


def _handle_log(msg: dict) -> None:
    ts = datetime.now(tz=KST).strftime("%H:%M:%S")
    line = f"[{ts}] {msg['message']}"
    lines: list = st.session_state["_log_lines"]
    lines.append(line)
    if len(lines) > 100:
        st.session_state["_log_lines"] = lines[-100:]
    else:
        st.session_state["_log_lines"] = lines


def _handle_completed(msg: dict) -> None:
    """
    1. compute_threshold + compute_metrics
    2. experiment_record 구성
    3. save_completed_experiment (3단계 저장)
    4. session_state.experiments 갱신
    5. anomaly_map 캐시 저장
    6. st.success()
    """
    exp_id: str = st.session_state["current_exp_id"]
    model_config: dict = st.session_state["model_config"]

    y_true = msg["y_true"]
    anomaly_scores = msg["anomaly_scores"]

    normal_scores = [s for s, lbl in zip(anomaly_scores, y_true) if lbl == 0]
    if normal_scores:
        threshold = compute_threshold(
            np.array(normal_scores, dtype=np.float32),
            model_config.get("threshold_method", "percentile"),
            float(model_config.get("threshold_value", 95.0)),
        )
    else:
        threshold = float(model_config.get("threshold_value", 0.5))

    metrics = compute_metrics(y_true, anomaly_scores, threshold)

    record = _build_experiment_record(
        exp_id=exp_id,
        status="completed",
        metrics=metrics,
        duration_seconds=msg.get("duration_seconds"),
    )

    try:
        check_disk_before_save(model_config["model_type"])
        save_completed_experiment(
            exp_id,
            msg["model"],
            record,
            preprocessing_config=st.session_state.get("preprocessing_config"),
            model_config=model_config,
        )
        if "experiments" not in st.session_state:
            st.session_state["experiments"] = {}
        st.session_state["experiments"][exp_id] = record

        anomaly_maps_dict: dict = msg.get("anomaly_maps", {})
        image_paths: list[str] = msg.get("image_paths", [])
        if image_paths and anomaly_maps_dict:
            maps_array = np.stack(
                [anomaly_maps_dict[p] for p in image_paths], axis=0
            )
            set_anomaly_map_cache(
                exp_id,
                {"anomaly_maps": maps_array, "image_paths": image_paths},
            )

        secs = msg.get("duration_seconds", 0)
        mins, sec = divmod(secs, 60)
        auc = metrics.get("auc", 0.0)
        st.session_state["_last_result"] = {
            "level": "success",
            "text": f"학습이 완료되었습니다. AUC: {auc:.4f} | 소요 시간: {mins}분 {sec}초",
        }

    except RuntimeError as e:
        err_str = str(e)
        if "ERR_HISTORY_WRITE_FAILED" in err_str:
            st.session_state["_last_result"] = {
                "level": "warning",
                "text": f"모델 파일은 저장되었으나 히스토리 기록에 실패했습니다. {err_str}",
            }
        else:
            st.session_state["_last_result"] = {
                "level": "error",
                "text": f"모델 저장에 실패했습니다. 디스크 공간을 확인해 주세요. {err_str}",
            }
    finally:
        _reset_run_state()
        del msg["model"]
        if st.session_state.get("device_info", {}).get("device") == "cuda":
            torch.cuda.empty_cache()


def _handle_error(msg: dict) -> None:
    tb = msg.get("traceback", "")
    st.session_state["_last_result"] = {
        "level": "error",
        "text": f"학습 중 오류가 발생했습니다.\n{tb}",
    }
    _reset_run_state()


def _handle_stopped(msg: dict) -> None:
    """status="중단" 레코드 생성 후 history.json append."""
    exp_id: str = st.session_state.get("current_exp_id", "")
    step = msg.get("step", 0)

    if exp_id:
        record = _build_experiment_record(
            exp_id=exp_id,
            status="중단",
            metrics=None,
            duration_seconds=None,
        )
        try:
            append_experiment(record)
            if "experiments" not in st.session_state:
                st.session_state["experiments"] = {}
            st.session_state["experiments"][exp_id] = record
        except RuntimeError:
            pass

    st.session_state["_last_result"] = {
        "level": "warning",
        "text": MSG["TRAIN_STOPPED"] + (f" ({step:,} step 완료 후 중단)" if step else ""),
    }
    _reset_run_state()


def _reset_run_state() -> None:
    """학습 종료 후 내부 상태 초기화."""
    st.session_state["current_run_status"] = "idle"
    st.session_state["current_exp_id"] = None
    st.session_state["_stop_event"] = None
    st.session_state["_result_queue"] = None
    st.session_state["_worker"] = None


# ── 실험 레코드 구성 ───────────────────────────────────────────────────────────

def _build_experiment_record(
    exp_id: str,
    status: str,
    metrics: dict | None,
    duration_seconds: int | None,
) -> dict:
    """00_Global §1.1 experiment 스키마에 맞는 레코드 생성."""
    model_config: dict = st.session_state["model_config"]
    preprocessing_config: dict = st.session_state["preprocessing_config"]
    dataset_path: str = st.session_state["dataset_path"]

    exp_cfg = load_config("./configs.yaml").get("experiment", {})

    record: dict = {
        "experiment_id":        exp_id,
        "name":                 exp_cfg.get("name", exp_id),
        "status":               status,
        "created_at":           exp_cfg.get("created_at", generate_created_at()),
        "model_type":           model_config["model_type"],
        "preprocessing_method": preprocessing_config.get("method", "none"),
        "preprocessing_params": preprocessing_config.get("params"),
        "model_params":         model_config.get("params", {}),
        "threshold_method":     model_config.get("threshold_method", "percentile"),
        "threshold_value":      model_config.get("threshold_value", 95.0),
        "dataset_path":         dataset_path,
        "image_size":           model_config.get("image_size", 256),
        "duration_seconds":     duration_seconds,
        "metrics":              metrics,
        "model_path":           None,
        "configs_path":         None,
    }

    if status == "중단":
        record["metrics"] = None
        record["model_path"] = None
        record["configs_path"] = None

    return record
