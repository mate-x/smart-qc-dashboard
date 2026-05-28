from __future__ import annotations

import queue
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from utils.cache_manager import set_anomaly_map_cache
from utils.checkpoint_manager import delete_checkpoint, list_checkpoints, load_checkpoint
from utils.config_manager import load_config, save_config_section
from utils.messages import MSG
from utils.metrics import compute_metrics, compute_threshold
from utils.storage import (
    append_experiment,
    check_disk_before_save,
    load_history,
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

    if status == "running":
        st.info("학습 중 새로고침 시 학습 상태를 확인할 수 없습니다.", icon="⚠️")
        finished = _drain_queue()
        _render_running_ui()
        if not finished:
            time.sleep(0.3)
        st.rerun()

    elif status == "paused":
        st.info("학습 중 새로고침 시 학습 상태를 확인할 수 없습니다.", icon="⚠️")
        _render_running_ui()

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

    # ── 체크포인트 재시작 섹션
    _render_checkpoint_resume_section()


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


def _render_checkpoint_resume_section() -> None:
    """저장된 체크포인트 목록을 보여주고 재시작 버튼을 제공."""
    ckpt_list = list_checkpoints()
    if not ckpt_list:
        return

    st.divider()
    with st.expander(f"⏩ 체크포인트에서 재시작 ({len(ckpt_list)}개 저장됨)", expanded=False):
        names = [p.name for p in ckpt_list]
        sel_name = st.selectbox("체크포인트 선택", names, key="tab4_ckpt_select")

        # 선택된 체크포인트 정보 표시
        sel_path = next(p for p in ckpt_list if p.name == sel_name)
        try:
            ckpt_meta = load_checkpoint(sel_path)
            model_type = ckpt_meta.get("model_type", "?")
            created_at = ckpt_meta.get("created_at", "?")

            if model_type == "efficientad":
                step        = ckpt_meta.get("step", 0)
                total_steps = ckpt_meta.get("total_steps", "?")
                pct         = f"{step / total_steps * 100:.1f}%" if isinstance(total_steps, int) and total_steps > 0 else "?"
                st.caption(
                    f"모델: **EfficientAD** | "
                    f"Step: {step:,} / {total_steps:,} ({pct}) | "
                    f"저장: {created_at}"
                )
            elif model_type == "patchcore":
                batch_idx     = ckpt_meta.get("batch_idx", 0)
                total_batches = ckpt_meta.get("total_batches", "?")
                pct           = f"{batch_idx / total_batches * 100:.1f}%" if isinstance(total_batches, int) and total_batches > 0 else "?"
                feat = ckpt_meta.get("accumulated_features")
                n_patches = feat.shape[0] if feat is not None and hasattr(feat, "shape") else 0
                st.caption(
                    f"모델: **PatchCore** | "
                    f"Batch: {batch_idx} / {total_batches} ({pct}) | "
                    f"추출 패치: {n_patches:,}개 | "
                    f"저장: {created_at}"
                )
            else:
                st.caption(f"모델: {model_type} | 저장: {created_at}")
        except Exception:
            st.caption("체크포인트 정보를 읽을 수 없습니다.")
            ckpt_meta = None

        col1, col2 = st.columns([3, 1])
        with col1:
            resume_btn = st.button(
                "▶ 이 체크포인트에서 재시작",
                type="primary",
                use_container_width=True,
                key="tab4_resume_btn",
            )
        with col2:
            delete_btn = st.button(
                "🗑 삭제",
                use_container_width=True,
                key="tab4_ckpt_delete_btn",
            )

        if resume_btn and ckpt_meta is not None:
            _handle_resume_training(sel_path, ckpt_meta)

        if delete_btn:
            if delete_checkpoint(sel_path):
                st.success(f"`{sel_name}` 삭제 완료.")
                st.rerun()
            else:
                st.error("삭제에 실패했습니다.")


# ── Running / Paused UI ────────────────────────────────────────────────────────

def _render_running_ui() -> None:
    """current_run_status == "running" 또는 "paused" 상태 UI."""
    status = st.session_state.get("current_run_status", "running")

    st.info("🔄 학습이 진행 중입니다. 탭을 전환해도 학습은 계속됩니다.")

    progress = st.session_state.get("_progress") or {}
    step     = progress.get("step", 0)
    total    = progress.get("total", 1)
    loss     = progress.get("loss")

    start_t = st.session_state.get("_training_start_time") or time.time()
    elapsed = time.time() - start_t

    pct   = step / total if total > 0 else 0.0
    if status == "paused":
        ckpt_path = st.session_state.get("_last_ckpt_path")
        ckpt_info = f" | 저장: {Path(ckpt_path).name}" if ckpt_path else ""
        label = f"⏸ 일시정지 | Step {step:,} / {total:,} ({pct*100:.1f}%) | 경과: {elapsed:.0f}s{ckpt_info}"
    else:
        label = f"Step {step:,} / {total:,} ({pct*100:.1f}%)"
        if loss is not None and loss > 0:
            label += f" | Loss: {loss:.4f}"
        label += f" | 경과: {elapsed:.0f}s"
    st.progress(pct, text=label)

    # Loss 곡선 (EfficientAD만 유효, PatchCore는 loss=0)
    loss_history = st.session_state.get("_loss_history") or []
    valid_history = [
        h for h in loss_history
        if h.get("loss") is not None and np.isfinite(h["loss"]) and h["loss"] > 0
    ]
    if valid_history:
        df  = pd.DataFrame(valid_history)
        fig = px.line(df, x="step", y="loss", title="학습 Loss 곡선", markers=True)
        fig.update_layout(height=250, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    elif loss_history and loss is not None and loss > 0:
        st.warning("Loss 값이 모두 NaN입니다. 학습률을 낮춰 주세요.")

    log_lines = st.session_state.get("_log_lines") or []
    log_text  = "\n".join(log_lines[-50:])
    st.text_area("학습 로그", value=log_text, height=200, disabled=True, key="tab4_log_area")

    # ── 제어 버튼
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⏸ 일시정지", disabled=(status != "running"), use_container_width=True, key="tab4_pause_btn"):
            pause_ev: threading.Event | None = st.session_state.get("_pause_event")
            if pause_ev:
                pause_ev.set()

    with col2:
        if st.button("▶ 재시작", disabled=(status != "paused"), use_container_width=True, key="tab4_resume_ctrl_btn"):
            pause_ev: threading.Event | None = st.session_state.get("_pause_event")
            if pause_ev:
                pause_ev.clear()
            st.session_state["current_run_status"] = "running"
            st.rerun()

    with col3:
        if st.button("⏹ 학습 중지", type="secondary", use_container_width=True, key="tab4_stop_btn"):
            stop_ev: threading.Event | None = st.session_state.get("_stop_event")
            if stop_ev:
                stop_ev.set()
            pause_ev = st.session_state.get("_pause_event")
            if pause_ev:
                pause_ev.clear()
            st.info("중지 신호를 전송했습니다. 현재 스텝 완료 후 중단됩니다.")


# ── 학습 시작 핸들러 ──────────────────────────────────────────────────────────

def _handle_start_training(experiment_name: str) -> None:
    """[학습 시작] 버튼 클릭 시 호출."""
    if st.session_state.get("current_run_status") != "idle":
        st.warning("이미 학습이 진행 중입니다.")
        st.stop()

    model_config: dict         = st.session_state["model_config"]
    preprocessing_config: dict = st.session_state["preprocessing_config"]
    dataset_path: str          = st.session_state["dataset_path"]
    device_info: dict          = st.session_state.get("device_info") or {"device": "cpu"}

    try:
        check_disk_before_save(model_config["model_type"])
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    if model_config.get("model_type") == "efficientad":
        use_imagenet_penalty = model_config.get("params", {}).get("use_imagenet_penalty", False)
        if use_imagenet_penalty:
            ok, count = validate_imagenet_penalty_dir()
            if not ok:
                st.error(
                    f"EfficientAD 학습에 필요한 ImageNet penalty 데이터가 없습니다. "
                    f"`{IMAGENET_PENALTY_DIR}` 경로에 이미지를 추가해 주세요."
                )
                st.stop()
            elif count < 1000:
                st.warning(f"ImageNet penalty 이미지가 {count}장입니다. 1,000장 이상 권장합니다.")

    exp_id     = generate_experiment_id(model_config["model_type"])
    created_at = generate_created_at()

    if not experiment_name.strip():
        experiment_name = f"{model_config['model_type'].upper()} {exp_id[-4:]}"

    save_config_section(
        section="experiment",
        data={"name": experiment_name, "created_at": created_at},
        path="./configs.yaml",
    )

    stop_event  = threading.Event()
    pause_event = threading.Event()
    result_queue: queue.Queue = queue.Queue()

    worker = TrainingWorker(
        experiment_id=exp_id,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        dataset_path=dataset_path,
        device=device_info.get("device", "cpu"),
        stop_event=stop_event,
        result_queue=result_queue,
        pause_event=pause_event,
    )
    worker.daemon = True
    worker.start()

    st.session_state["current_run_status"]  = "running"
    st.session_state["current_exp_id"]      = exp_id
    st.session_state["_stop_event"]         = stop_event
    st.session_state["_pause_event"]        = pause_event
    st.session_state["_result_queue"]       = result_queue
    st.session_state["_worker"]             = worker
    st.session_state["_last_ckpt_path"]     = None
    st.session_state["_progress"]           = {
        "step": 0, "total": _get_total_steps(model_config), "loss": None, "elapsed": 0.0,
    }
    st.session_state["_log_lines"]          = []
    st.session_state["_loss_history"]       = []
    st.session_state["_training_start_time"] = time.time()

    st.rerun()


def _handle_resume_training(ckpt_path: "Path", ckpt: dict) -> None:
    """체크포인트에서 TrainingWorker를 재시작."""
    if st.session_state.get("current_run_status") != "idle":
        st.warning("이미 학습이 진행 중입니다.")
        return

    model_config         = ckpt["model_config"]
    preprocessing_config = ckpt["preprocessing_config"]
    dataset_path         = ckpt["dataset_path"]
    model_type           = ckpt["model_type"]
    device_info          = st.session_state.get("device_info") or {"device": "cpu"}

    # 기존 exp_id가 history에 있으면 충돌 방지를 위해 새 ID 생성
    old_exp_id   = ckpt["experiment_id"]
    existing_ids = {r.get("experiment_id") for r in load_history()}
    exp_id       = old_exp_id if old_exp_id not in existing_ids else generate_experiment_id(model_type)

    if exp_id != old_exp_id:
        st.info(
            f"체크포인트의 실험 ID(`{old_exp_id}`)가 이미 기록에 존재합니다. "
            f"새 ID(`{exp_id}`)로 재시작합니다."
        )

    # session_state에 탭3 설정 복원 (다른 탭 미설정 상태에서 resume 허용)
    st.session_state["model_config"]         = model_config
    st.session_state["preprocessing_config"] = preprocessing_config
    st.session_state["dataset_path"]         = dataset_path

    save_config_section(
        section="experiment",
        data={"name": f"Resume {exp_id[-4:]}", "created_at": generate_created_at()},
        path="./configs.yaml",
    )

    stop_event   = threading.Event()
    pause_event  = threading.Event()
    result_queue: queue.Queue = queue.Queue()

    # 모델 타입별 resume 파라미터 추출
    worker_kwargs: dict = {}
    if model_type == "efficientad":
        worker_kwargs = {
            "start_step":               ckpt.get("step", 0),
            "student_state_dict":       ckpt.get("student_state_dict"),
            "autoencoder_state_dict":   ckpt.get("autoencoder_state_dict"),
            "optimizer_st_state_dict":  ckpt.get("optimizer_st_state_dict"),
            "optimizer_ae_state_dict":  ckpt.get("optimizer_ae_state_dict"),
            "scheduler_st_state_dict":  ckpt.get("scheduler_st_state_dict"),
            "scheduler_ae_state_dict":  ckpt.get("scheduler_ae_state_dict"),
            "loss_history":             ckpt.get("loss_history", []),
        }
    elif model_type == "patchcore":
        worker_kwargs = {
            "start_batch_idx":     ckpt.get("batch_idx", 0),
            "accumulated_features": ckpt.get("accumulated_features"),
        }

    worker = TrainingWorker(
        experiment_id=exp_id,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        dataset_path=dataset_path,
        device=device_info.get("device", "cpu"),
        stop_event=stop_event,
        result_queue=result_queue,
        pause_event=pause_event,
        **worker_kwargs,
    )
    worker.daemon = True
    worker.start()

    # 진행 표시 초기값 (이전 체크포인트 기준)
    if model_type == "efficientad":
        init_step  = ckpt.get("step", 0)
        init_total = ckpt.get("total_steps", _get_total_steps(model_config))
    else:
        init_step  = ckpt.get("batch_idx", 0)
        init_total = ckpt.get("total_batches", 1)

    st.session_state["current_run_status"]   = "running"
    st.session_state["current_exp_id"]       = exp_id
    st.session_state["_stop_event"]          = stop_event
    st.session_state["_pause_event"]         = pause_event
    st.session_state["_result_queue"]        = result_queue
    st.session_state["_worker"]              = worker
    st.session_state["_last_ckpt_path"]      = str(ckpt_path)
    st.session_state["_progress"]            = {
        "step": init_step, "total": init_total, "loss": None, "elapsed": 0.0,
    }
    st.session_state["_log_lines"]           = []
    st.session_state["_loss_history"]        = list(ckpt.get("loss_history", []))
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
        elif msg_type == "paused":
            _handle_paused(msg)
            # paused는 종료가 아니므로 계속 드레인
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
    loss_val = msg.get("loss")
    if loss_val is not None and loss_val > 0:
        loss_history: list = st.session_state["_loss_history"]
        loss_history.append({"step": msg["step"], "loss": loss_val})


def _handle_log(msg: dict) -> None:
    ts   = datetime.now(tz=KST).strftime("%H:%M:%S")
    line = f"[{ts}] {msg['message']}"
    lines: list = st.session_state["_log_lines"]
    lines.append(line)
    st.session_state["_log_lines"] = lines[-100:]


def _handle_paused(msg: dict) -> None:
    """background thread가 체크포인트 저장 후 전송하는 메시지."""
    st.session_state["current_run_status"] = "paused"
    st.session_state["_last_ckpt_path"]    = msg.get("ckpt_path")


def _handle_completed(msg: dict) -> None:
    """
    1. compute_threshold + compute_metrics
    2. experiment_record 구성
    3. save_completed_experiment (3단계 저장)
    4. session_state.experiments 갱신
    5. anomaly_map 캐시 저장
    """
    exp_id: str       = st.session_state["current_exp_id"]
    model_config: dict = st.session_state["model_config"]

    y_true        = msg["y_true"]
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
        image_paths: list[str]  = msg.get("image_paths", [])
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
            "text":  f"학습이 완료되었습니다. AUC: {auc:.4f} | 소요 시간: {mins}분 {sec}초",
        }

    except RuntimeError as e:
        err_str = str(e)
        if "ERR_HISTORY_WRITE_FAILED" in err_str:
            st.session_state["_last_result"] = {
                "level": "warning",
                "text":  f"모델 파일은 저장되었으나 히스토리 기록에 실패했습니다. {err_str}",
            }
        else:
            st.session_state["_last_result"] = {
                "level": "error",
                "text":  f"모델 저장에 실패했습니다. 디스크 공간을 확인해 주세요. {err_str}",
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
        "text":  f"학습 중 오류가 발생했습니다.\n{tb}",
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
        "text":  MSG["TRAIN_STOPPED"] + (f" ({step:,} step 완료 후 중단)" if step else ""),
    }
    _reset_run_state()


def _reset_run_state() -> None:
    """학습 종료 후 내부 상태 초기화."""
    st.session_state["current_run_status"] = "idle"
    st.session_state["current_exp_id"]     = None
    st.session_state["_stop_event"]        = None
    st.session_state["_pause_event"]       = None
    st.session_state["_result_queue"]      = None
    st.session_state["_worker"]            = None
    st.session_state["_last_ckpt_path"]    = None


# ── 실험 레코드 구성 ───────────────────────────────────────────────────────────

def _build_experiment_record(
    exp_id: str,
    status: str,
    metrics: dict | None,
    duration_seconds: int | None,
) -> dict:
    """00_Global §1.1 experiment 스키마에 맞는 레코드 생성."""
    model_config: dict         = st.session_state["model_config"]
    preprocessing_config: dict = st.session_state["preprocessing_config"]
    dataset_path: str          = st.session_state["dataset_path"]

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
        record["metrics"]      = None
        record["model_path"]   = None
        record["configs_path"] = None

    return record
