"""
api/explorer/services/training_service.py

탭3 · Training:
    단일 학습: start_training / resume_training / pause_training / unpause_training / stop_training
    배치 학습: start_batch / skip_batch_item / stop_batch_all
    조회: get_status / get_checkpoints / remove_checkpoint
    내부: result_queue 폴링 루프 (asyncio Task) + WS 브로드캐스트 큐

WS 브로드캐스트 메시지 타입:
    progress / log / stage        — TrainingWorker 원본 relay
    paused                        — {type, step, ckpt_path}
    completed                     — {type, exp_id, auc, duration_seconds, message}
    stopped                       — {type, step}
    error                         — {type, message, traceback}
    batch_item_started            — {type, exp_id, queue_idx}
    batch_item_skipped            — {type}
    batch_item_error              — {type, traceback}
    batch_stopped                 — {type, step}
    batch_completed               — {type, completed, failed, skipped}
"""
from __future__ import annotations

import asyncio
import queue as _queue
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from api.explorer.state import get_state
from utils.checkpoint_manager import delete_checkpoint, list_checkpoints, load_checkpoint
from utils.metrics import compute_metrics, compute_threshold
from utils.storage import append_experiment, check_disk_space, save_completed_experiment
from utils.training_worker import TrainingWorker

KST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------------------
# In-memory 학습 상태
# ---------------------------------------------------------------------------

_run: dict = {
    "status":             "idle",   # "idle" | "running" | "paused"
    "exp_id":             None,
    "experiment_name":    None,
    "created_at":         None,
    "stop_event":         None,
    "pause_event":        None,
    "result_queue":       None,
    "worker":             None,
    # Progress (GET /api/training/status + WS 재연결 복구용)
    "progress":           None,     # {step, total, loss, elapsed}
    "current_stage_idx":  None,
    "current_stage_name": None,
    "log_lines":          [],       # 최근 100줄
    "loss_history":       [],
    "last_ckpt_path":     None,
    "model_type":         None,
    "model_config":       None,
    "preprocessing_config": None,
    # 배치 상태 (배치 흐름이 직접 관리 — _reset_run_state에서 초기화 안 함)
    "batch_mode":         False,
    "batch_total":        0,
    "batch_skip_current": False,
    "batch_advance_pending": False,
    "batch_stopping":     False,    # stop_batch_all() 진행 중 플래그
    "set_id":             None,     # 현재 학습 중인 큐 항목의 set_id (배치 자동 실험 세트 식별자)
    "current_batch_indices": [],
    # 내부
    "_poll_task":         None,
}

# WebSocket 브로드캐스트 큐 (Part 4에서 register_ws_queue로 연결)
_ws_queue: asyncio.Queue | None = None


def register_ws_queue(q: asyncio.Queue) -> None:
    global _ws_queue
    _ws_queue = q


def unregister_ws_queue() -> None:
    global _ws_queue
    _ws_queue = None


async def _broadcast(msg: dict) -> None:
    if _ws_queue is not None:
        await _ws_queue.put(msg)


# ---------------------------------------------------------------------------
# ID / 타임스탬프
# ---------------------------------------------------------------------------

def _generate_experiment_id(model_type: str) -> str:
    now = datetime.now(tz=KST)
    return (
        f"{model_type}_{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}"
        f"_{uuid.uuid4().hex[:4]}"
    )


def _generate_created_at() -> str:
    return datetime.now(tz=KST).isoformat()


def _get_total_steps(model_config: dict) -> int:
    if model_config.get("model_type") == "efficientad":
        return model_config.get("params", {}).get("train_steps", 70000)
    return 1  # PatchCore는 배치 수 기준 — 초기값 1


# ---------------------------------------------------------------------------
# Public — 상태 조회
# ---------------------------------------------------------------------------

def get_status() -> dict:
    return {
        "status":             _run["status"],
        "exp_id":             _run["exp_id"],
        "batch_mode":         _run["batch_mode"],
        "batch_total":        _run["batch_total"],
        "progress":           _run["progress"],
        "current_stage_idx":  _run["current_stage_idx"],
        "current_stage_name": _run["current_stage_name"],
        "log_lines":          list(_run["log_lines"]),
        "loss_history":       list(_run["loss_history"]),
        "last_ckpt_path":     _run["last_ckpt_path"],
        "model_type":         _run.get("model_type"),
    }


# ---------------------------------------------------------------------------
# Public — 체크포인트
# ---------------------------------------------------------------------------

def get_checkpoints() -> list[dict]:
    result: list[dict] = []
    for ckpt_path in list_checkpoints():
        item: dict = {"name": ckpt_path.name, "model_type": "", "created_at": ""}
        try:
            meta = load_checkpoint(ckpt_path)
            item["model_type"] = meta.get("model_type", "")
            item["created_at"] = meta.get("created_at", "")
            if item["model_type"] == "efficientad":
                item["step"]        = meta.get("step")
                item["total_steps"] = meta.get("total_steps")
            elif item["model_type"] == "patchcore":
                item["batch_idx"]    = meta.get("batch_idx")
                item["total_batches"] = meta.get("total_batches")
                feat = meta.get("accumulated_features")
                item["n_patches"] = (
                    int(feat.shape[0]) if feat is not None and hasattr(feat, "shape") else None
                )
        except Exception:
            pass
        result.append(item)
    return result


def remove_checkpoint(name: str) -> bool:
    target = next((p for p in list_checkpoints() if p.name == name), None)
    if target is None:
        raise LookupError(f"체크포인트를 찾을 수 없습니다: {name}")
    return delete_checkpoint(target)


# ---------------------------------------------------------------------------
# Public — 단일 학습
# ---------------------------------------------------------------------------

def start_training(experiment_name: str) -> str:
    """새 학습 시작. exp_id 반환."""
    if _run["status"] != "idle":
        raise RuntimeError("이미 학습이 진행 중입니다.")

    state = get_state()
    model_config        = state["model_config"]
    preprocessing_config = state["preprocessing_config"]
    dataset_path        = state["dataset_path"]

    if model_config is None:
        raise ValueError("model_config가 설정되지 않았습니다. 탭2에서 설정을 저장하세요.")
    if preprocessing_config is None:
        raise ValueError("preprocessing_config가 설정되지 않았습니다. 탭2에서 설정을 저장하세요.")
    if not dataset_path:
        raise ValueError("dataset_path가 설정되지 않았습니다. 탭1에서 데이터셋을 검증하세요.")

    ok, free_mb = check_disk_space(required_mb=100.0)
    if not ok:
        raise RuntimeError(
            f"ERR_DISK_SPACE: 디스크 여유 공간이 부족합니다 ({free_mb:.0f} MB). "
            "모델 저장에 최소 100 MB가 필요합니다."
        )

    if model_config.get("model_type") == "efficientad":
        use_imagenet = model_config.get("params", {}).get("use_imagenet_penalty", False)
        if use_imagenet:
            from utils.storage import validate_imagenet_penalty_dir
            ok, _ = validate_imagenet_penalty_dir()
            if not ok:
                raise ValueError("ImageNet penalty 디렉터리에 이미지가 없습니다.")

    model_type = model_config["model_type"]
    exp_id     = _generate_experiment_id(model_type)
    created_at = _generate_created_at()
    name       = experiment_name.strip() or f"{model_type.upper()} {exp_id[-4:]}"
    device     = (state.get("device_info") or {}).get("device", "cpu")

    _start_worker(
        exp_id=exp_id,
        experiment_name=name,
        created_at=created_at,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        dataset_path=dataset_path,
        device=device,
    )
    return exp_id, _run["model_type"]


def resume_training(checkpoint_name: str) -> str:
    """체크포인트에서 재시작. exp_id 반환."""
    if _run["status"] != "idle":
        raise RuntimeError("이미 학습이 진행 중입니다.")

    ckpt_path = next((p for p in list_checkpoints() if p.name == checkpoint_name), None)
    if ckpt_path is None:
        raise LookupError(f"체크포인트를 찾을 수 없습니다: {checkpoint_name}")

    ckpt                 = load_checkpoint(ckpt_path)
    model_config         = ckpt["model_config"]
    preprocessing_config = ckpt["preprocessing_config"]
    dataset_path         = ckpt["dataset_path"]
    model_type           = ckpt["model_type"]

    from utils.storage import load_history
    old_exp_id    = ckpt["experiment_id"]
    existing_ids  = {r.get("experiment_id") for r in load_history()}
    exp_id        = old_exp_id if old_exp_id not in existing_ids else _generate_experiment_id(model_type)
    created_at    = _generate_created_at()
    device        = (get_state().get("device_info") or {}).get("device", "cpu")

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
        init_progress = {
            "step":    ckpt.get("step", 0),
            "total":   ckpt.get("total_steps", model_config.get("params", {}).get("train_steps", 70000)),
            "loss":    None,
            "elapsed": 0.0,
        }
        init_loss_history = list(ckpt.get("loss_history", []))
    else:  # patchcore
        worker_kwargs = {
            "start_batch_idx":      ckpt.get("batch_idx", 0),
            "accumulated_features": ckpt.get("accumulated_features"),
        }
        init_progress = {
            "step":    ckpt.get("batch_idx", 0),
            "total":   ckpt.get("total_batches", 1),
            "loss":    None,
            "elapsed": 0.0,
        }
        init_loss_history = []

    # 체크포인트 설정을 현재 state에 복원
    state = get_state()
    state["model_config"]         = model_config
    state["preprocessing_config"] = preprocessing_config
    state["dataset_path"]         = dataset_path

    _start_worker(
        exp_id=exp_id,
        experiment_name=f"Resume {exp_id[-4:]}",
        created_at=created_at,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        dataset_path=dataset_path,
        device=device,
        init_progress=init_progress,
        init_loss_history=init_loss_history,
        **worker_kwargs,
    )
    return exp_id


def pause_training() -> None:
    if _run["status"] != "running":
        raise RuntimeError("실행 중인 학습이 없습니다.")
    if _run["pause_event"]:
        _run["pause_event"].set()


def unpause_training() -> None:
    if _run["status"] != "paused":
        raise RuntimeError("일시정지 상태가 아닙니다.")
    if _run["pause_event"]:
        _run["pause_event"].clear()
    _run["status"] = "running"


def stop_training() -> None:
    if _run["status"] == "idle":
        raise RuntimeError("실행 중인 학습이 없습니다.")
    if _run["stop_event"]:
        _run["stop_event"].set()
    if _run["pause_event"]:
        _run["pause_event"].clear()


# ---------------------------------------------------------------------------
# Public — 배치 학습
# ---------------------------------------------------------------------------

def start_batch() -> tuple[str, int, str]:
    """배치 학습 시작. (first_exp_id, batch_total) 반환."""
    if _run["status"] != "idle":
        raise RuntimeError("이미 학습이 진행 중입니다.")

    state       = get_state()
    queue_items = state["experiment_queue"]
    pending     = [(i, item) for i, item in enumerate(queue_items) if item.get("status") == "pending"]

    if not pending:
        raise ValueError("대기중인 항목이 없습니다.")
    if not state.get("dataset_path"):
        raise ValueError("dataset_path가 설정되지 않았습니다. 탭1에서 데이터셋을 검증하세요.")

    _run["batch_mode"]            = True
    _run["batch_total"]           = len(pending)
    _run["batch_advance_pending"] = False
    _run["batch_stopping"]        = False
    _run["batch_skip_current"]    = False
    _run["current_batch_indices"] = [i for i, _ in pending]

    first_idx, first_item = pending[0]
    queue_items[first_idx] = {**queue_items[first_idx], "status": "running"}

    model_config         = first_item.get("model_cfg", {})
    preprocessing_config = first_item.get("preprocessing_config", {})
    device               = (state.get("device_info") or {}).get("device", "cpu")
    exp_id               = _generate_experiment_id(model_config.get("model_type", "unknown"))

    _run["set_id"] = first_item.get("set_id")

    _start_worker(
        exp_id=exp_id,
        experiment_name=first_item.get("name", ""),
        created_at=_generate_created_at(),
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        dataset_path=state["dataset_path"],
        device=device,
    )
    return exp_id, _run["batch_total"], _run["model_type"]


def skip_batch_item() -> None:
    if not _run["batch_mode"]:
        raise RuntimeError("배치 모드가 아닙니다.")
    if _run["status"] == "running":
        _run["batch_skip_current"] = True
        if _run["pause_event"]:
            _run["pause_event"].set()
    elif _run["status"] == "paused":
        _run["batch_skip_current"] = True
        if _run["stop_event"]:
            _run["stop_event"].set()
        if _run["pause_event"]:
            _run["pause_event"].clear()
    else:
        raise RuntimeError("실행 중이거나 일시정지 상태여야 합니다.")


def stop_batch_all() -> None:
    if not _run["batch_mode"]:
        raise RuntimeError("배치 모드가 아닙니다.")
    _run["batch_stopping"] = True
    if _run["stop_event"]:
        _run["stop_event"].set()
    if _run["pause_event"]:
        _run["pause_event"].clear()


# ---------------------------------------------------------------------------
# Worker 시작 (내부)
# ---------------------------------------------------------------------------

def _start_worker(
    exp_id: str,
    experiment_name: str,
    created_at: str,
    model_config: dict,
    preprocessing_config: dict,
    dataset_path: str,
    device: str,
    init_progress: dict | None = None,
    init_loss_history: list | None = None,
    **worker_kwargs,
) -> str:
    stop_event  = threading.Event()
    pause_event = threading.Event()
    result_q: _queue.Queue = _queue.Queue()

    worker = TrainingWorker(
        experiment_id=exp_id,
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        dataset_path=dataset_path,
        device=device,
        stop_event=stop_event,
        result_queue=result_q,
        pause_event=pause_event,
        **worker_kwargs,
    )
    worker.daemon = True
    worker.start()

    _run["status"]             = "running"
    _run["model_type"]         = model_config.get("model_type", "")
    _run["model_config"]         = model_config
    _run["preprocessing_config"] = preprocessing_config
    _run["exp_id"]             = exp_id
    _run["experiment_name"]    = experiment_name
    _run["created_at"]         = created_at
    _run["stop_event"]         = stop_event
    _run["pause_event"]        = pause_event
    _run["result_queue"]       = result_q
    _run["worker"]             = worker
    _run["last_ckpt_path"]     = None
    _run["current_stage_idx"]  = None
    _run["current_stage_name"] = None
    _run["log_lines"]          = []
    _run["loss_history"]       = list(init_loss_history) if init_loss_history else []
    _run["progress"]           = init_progress or {
        "step": 0, "total": _get_total_steps(model_config), "loss": None, "elapsed": 0.0,
    }

    if _run["_poll_task"] and not _run["_poll_task"].done():
        _run["_poll_task"].cancel()
    _run["_poll_task"] = asyncio.create_task(_poll_result_queue())

    return exp_id


# ---------------------------------------------------------------------------
# result_queue 폴링 루프 (asyncio background Task)
# ---------------------------------------------------------------------------

async def _poll_result_queue() -> None:
    result_q: _queue.Queue = _run["result_queue"]

    while True:
        try:
            msg = result_q.get_nowait()
        except _queue.Empty:
            await asyncio.sleep(0.1)
            continue

        msg_type = msg.get("type")

        if msg_type == "progress":
            _handle_progress(msg)
            await _broadcast(dict(msg))
        elif msg_type == "log":
            _handle_log(msg)
            await _broadcast(dict(msg))
        elif msg_type == "stage":
            _handle_stage(msg)
            await _broadcast(dict(msg))
        elif msg_type == "paused":
            await _handle_paused(msg)
        elif msg_type == "completed":
            await _handle_completed(msg)
            return
        elif msg_type == "error":
            await _handle_error(msg)
            return
        elif msg_type == "stopped":
            await _handle_stopped(msg)
            return


# ---------------------------------------------------------------------------
# 메시지 핸들러
# ---------------------------------------------------------------------------

def _handle_progress(msg: dict) -> None:
    _run["progress"] = {
        "step":    msg["step"],
        "total":   msg["total"],
        "loss":    msg["loss"],
        "elapsed": msg["elapsed"],
    }
    loss_val = msg.get("loss")
    if loss_val is not None and loss_val > 0:
        _run["loss_history"].append({"step": msg["step"], "loss": loss_val})


def _handle_log(msg: dict) -> None:
    ts   = datetime.now(tz=KST).strftime("%H:%M:%S")
    line = f"[{ts}] {msg['message']}"
    _run["log_lines"].append(line)
    _run["log_lines"] = _run["log_lines"][-100:]


def _handle_stage(msg: dict) -> None:
    _run["current_stage_idx"]  = msg["stage_idx"]
    _run["current_stage_name"] = msg["stage_name"]


async def _handle_paused(msg: dict) -> None:
    if _run["batch_skip_current"]:
        # 배치 건너뛰기(B안): 체크포인트 저장 완료 → stop으로 해제
        _mark_batch_item("건너뜀")
        _save_batch_item_to_history("건너뜀")
        _run["batch_skip_current"]    = False
        _run["batch_advance_pending"] = True
        if _run["stop_event"]:
            _run["stop_event"].set()
        if _run["pause_event"]:
            _run["pause_event"].clear()
        # stopped 메시지가 곧 도착 → ghost stop으로 처리됨
    else:
        _run["status"]        = "paused"
        _run["last_ckpt_path"] = msg.get("ckpt_path")
        await _broadcast({
            "type":     "paused",
            "step":     msg.get("step"),
            "ckpt_path": msg.get("ckpt_path"),
        })


async def _handle_completed(msg: dict) -> None:
    exp_id               = _run["exp_id"]
    model_config         = _run.get("model_config") or {}
    preprocessing_config = _run.get("preprocessing_config") or {}
    batch_mode           = _run["batch_mode"]  # reset 전에 캡처

    y_true         = msg["y_true"]
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

    early_stopped = bool(msg.get("early_stopped", False))
    metrics = compute_metrics(y_true, anomaly_scores, threshold)
    record  = _build_experiment_record(exp_id, "completed", metrics, msg.get("duration_seconds"), early_stopped)

    batch_item_status = "completed"
    try:
        ok, free_mb = check_disk_space(required_mb=100.0)
        if not ok:
            raise RuntimeError(
                f"ERR_DISK_SPACE: 디스크 여유 공간이 부족합니다 ({free_mb:.0f} MB). "
                "모델 저장에 최소 100 MB가 필요합니다."
            )
        save_completed_experiment(
            exp_id, msg["model"], record,
            preprocessing_config=preprocessing_config,
            model_config=model_config,
        )

        secs             = msg.get("duration_seconds", 0)
        hours, rem       = divmod(secs, 3600)
        mins, sec        = divmod(rem, 60)
        auc              = metrics.get("auc", 0.0)
        dur_str = f"{hours}시간 {mins}분 {sec}초" if hours > 0 else (f"{mins}분 {sec}초" if mins > 0 else f"{sec}초")
        await _broadcast({
            "type":             "completed",
            "exp_id":           exp_id,
            "auc":              round(auc, 4),
            "duration_seconds": secs,
            "message":          f"학습 완료. AUC: {auc:.4f} | {dur_str}",
            "early_stopped":    early_stopped,
        })

    except RuntimeError as e:
        batch_item_status = "failed"
        await _broadcast({"type": "error", "message": str(e), "traceback": ""})

    finally:
        del msg["model"]
        device = (get_state().get("device_info") or {}).get("device", "cpu")
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

    if batch_mode:
        _mark_batch_item(batch_item_status)
        _reset_run_state()
        await _advance_batch_queue()
    else:
        _reset_run_state()


async def _handle_error(msg: dict) -> None:
    tb         = msg.get("traceback", "")
    batch_mode = _run["batch_mode"]

    if batch_mode:
        _mark_batch_item("failed")
        _save_batch_item_to_history("실패")
        await _broadcast({"type": "batch_item_error", "traceback": tb[:300]})
        _reset_run_state()
        await _advance_batch_queue()
    else:
        await _broadcast({"type": "error", "traceback": tb})
        _reset_run_state()


async def _handle_stopped(msg: dict) -> None:
    step = msg.get("step", 0)

    # ghost stop: batch_advance_pending이 True → 건너뜀/완료 후 남은 stop 메시지
    if _run["batch_advance_pending"]:
        _run["batch_advance_pending"] = False
        _reset_run_state()
        await _advance_batch_queue()
        return

    exp_id     = _run["exp_id"]
    batch_skip = _run["batch_skip_current"]

    if exp_id:
        item_status = "건너뜀" if batch_skip else "중단"
        record = _build_experiment_record(exp_id, item_status, None, None)
        try:
            append_experiment(record)
        except RuntimeError:
            pass

    if batch_skip:
        _mark_batch_item("skipped")
        _run["batch_skip_current"]    = False
        _run["batch_advance_pending"] = False
        await _broadcast({"type": "batch_item_skipped"})
        _reset_run_state()
        await _advance_batch_queue()

    elif _run["batch_stopping"]:
        # stop_batch_all()로 인한 전체 중단
        _mark_batch_item("stopped")
        _run["batch_mode"]     = False
        _run["batch_stopping"] = False
        await _broadcast({"type": "batch_stopped", "step": step})
        _reset_run_state()

    elif _run["batch_mode"]:
        # 배치 중 단일 stop 명령
        _mark_batch_item("stopped")
        _run["batch_mode"] = False
        await _broadcast({"type": "stopped", "step": step})
        _reset_run_state()

    else:
        await _broadcast({"type": "stopped", "step": step})
        _reset_run_state()


# ---------------------------------------------------------------------------
# 배치 자동 진행
# ---------------------------------------------------------------------------

async def _advance_batch_queue() -> None:
    state       = get_state()
    queue_items = state["experiment_queue"]
    pending     = [(i, item) for i, item in enumerate(queue_items) if item.get("status") == "pending"]

    if not pending:
        current_indices = _run.get("current_batch_indices", [])
        completed = sum(1 for i in current_indices if queue_items[i].get("status") == "completed")
        failed    = sum(1 for i in current_indices if queue_items[i].get("status") == "failed")
        skipped   = sum(1 for i in current_indices if queue_items[i].get("status") == "skipped")
        _run["batch_mode"] = False
        await _broadcast({
            "type":      "batch_completed",
            "completed": completed,
            "failed":    failed,
            "skipped":   skipped,
        })
        return

    next_idx, next_item = pending[0]
    queue_items[next_idx] = {**queue_items[next_idx], "status": "running"}

    model_config         = next_item.get("model_cfg", {})
    preprocessing_config = next_item.get("preprocessing_config", {})
    device               = (state.get("device_info") or {}).get("device", "cpu")
    exp_id               = _generate_experiment_id(model_config.get("model_type", "unknown"))

    _run["set_id"] = next_item.get("set_id")

    _start_worker(
        exp_id=exp_id,
        experiment_name=next_item.get("name", ""),
        created_at=_generate_created_at(),
        model_config=model_config,
        preprocessing_config=preprocessing_config,
        dataset_path=state["dataset_path"],
        device=device,
    )
    await _broadcast({
        "type":       "batch_item_started",
        "exp_id":     exp_id,
        "queue_idx":  next_idx,
        "model_type": model_config.get("model_type", ""),
    })


# ---------------------------------------------------------------------------
# 배치 헬퍼
# ---------------------------------------------------------------------------

def _mark_batch_item(status: str) -> None:
    queue_items = get_state()["experiment_queue"]
    for i, item in enumerate(queue_items):
        if item.get("status") == "running":
            queue_items[i] = {**item, "status": status}
            break


def _save_batch_item_to_history(status: str) -> None:
    exp_id = _run.get("exp_id", "")
    if not exp_id:
        return
    try:
        record = _build_experiment_record(exp_id, status, None, None)
        append_experiment(record)
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# 상태 초기화
# ---------------------------------------------------------------------------

def _reset_run_state() -> None:
    """학습 종료 후 per-run 상태 초기화. 배치 관련 플래그는 배치 흐름이 관리."""
    _run["status"]             = "idle"
    _run["exp_id"]             = None
    _run["experiment_name"]    = None
    _run["created_at"]         = None
    _run["stop_event"]         = None
    _run["pause_event"]        = None
    _run["result_queue"]       = None
    _run["worker"]             = None
    _run["last_ckpt_path"]     = None
    _run["current_stage_idx"]  = None
    _run["current_stage_name"] = None
    _run["batch_skip_current"]   = False
    _run["model_type"]           = None
    _run["model_config"]         = None
    _run["preprocessing_config"] = None


# ---------------------------------------------------------------------------
# 실험 레코드 구성
# ---------------------------------------------------------------------------

def _build_experiment_record(
    exp_id: str,
    status: str,
    metrics: dict | None,
    duration_seconds: int | None,
    early_stopped: bool = False,
) -> dict:
    state                      = get_state()
    model_config: dict         = _run.get("model_config") or {}
    preprocessing_config: dict = _run.get("preprocessing_config") or {}
    dataset_path: str          = state.get("dataset_path") or ""
    product_name: str    = state.get("product_name") or ""
    experiment_name: str = _run.get("experiment_name") or exp_id
    created_at: str      = _run.get("created_at") or _generate_created_at()

    record: dict = {
        "experiment_id":        exp_id,
        "name":                 experiment_name,
        "status":               status,
        "created_at":           created_at,
        "model_type":           model_config.get("model_type", ""),
        "preprocessing_method": preprocessing_config.get("method", "none"),
        "preprocessing_params": preprocessing_config.get("params"),
        "background_method":    preprocessing_config.get("background_method", "none"),
        "model_params":         model_config.get("params", {}),
        "threshold_method":     model_config.get("threshold_method", "percentile"),
        "threshold_value":      model_config.get("threshold_value", 95.0),
        "dataset_path":         dataset_path,
        "product_name":         product_name,
        "image_size":           preprocessing_config.get("image_size", 256),
        "duration_seconds":     duration_seconds,
        "metrics":              metrics,
        "model_path":           None,
        "configs_path":         None,
        "set_id":               _run.get("set_id"),
        "early_stopped":        early_stopped,
    }

    if status == "중단":
        record["metrics"]      = None
        record["model_path"]   = None
        record["configs_path"] = None

    return record
