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
from utils.training_worker import TrainingWorker, EFFICIENTAD_STAGES, PATCHCORE_STAGES
from tabs.tab2_config import _build_queue_df, _style_queue_rows

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
    st.header("탭3. 학습 시작 + 학습 로그")

    # 배치 자동 진행 처리 (advance_pending + idle 상태)
    if (
        st.session_state.get("_batch_advance_pending", False)
        and st.session_state.get("_batch_queue_mode", False)
        and st.session_state.get("current_run_status", "idle") == "idle"
        and st.session_state.get("dataset_path")
    ):
        st.session_state["_batch_advance_pending"] = False
        _advance_batch_queue()
        return

    queue_items: list[dict] = st.session_state.get("experiment_queue", [])
    status = st.session_state.get("current_run_status", "idle")
    batch_mode = st.session_state.get("_batch_queue_mode", False)

    if queue_items:
        # 상단 2열: 대기열 테이블(좌) + 배치 제어/상태(우)
        col_queue, col_ctrl = st.columns(2)

        with col_queue:
            st.markdown("#### 실험 대기열")
            df = _build_queue_df(queue_items)
            st.dataframe(
                df.style.apply(_style_queue_rows, axis=1),
                use_container_width=True,
                hide_index=True,
                key="tab3_queue_display",
            )

        with col_ctrl:
            if batch_mode:
                batch_total = st.session_state.get("_batch_total_count", len(queue_items))
                done_count = sum(
                    1 for item in queue_items
                    if item.get("status") in ("완료", "실패", "건너뜀")
                )
                st.info(f"🔄 일괄 학습 진행 중: {done_count} / {batch_total} 완료")

                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button(
                        "⏸ 일시정지",
                        disabled=(status != "running"),
                        use_container_width=True,
                        key="tab3_batch_pause_btn",
                    ):
                        pause_ev = st.session_state.get("_pause_event")
                        if pause_ev:
                            pause_ev.set()

                with b2:
                    if st.button(
                        "⏭ 현재 학습 건너뛰기",
                        disabled=(status not in ("running", "paused")),
                        use_container_width=True,
                        key="tab3_batch_skip_btn",
                    ):
                        st.session_state["_batch_skip_current"] = True
                        if status == "running":
                            pause_ev = st.session_state.get("_pause_event")
                            if pause_ev:
                                pause_ev.set()
                        else:
                            stop_ev = st.session_state.get("_stop_event")
                            if stop_ev:
                                stop_ev.set()
                            pause_ev = st.session_state.get("_pause_event")
                            if pause_ev:
                                pause_ev.clear()

                with b3:
                    if st.button(
                        "⏹ 전체 학습 중단",
                        type="secondary",
                        use_container_width=True,
                        key="tab3_batch_stop_all_btn",
                    ):
                        st.session_state["_batch_queue_mode"] = False
                        stop_ev = st.session_state.get("_stop_event")
                        if stop_ev:
                            stop_ev.set()
                        pause_ev = st.session_state.get("_pause_event")
                        if pause_ev:
                            pause_ev.clear()
                        st.info("전체 학습 중단 신호를 전송했습니다.")

                if status in ("running", "paused"):
                    st.info("학습 중 새로고침 시 학습 상태를 확인할 수 없습니다.", icon="⚠️")
                    st.info("🔄 학습이 진행 중입니다. 탭을 전환해도 학습은 계속됩니다.")

            else:
                pending_count = sum(
                    1 for item in queue_items if item.get("status") == "대기중"
                )
                can_start = pending_count > 0 and status == "idle"
                if st.button(
                    f"🚀 일괄 학습 시작 ({pending_count}개 대기중)",
                    type="primary",
                    disabled=not can_start,
                    key="tab3_batch_start_btn",
                ):
                    _handle_batch_start()

        st.divider()

    if not _guard():
        return

    # 하단 전체 너비: 진행 표시 + 로그
    in_batch = queue_items and batch_mode
    if status == "running":
        if not in_batch:
            st.info("학습 중 새로고침 시 학습 상태를 확인할 수 없습니다.", icon="⚠️")
        finished = _drain_queue()
        _render_running_ui(show_info_banner=not in_batch)
        if not finished:
            time.sleep(0.3)
        st.rerun()

    elif status == "paused":
        if not in_batch:
            st.info("학습 중 새로고침 시 학습 상태를 확인할 수 없습니다.", icon="⚠️")
        _render_running_ui(show_info_banner=not in_batch)

    else:
        _show_last_result()
        _render_idle_ui()


# ── 배치 대기열 헬퍼 함수 (FR-T3-15) ────────────────────────────────────────────

def _mark_current_batch_item(status: str) -> None:
    """배치 대기열에서 '진행중' 항목의 상태를 변경한다 (FR-T3-15)."""
    queue_items = list(st.session_state.get("experiment_queue", []))
    for i, item in enumerate(queue_items):
        if item.get("status") == "진행중":
            queue_items[i] = {**item, "status": status}
            break
    st.session_state["experiment_queue"] = queue_items


def _save_batch_item_to_history(status: str) -> None:
    """건너뜀/실패 항목을 history.json에 기록한다 (FR-T3-15)."""
    exp_id = st.session_state.get("current_exp_id", "")
    if not exp_id:
        return
    try:
        record = _build_experiment_record(exp_id, status, None, None)
        append_experiment(record)
        st.session_state.setdefault("experiments", {})[exp_id] = record
    except RuntimeError:
        pass


def _advance_batch_queue() -> None:
    """
    FR-T3-15: 배치 대기열에서 다음 '대기중' 항목으로 학습을 진행한다.
    대기중 항목이 없으면 배치 모드를 종료하고 완료 메시지를 표시한다.
    """
    queue_items = list(st.session_state.get("experiment_queue", []))
    pending = [
        (i, item) for i, item in enumerate(queue_items)
        if item.get("status") == "대기중"
    ]

    if not pending:
        # 모든 항목 처리 완료 → 배치 종료
        st.session_state["_batch_queue_mode"] = False
        completed = sum(1 for item in queue_items if item.get("status") == "완료")
        failed    = sum(1 for item in queue_items if item.get("status") == "실패")
        skipped   = sum(1 for item in queue_items if item.get("status") == "건너뜀")
        st.session_state["_last_result"] = {
            "level": "success" if not failed else "warning",
            "text": (
                f"일괄 학습이 완료되었습니다. "
                f"완료: {completed}개 | 실패: {failed}개 | 건너뜀: {skipped}개"
            ),
        }
        return

    # 다음 대기중 항목 시작
    next_idx, next_item = pending[0]
    st.session_state["preprocessing_config"] = dict(next_item["preprocessing_config"])
    st.session_state["model_config"]         = dict(next_item["model_config"])
    st.session_state["current_set_id"]       = next_item.get("set_id")

    queue_items[next_idx] = {**queue_items[next_idx], "status": "진행중"}
    st.session_state["experiment_queue"] = queue_items

    _handle_start_training(next_item.get("name", ""))


# ── 탭3 상단 대기열 표시 (FR-T3-14, FR-T3-15) ─────────────────────────────────

def _render_queue_at_tab3_top() -> None:
    """
    FR-T3-14: 탭3 최상단 실험 대기열 테이블.
    FR-T3-15: 일괄 학습 시작 버튼 / 진행 배너 / 제어 버튼(⏸/⏭/⏹) / 자동 진행.
    len(experiment_queue) == 0 이면 렌더링하지 않는다.
    """
    queue_items: list[dict] = st.session_state.get("experiment_queue", [])

    # ── 배치 자동 진행 처리 (advance_pending + idle 상태) ──────────────────────
    if (
        st.session_state.get("_batch_advance_pending", False)
        and st.session_state.get("_batch_queue_mode", False)
        and st.session_state.get("current_run_status", "idle") == "idle"
        and st.session_state.get("dataset_path")
    ):
        st.session_state["_batch_advance_pending"] = False
        _advance_batch_queue()
        return  # _advance_batch_queue → _handle_start_training → st.rerun()

    if not queue_items:
        return

    st.markdown("#### 실험 대기열")

    # 대기열 테이블 — FR-T2-17과 동일 스타일
    df = _build_queue_df(queue_items)
    st.dataframe(
        df.style.apply(_style_queue_rows, axis=1),
        use_container_width=True,
        hide_index=True,
        key="tab3_queue_display",
    )

    batch_mode = st.session_state.get("_batch_queue_mode", False)
    run_status = st.session_state.get("current_run_status", "idle")

    if batch_mode:
        # 배치 진행 중 배너
        batch_total = st.session_state.get("_batch_total_count", len(queue_items))
        done_count  = sum(
            1 for item in queue_items
            if item.get("status") in ("완료", "실패", "건너뜀")
        )
        st.info(f"🔄 일괄 학습 진행 중: {done_count} / {batch_total} 완료")

        # ── 제어 버튼 3개 (FR-T3-15) ────────────────────────────────────────
        b1, b2, b3 = st.columns(3)

        with b1:
            if st.button(
                "⏸ 일시정지",
                disabled=(run_status != "running"),
                use_container_width=True,
                key="tab3_batch_pause_btn",
            ):
                pause_ev = st.session_state.get("_pause_event")
                if pause_ev:
                    pause_ev.set()

        with b2:
            if st.button(
                "⏭ 현재 학습 건너뛰기",
                disabled=(run_status not in ("running", "paused")),
                use_container_width=True,
                key="tab3_batch_skip_btn",
            ):
                st.session_state["_batch_skip_current"] = True
                if run_status == "running":
                    # 체크포인트 저장 후 건너뛰기 (B안) — pause_event로 저장 트리거
                    pause_ev = st.session_state.get("_pause_event")
                    if pause_ev:
                        pause_ev.set()
                else:
                    # 이미 일시정지 상태 → 그냥 중단
                    stop_ev = st.session_state.get("_stop_event")
                    if stop_ev:
                        stop_ev.set()
                    pause_ev = st.session_state.get("_pause_event")
                    if pause_ev:
                        pause_ev.clear()

        with b3:
            if st.button(
                "⏹ 전체 학습 중단",
                type="secondary",
                use_container_width=True,
                key="tab3_batch_stop_all_btn",
            ):
                st.session_state["_batch_queue_mode"] = False
                stop_ev = st.session_state.get("_stop_event")
                if stop_ev:
                    stop_ev.set()
                pause_ev = st.session_state.get("_pause_event")
                if pause_ev:
                    pause_ev.clear()
                st.info("전체 학습 중단 신호를 전송했습니다.")

    else:
        # 일괄 학습 시작 버튼
        pending_count = sum(
            1 for item in queue_items if item.get("status") == "대기중"
        )
        can_start = pending_count > 0 and run_status == "idle"
        if st.button(
            f"🚀 일괄 학습 시작 ({pending_count}개 대기중)",
            type="primary",
            disabled=not can_start,
            key="tab3_batch_start_btn",
        ):
            _handle_batch_start()

    st.divider()


def _handle_batch_start() -> None:
    """
    FR-T3-15: 일괄 학습 시작 처리.
    첫 번째 '대기중' 항목의 설정을 session_state에 로드하고
    단일 학습 시작 흐름(_handle_start_training)을 호출한다.
    순차 실행(완료→자동 다음) 로직은 다음 태스크에서 구현.
    """
    queue_items: list[dict] = st.session_state.get("experiment_queue", [])
    pending = [
        (i, item) for i, item in enumerate(queue_items)
        if item.get("status") == "대기중"
    ]

    if not pending:
        st.warning("대기중인 항목이 없습니다.")
        return

    if st.session_state.get("current_run_status") != "idle":
        st.warning("이미 학습이 진행 중입니다.")
        return

    if not st.session_state.get("dataset_path"):
        st.error("먼저 탭1에서 데이터셋 경로를 설정해 주세요.")
        return

    # 배치 모드 + 전체 카운트 설정
    pending_count = len(pending)
    st.session_state["_batch_queue_mode"]  = True
    st.session_state["_batch_total_count"] = pending_count

    # 첫 번째 대기중 항목의 설정을 session_state에 로드
    first_idx, first_item = pending[0]
    st.session_state["preprocessing_config"] = dict(first_item["preprocessing_config"])
    st.session_state["model_config"]         = dict(first_item["model_config"])

    # 첫 번째 항목 상태 → "진행중"
    queue = list(st.session_state["experiment_queue"])
    queue[first_idx] = {**queue[first_idx], "status": "진행중"}
    st.session_state["experiment_queue"] = queue

    # 단일 학습 시작 (item name을 실험명으로 사용)
    _handle_start_training(first_item.get("name", ""))


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
                "새로 학습을 시작하거나 탭4에서 히스토리를 확인하세요."
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
        key="tab3_experiment_name",
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

# FR-T3-13: ETA 계산 대상 루프 단계 이름 집합
_LOOP_STAGE_NAMES = frozenset({"학습 루프", "특징 추출"})
_ETA_MIN_STEPS_EFFICIENTAD = 100  # EfficientAD: 최소 100 step 후 ETA 신뢰도 확보


def _compute_eta(
    stage_name: str | None,
    step: int,
    total: int,
    elapsed: float,
    model_type: str,
) -> str | None:
    """
    FR-T3-13: 학습 루프 단계 ETA 계산.

    루프 단계(학습 루프 / 특징 추출)에서만 계산.
    비-루프 단계는 None 반환 → 호출자가 '진행 중...' 표시.

    EfficientAD: 100 step 이상 진행 후부터 신뢰도 확보.
    PatchCore: 1 배치 이상 진행 후 계산.
    """
    if stage_name not in _LOOP_STAGE_NAMES:
        return None

    remaining = total - step
    if step <= 0 or elapsed <= 0.0 or remaining <= 0:
        return None

    if model_type == "efficientad" and step < _ETA_MIN_STEPS_EFFICIENTAD:
        return None

    eta_seconds = elapsed / step * remaining
    return f"{eta_seconds:.0f}s"


def _render_stage_indicator() -> None:
    """
    FR-T3-11/12: 학습 단계 스텝 인디케이터 (진행률 바 위 가로 배치).

    완료 단계: ✅ ①이름  (회색)
    현재 단계: 🔵 ②이름  (파란색 bold)
    미완료 단계: ○ ③이름  (연회색)
    """
    model_config = st.session_state.get("model_config") or {}
    model_type   = model_config.get("model_type", "")
    stages       = EFFICIENTAD_STAGES if model_type == "efficientad" else PATCHCORE_STAGES

    current_idx: int | None = st.session_state.get("_current_stage_idx")

    _CIRCLED = "①②③④⑤⑥⑦"
    parts: list[str] = []
    for idx, name in stages:
        num = _CIRCLED[idx] if idx < len(_CIRCLED) else str(idx + 1)
        if current_idx is None or idx > current_idx:
            parts.append(f'<span style="color:#aaaaaa">○&nbsp;{num}{name}</span>')
        elif idx < current_idx:
            parts.append(f'<span style="color:#777777">✅&nbsp;{num}{name}</span>')
        else:
            parts.append(f'<span style="color:#1f77b4"><b>🔵&nbsp;{num}{name}</b></span>')

    html = "&nbsp;&nbsp;".join(parts)
    st.markdown(
        f'<div style="font-size:0.88em;padding:4px 0 6px 0">{html}</div>',
        unsafe_allow_html=True,
    )


def _render_running_ui(show_info_banner: bool = True) -> None:
    """current_run_status == "running" 또는 "paused" 상태 UI."""
    status = st.session_state.get("current_run_status", "running")

    if show_info_banner:
        st.info("🔄 학습이 진행 중입니다. 탭을 전환해도 학습은 계속됩니다.")

    _render_stage_indicator()

    progress = st.session_state.get("_progress") or {}
    step     = progress.get("step", 0)
    total    = progress.get("total", 1)
    loss     = progress.get("loss")

    start_t = st.session_state.get("_training_start_time") or time.time()
    elapsed = time.time() - start_t

    # FR-T3-13: ETA 계산
    stage_name = st.session_state.get("_current_stage_name")
    model_type = (st.session_state.get("model_config") or {}).get("model_type", "")
    eta_str    = _compute_eta(stage_name, step, total, elapsed, model_type)

    pct = step / total if total > 0 else 0.0
    if status == "paused":
        ckpt_path = st.session_state.get("_last_ckpt_path")
        ckpt_info = f" | 저장: {Path(ckpt_path).name}" if ckpt_path else ""
        label = f"⏸ 일시정지 | Step {step:,} / {total:,} ({pct*100:.1f}%) | 경과: {elapsed:.0f}s{ckpt_info}"
    else:
        label = f"Step {step:,} / {total:,} ({pct*100:.1f}%)"
        if loss is not None and loss > 0:
            label += f" | Loss: {loss:.4f}"
        label += f" | 경과: {elapsed:.0f}s"
        if eta_str is not None:
            label += f" | ETA: {eta_str}"
        else:
            label += " | 진행 중..."
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
    log_text = "\n".join(log_lines[-50:])
    st.text_area("학습 로그", value=log_text, height=200, disabled=True, key="tab3_log_area")

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

    # background_method에 따라 학습에 사용할 실효 경로 결정
    bg_method = preprocessing_config.get("background_method", "none")
    if bg_method == "sam2":
        bg_clean_path = Path(dataset_path) / "background_clean"
        if bg_clean_path.is_dir():
            dataset_path = str(bg_clean_path)
        else:
            st.error(
                "background_clean/ 폴더가 없습니다. "
                "SAM2 처리된 이미지를 먼저 준비하거나 배경 분리 설정을 '없음'으로 변경해 주세요."
            )
            st.stop()

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
    st.session_state["_current_stage_idx"]  = None
    st.session_state["_current_stage_name"] = None
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

    bg_method = preprocessing_config.get("background_method", "none")
    if bg_method == "sam2":
        bg_clean_path = Path(dataset_path) / "background_clean"
        if bg_clean_path.is_dir():
            dataset_path = str(bg_clean_path)

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
    st.session_state["_current_stage_idx"]   = None
    st.session_state["_current_stage_name"]  = None
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
        elif msg_type == "stage":
            _handle_stage(msg)
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
    if st.session_state.get("_batch_skip_current", False):
        # 배치 건너뛰기 (B안): 체크포인트 저장 완료 → 건너뜀 처리
        _mark_current_batch_item("건너뜀")
        _save_batch_item_to_history("건너뜀")
        st.session_state["_batch_skip_current"]   = False
        st.session_state["_batch_advance_pending"] = True
        # 체크포인트를 저장한 worker가 아직 pause loop에 있으므로 stop으로 해제
        stop_ev = st.session_state.get("_stop_event")
        if stop_ev:
            stop_ev.set()
        pause_ev = st.session_state.get("_pause_event")
        if pause_ev:
            pause_ev.clear()
        _reset_run_state()
        # _batch_advance_pending = True → 다음 rerun에서 자동 진행
    else:
        st.session_state["current_run_status"] = "paused"
        st.session_state["_last_ckpt_path"]    = msg.get("ckpt_path")


def _handle_stage(msg: dict) -> None:
    """FR-T3-11/12: stage 메시지 처리 — 현재 학습 단계 갱신."""
    st.session_state["_current_stage_idx"]  = msg["stage_idx"]
    st.session_state["_current_stage_name"] = msg["stage_name"]


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
        # 배치 모드: 항목 완료 마킹 + 자동 진행 예약
        if st.session_state.get("_batch_queue_mode", False):
            _mark_current_batch_item("완료")
            st.session_state["_batch_advance_pending"] = True

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
        # 배치 모드 저장 오류: 실패 처리 후 다음 항목 진행
        if st.session_state.get("_batch_queue_mode", False):
            _mark_current_batch_item("실패")
            st.session_state["_batch_advance_pending"] = True
    finally:
        _reset_run_state()
        del msg["model"]
        if st.session_state.get("device_info", {}).get("device") == "cuda":
            torch.cuda.empty_cache()


def _handle_error(msg: dict) -> None:
    tb = msg.get("traceback", "")
    if st.session_state.get("_batch_queue_mode", False):
        # 배치 모드: 실패 기록 후 다음 항목 자동 진행
        _mark_current_batch_item("실패")
        _save_batch_item_to_history("실패")
        st.session_state["_batch_advance_pending"] = True
        st.session_state["_last_result"] = {
            "level": "warning",
            "text":  f"항목 학습 중 오류 발생. 다음 항목으로 진행합니다.\n{tb[:300]}",
        }
    else:
        st.session_state["_last_result"] = {
            "level": "error",
            "text":  f"학습 중 오류가 발생했습니다.\n{tb}",
        }
    _reset_run_state()


def _handle_stopped(msg: dict) -> None:
    """status="중단" 레코드 생성 후 history.json append."""
    # 배치 건너뛰기/완료로 이미 advance가 예약된 경우: 이 stop은 ghost stop이므로 무시
    if st.session_state.get("_batch_advance_pending", False):
        _reset_run_state()
        return

    exp_id: str = st.session_state.get("current_exp_id", "")
    step = msg.get("step", 0)

    # 배치 건너뛰기가 paused → stopped 경로로 왔을 경우 처리
    batch_skip = st.session_state.get("_batch_skip_current", False)

    if exp_id:
        item_status = "건너뜀" if batch_skip else "중단"
        record = _build_experiment_record(
            exp_id=exp_id,
            status=item_status,
            metrics=None,
            duration_seconds=None,
        )
        try:
            append_experiment(record)
            st.session_state.setdefault("experiments", {})[exp_id] = record
        except RuntimeError:
            pass

    if batch_skip:
        # paused → stopped 경로 건너뛰기
        _mark_current_batch_item("건너뜀")
        st.session_state["_batch_skip_current"]   = False
        st.session_state["_batch_advance_pending"] = True
        st.session_state["_last_result"] = {
            "level": "warning",
            "text": "항목이 건너뛰어졌습니다. 다음 항목으로 진행합니다.",
        }
    elif st.session_state.get("_batch_queue_mode", False):
        # 배치 전체 중단 (⏹ 전체 중단 또는 기존 학습 중지)
        _mark_current_batch_item("중단")
        st.session_state["_batch_queue_mode"] = False
        st.session_state["_last_result"] = {
            "level": "warning",
            "text": MSG["TRAIN_STOPPED"] + (f" ({step:,} step 완료 후 중단)" if step else ""),
        }
    else:
        # 일반 단일 학습 중단
        st.session_state["_last_result"] = {
            "level": "warning",
            "text": MSG["TRAIN_STOPPED"] + (f" ({step:,} step 완료 후 중단)" if step else ""),
        }

    _reset_run_state()


def _reset_run_state() -> None:
    """학습 종료 후 내부 상태 초기화."""
    st.session_state["current_run_status"]  = "idle"
    st.session_state["current_exp_id"]      = None
    st.session_state["current_set_id"]      = None
    st.session_state["_stop_event"]         = None
    st.session_state["_pause_event"]        = None
    st.session_state["_result_queue"]       = None
    st.session_state["_worker"]             = None
    st.session_state["_last_ckpt_path"]     = None
    st.session_state["_current_stage_idx"]  = None
    st.session_state["_current_stage_name"] = None
    st.session_state["_batch_skip_current"] = False
    # _batch_queue_mode, _batch_advance_pending, _batch_total_count은 배치 흐름이 관리


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
        "product_name":         st.session_state.get("product_name", ""),
        "model_type":           model_config["model_type"],
        "background_method":    preprocessing_config.get("background_method", "none"),
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
        "set_id":               st.session_state.get("current_set_id"),
    }

    if status == "중단":
        record["metrics"]      = None
        record["model_path"]   = None
        record["configs_path"] = None

    return record
