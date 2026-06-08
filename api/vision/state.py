"""
api/vision/state.py

단일 사용자 전제 — st.session_state + @st.cache_resource 대체.
GPU 공통 유틸은 api.common 참조.

_model_cache : (model_path, model_type, device) → 로드된 모델 객체
_state       : 검사 세션 전역 상태 (INSPECTION_SESSION_SCHEMA 대응)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from utils.model_factory import load_model_for_inference

# ---------------------------------------------------------------------------
# 모델 캐시
# ---------------------------------------------------------------------------

_model_cache: dict[tuple[str, str, str], Any] = {}


def get_model(model_path: str, model_type: str, device: str) -> Any:
    """
    (model_path, model_type, device) 키로 캐시된 모델 반환.
    캐시 미스 시 load_model_for_inference() 호출 후 저장.

    Raises:
        RuntimeError("ERR_INSP_MODEL_LOAD_FAILED: ...")
    """
    key = (model_path, model_type, device)
    if key not in _model_cache:
        configs_path = Path(model_path) / "configs.yaml"
        try:
            with open(configs_path, encoding="utf-8") as f:
                configs = yaml.safe_load(f) or {}
        except FileNotFoundError as e:
            raise RuntimeError(
                f"ERR_INSP_MODEL_LOAD_FAILED: configs.yaml 없음 — {configs_path}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"ERR_INSP_MODEL_LOAD_FAILED: configs.yaml 파싱 실패 — {e}"
            ) from e

        model_config = configs.get("model", {})
        model_config["model_type"] = model_type

        try:
            model = load_model_for_inference(
                exp_id="insp",
                model_path=model_path,
                model_config=model_config,
                device=device,
            )
        except RuntimeError as e:
            raise RuntimeError(f"ERR_INSP_MODEL_LOAD_FAILED: {e}") from e

        _model_cache[key] = model

    return _model_cache[key]


def clear_model_cache() -> None:
    """모델 교체 시 호출 — 기존 캐시 전체 무효화 (A-19)."""
    _model_cache.clear()


# ---------------------------------------------------------------------------
# 검사 세션 상태
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "insp_active_model":     None,   # dict | None — history.json experiment 레코드
    "insp_records":          [],     # list[dict] — inspection_record 배열
    "insp_seq_counter":      0,      # int — 다음 seq 값
    "insp_auto_active":      False,  # bool — 자동 검사 실행 중 여부
    "insp_last_result":      None,   # dict | None
    "insp_last_anomaly_map": None,   # np.ndarray | None — (H, W) float32
    "insp_defect_popup":     False,  # bool
    "insp_test_pool":        [],     # list[tuple[str, str]] — (절대경로, "양품"|"불량")
    "insp_pool_index":       0,      # int — 현재 샘플링 위치
}


def get_state() -> dict[str, Any]:
    """전역 상태 dict 직접 참조 반환 (수정 가능)."""
    return _state


def reset_inspection_state() -> None:
    """
    모델 교체 시 호출. insp_active_model은 유지 (R-INSP-05).
    clear_model_cache() 를 먼저 호출한 뒤 이 함수를 호출할 것.
    test_pool / pool_index 포함 전체 초기화 — 모델 교체 전용.
    """
    _state["insp_records"]          = []
    _state["insp_seq_counter"]      = 0
    _state["insp_auto_active"]      = False
    _state["insp_last_result"]      = None
    _state["insp_last_anomaly_map"] = None
    _state["insp_defect_popup"]     = False
    _state["insp_test_pool"]        = []
    _state["insp_pool_index"]       = 0


def reset_records_only() -> None:
    """
    이력 초기화 시 호출 (DELETE /api/inspection/records).
    test_pool / pool_index / active_model 은 유지 — 이력만 리셋.
    """
    _state["insp_records"]          = []
    _state["insp_seq_counter"]      = 0
    _state["insp_auto_active"]      = False
    _state["insp_last_result"]      = None
    _state["insp_last_anomaly_map"] = None
    _state["insp_defect_popup"]     = False


# ---------------------------------------------------------------------------
# 테스트 풀 샘플링 (sample_from_pool의 st-free 재구현)
# ---------------------------------------------------------------------------

def sample_from_pool() -> tuple[str, str, bool]:
    """
    _state["insp_test_pool"] 에서 insp_pool_index 위치 샘플 반환 후 index 증가.
    pool 소진(index >= len(pool)) 시 재셔플 + index 리셋 (A-16).

    Raises:
        RuntimeError("ERR_INSP_TEST_POOL_EMPTY"): pool이 비어 있을 때.
    Returns:
        (image_path, gt_label, was_reshuffled)
    """
    pool: list[tuple[str, str]] = _state["insp_test_pool"]
    index: int = _state["insp_pool_index"]

    if not pool:
        raise RuntimeError(
            "ERR_INSP_TEST_POOL_EMPTY: 테스트 이미지가 없습니다. "
            "데이터셋 경로를 확인하거나 탭3에서 모델을 재선택해 주세요."
        )

    was_reshuffled = False
    if index >= len(pool):
        random.shuffle(pool)
        _state["insp_test_pool"] = pool
        index = 0
        was_reshuffled = True

    sample = pool[index]
    _state["insp_pool_index"] = index + 1
    return sample[0], sample[1], was_reshuffled


# ---------------------------------------------------------------------------
# Anomaly Score 정규화 (insp_session_init.normalize_anomaly_score 이식)
# ---------------------------------------------------------------------------

def normalize_anomaly_score(
    raw_score: float,
    score_min: float,
    score_max: float,
) -> float:
    """학습 테스트셋 min/max 기준으로 [0, 1] 범위 정규화."""
    if score_max <= score_min:
        return 0.0
    normalized = (raw_score - score_min) / (score_max - score_min)
    return float(max(0.0, min(1.0, normalized)))
