"""
api/services/inspection_service.py

비즈니스 로직 레이어. HTTP/WebSocket 관심사와 분리.
HTTP 예외(HTTPException)를 직접 raise하지 않고, 표준 예외를 raise하여
호출자(라우터)가 HTTP 상태 코드로 변환하도록 위임.

예외 변환 규칙 (라우터 기준):
    LookupError  → 404
    ValueError   → 400
    FileNotFoundError → 400
    RuntimeError → 500
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

from api.state import (
    clear_model_cache,
    get_gpu_warning,
    get_model,
    get_state,
    normalize_anomaly_score,
    reset_inspection_state,
    sample_from_pool,
)
from inspection.utils.test_sampler import build_test_pool
from utils.image_utils import apply_preprocessing
from utils.model_factory import run_inference
from utils.storage import load_history
from utils.threshold_utils import resolve_threshold

KST = timezone(timedelta(hours=9))


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def apply_model(experiment_id: str, source_path: str | None = None) -> dict:
    """
    R-INSP-05 7단계 모델 초기화.

    Raises:
        LookupError: experiment_id 에 해당하는 실험 없음
        ValueError: 미완료 실험 또는 빈 test pool
        FileNotFoundError: 데이터셋 경로 없음
        RuntimeError: 모델 로드 실패
    """
    # 1. GPU 경고
    gpu_warning = get_gpu_warning()

    # 실험 조회
    all_records = load_history()
    experiment  = next(
        (r for r in all_records if r.get("experiment_id") == experiment_id),
        None,
    )
    if experiment is None:
        raise LookupError(f"실험을 찾을 수 없습니다: {experiment_id}")
    if experiment.get("status") != "completed":
        raise ValueError("완료된 실험만 적용할 수 있습니다.")

    # 2. 모델 캐시 무효화
    clear_model_cache()

    # 3. 상태 초기화
    reset_inspection_state()

    # 4. active_model 갱신
    effective_source_path = source_path if source_path and source_path.strip() else experiment["dataset_path"]

    preprocessing_config = {
        "method":     experiment.get("preprocessing_method", "none"),
        "params":     experiment.get("preprocessing_params") or {},
        "image_size": experiment.get("image_size", 256),
    }
    _metrics    = experiment.get("metrics") or {}
    _all_scores = _metrics.get("anomaly_scores") or []
    score_min   = float(min(_all_scores)) if _all_scores else 0.0
    score_max   = float(max(_all_scores)) if _all_scores else 1.0

    raw_threshold        = resolve_threshold(experiment)
    threshold_normalized = normalize_anomaly_score(raw_threshold, score_min, score_max)

    device = _get_device()
    state  = get_state()
    state["insp_active_model"] = {
        "experiment_id":        experiment["experiment_id"],
        "name":                 experiment.get("name", experiment["experiment_id"]),
        "model_path":           experiment["model_path"],
        "model_type":           experiment["model_type"],
        "threshold":            threshold_normalized,
        "dataset_path":         effective_source_path,
        "preprocessing_config": preprocessing_config,
        "score_min":            score_min,
        "score_max":            score_max,
        "device":               device,
    }

    # 5. test pool 구성
    try:
        pool = build_test_pool(effective_source_path)
    except FileNotFoundError:
        state["insp_active_model"] = None
        raise

    if not pool:
        state["insp_active_model"] = None
        raise ValueError(
            "ERR_INSP_TEST_POOL_EMPTY: 테스트 이미지를 찾을 수 없습니다. "
            f"데이터셋 경로를 확인해 주세요: {effective_source_path}/test/"
        )

    state["insp_test_pool"]  = pool
    state["insp_pool_index"] = 0

    # 6. 모델 사전 로드 (캐시 warming)
    active = state["insp_active_model"]
    try:
        get_model(
            model_path=active["model_path"],
            model_type=active["model_type"],
            device=active["device"],
        )
    except RuntimeError:
        state["insp_active_model"] = None
        raise

    # 7. 응답
    return {
        "success":      True,
        "active_model": state["insp_active_model"],
        "gpu_warning":  gpu_warning,
    }


def update_source_path(source_path: str | None) -> dict:
    """
    모델 재로드 없이 검사 이미지 풀만 교체.
    source_path가 None 또는 빈 문자열이면 실험의 원래 dataset_path로 초기화.

    Raises:
        RuntimeError: 모델 미선택
        LookupError: 원래 실험 기록 없음 (초기화 시)
        FileNotFoundError: 경로 없음
        ValueError: 빈 풀
    """
    state  = get_state()
    active = state.get("insp_active_model")
    if active is None:
        raise RuntimeError("적용된 모델이 없습니다. 먼저 모델을 선택해 주세요.")

    if not source_path or not source_path.strip():
        all_records = load_history()
        experiment  = next(
            (r for r in all_records if r.get("experiment_id") == active["experiment_id"]),
            None,
        )
        if experiment is None:
            raise LookupError("원래 실험 기록을 찾을 수 없습니다.")
        effective_path = experiment["dataset_path"]
    else:
        effective_path = source_path.strip()

    try:
        pool = build_test_pool(effective_path)
    except FileNotFoundError:
        raise

    if not pool:
        raise ValueError(
            "ERR_INSP_TEST_POOL_EMPTY: 테스트 이미지를 찾을 수 없습니다. "
            f"경로를 확인해 주세요: {effective_path}/test/"
        )

    state["insp_test_pool"]  = pool
    state["insp_pool_index"] = 0
    active["dataset_path"]   = effective_path

    return {"success": True, "source_path": effective_path}


def run_single_inspection() -> dict:
    """
    단일 이미지 추론. REST POST와 WebSocket 자동 검사 루프 공용.

    Raises:
        RuntimeError: 모델 미선택, pool 비어있음, 추론 실패
    Returns:
        inspection_record + was_reshuffled 필드
    """
    state  = get_state()
    active = state.get("insp_active_model")
    if active is None:
        raise RuntimeError("모델이 선택되지 않았습니다.")
    if not state.get("insp_test_pool"):
        raise RuntimeError("ERR_INSP_TEST_POOL_EMPTY: 테스트 이미지가 없습니다.")

    threshold = active["threshold"]
    score_min = active.get("score_min", 0.0)
    score_max = active.get("score_max", 1.0)
    device    = active.get("device", _get_device())

    # 1. 이미지 샘플링 (A-16)
    image_path, _gt_label, was_reshuffled = sample_from_pool()

    # 2. 모델 (캐시)
    model = get_model(
        model_path=active["model_path"],
        model_type=active["model_type"],
        device=device,
    )

    # 3. 전처리 + 추론
    preprocessing_config = active.get("preprocessing_config") or {"method": "none", "params": {}}
    _, image_tensor = apply_preprocessing(image_path, preprocessing_config)
    image_tensor    = image_tensor.unsqueeze(0)
    anomaly_map     = run_inference(model, image_tensor)   # (H, W) float32

    # 4. Score 정규화 + 판정
    raw_score     = float(np.max(anomaly_map))
    anomaly_score = normalize_anomaly_score(raw_score, score_min, score_max)
    verdict       = "불량" if anomaly_score >= threshold else "양품"

    # 5. inspection_record
    seq    = state["insp_seq_counter"] + 1
    record = {
        "seq":           seq,
        "inspected_at":  datetime.now(tz=KST).strftime("%Y-%m-%d %H:%M:%S"),
        "image_name":    Path(image_path).name,
        "image_path":    image_path,
        "verdict":       verdict,
        "anomaly_score": round(anomaly_score, 6),
    }

    # 6. 상태 갱신
    state["insp_records"].append(record)
    state["insp_seq_counter"]      = seq
    state["insp_last_result"]      = record
    state["insp_last_anomaly_map"] = anomaly_map

    # 7. 자동 검사 중 불량 → 루프 중지 + 팝업
    if verdict == "불량" and state.get("insp_auto_active"):
        state["insp_auto_active"]  = False
        state["insp_defect_popup"] = True

    return {**record, "was_reshuffled": was_reshuffled}
