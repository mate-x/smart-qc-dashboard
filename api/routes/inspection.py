"""
api/routes/inspection.py

탭3 — 모델 교체:
    POST /api/inspection/model        모델 적용  (body: {experiment_id})
    GET  /api/inspection/model        현재 적용 모델 조회

탭1 — 실시간 검사:
    POST /api/inspection/run          수동 검사 1회
    GET  /api/inspection/image/last   마지막 원본 이미지
    GET  /api/inspection/anomaly-map/last  마지막 Anomaly Map 히트맵
    GET  /api/inspection/overlay/last 마지막 이상영역 오버레이

탭2 — 검사 이력:
    GET    /api/inspection/records     이력 목록 (query: verdict=양품|불량|전체)
    GET    /api/inspection/records/csv CSV 다운로드
    DELETE /api/inspection/records     이력 초기화
"""
from __future__ import annotations

import io
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

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
from utils.image_utils import anomaly_map_to_heatmap, apply_preprocessing
from utils.model_factory import run_inference
from utils.storage import load_history

router = APIRouter()

KST = timezone(timedelta(hours=9))

_CSV_COLUMNS = ["번호", "시각", "이미지명", "판정결과", "Anomaly Score"]
_KEY_MAP: dict[str, str] = {
    "번호":          "seq",
    "시각":          "inspected_at",
    "이미지명":      "image_name",
    "판정결과":      "verdict",
    "Anomaly Score": "anomaly_score",
}


# ---------------------------------------------------------------------------
# 탭3 — 모델 교체
# ---------------------------------------------------------------------------

class ApplyModelRequest(BaseModel):
    experiment_id: str


@router.post("/api/inspection/model")
def apply_model(req: ApplyModelRequest) -> dict:
    """
    R-INSP-05 초기화 순서:
      1. GPU 메모리 체크
      2. 모델 캐시 무효화
      3. 상태 초기화 (insp_active_model 제외)
      4. insp_active_model 갱신
      5. build_test_pool — 실패 시 400
      6. 모델 사전 로드 — 실패 시 500
      7. 응답 반환 (gpu_warning 포함)
    """
    # 1
    gpu_warning = get_gpu_warning()

    all_records = load_history()
    experiment  = next(
        (r for r in all_records if r.get("experiment_id") == req.experiment_id),
        None,
    )
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"실험을 찾을 수 없습니다: {req.experiment_id}")
    if experiment.get("status") != "completed":
        raise HTTPException(status_code=400, detail="완료된 실험만 적용할 수 있습니다.")

    # 2
    clear_model_cache()

    # 3
    reset_inspection_state()

    # 4
    preprocessing_config = {
        "method":     experiment.get("preprocessing_method", "none"),
        "params":     experiment.get("preprocessing_params") or {},
        "image_size": experiment.get("image_size", 256),
    }
    _metrics    = experiment.get("metrics") or {}
    _all_scores = _metrics.get("anomaly_scores") or []
    score_min   = float(min(_all_scores)) if _all_scores else 0.0
    score_max   = float(max(_all_scores)) if _all_scores else 1.0

    raw_threshold        = _resolve_threshold(experiment)
    threshold_normalized = normalize_anomaly_score(raw_threshold, score_min, score_max)

    device = _get_device()
    state  = get_state()
    state["insp_active_model"] = {
        "experiment_id":        experiment["experiment_id"],
        "model_path":           experiment["model_path"],
        "model_type":           experiment["model_type"],
        "threshold":            threshold_normalized,
        "dataset_path":         experiment["dataset_path"],
        "preprocessing_config": preprocessing_config,
        "score_min":            score_min,
        "score_max":            score_max,
        "device":               device,
    }

    # 5
    try:
        pool = build_test_pool(experiment["dataset_path"])
    except FileNotFoundError as e:
        state["insp_active_model"] = None
        raise HTTPException(status_code=400, detail=str(e))

    if not pool:
        state["insp_active_model"] = None
        raise HTTPException(
            status_code=400,
            detail=(
                "ERR_INSP_TEST_POOL_EMPTY: 테스트 이미지를 찾을 수 없습니다. "
                f"데이터셋 경로를 확인해 주세요: {experiment['dataset_path']}/test/"
            ),
        )
    state["insp_test_pool"]  = pool
    state["insp_pool_index"] = 0

    # 6
    active = state["insp_active_model"]
    try:
        get_model(
            model_path=active["model_path"],
            model_type=active["model_type"],
            device=active["device"],
        )
    except RuntimeError as e:
        state["insp_active_model"] = None
        raise HTTPException(status_code=500, detail=str(e))

    # 7
    return {
        "success":      True,
        "active_model": state["insp_active_model"],
        "gpu_warning":  gpu_warning,
    }


@router.get("/api/inspection/model")
def get_active_model() -> dict:
    return {"active_model": get_state()["insp_active_model"]}


# ---------------------------------------------------------------------------
# 탭1 — 실시간 검사
# ---------------------------------------------------------------------------

@router.post("/api/inspection/run")
def run_inspection_endpoint() -> dict:
    """수동 검사 1회. 핵심 로직은 run_single_inspection()에 분리 (WebSocket 공용)."""
    try:
        return run_single_inspection()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


def run_single_inspection() -> dict:
    """
    단일 이미지 추론 흐름.
    POST /api/inspection/run 과 WebSocket 자동 검사 루프가 공용으로 호출.

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


@router.get("/api/inspection/image/last")
def get_last_image():
    """마지막 검사 원본 이미지 FileResponse."""
    state = get_state()
    last  = state.get("insp_last_result")
    if last is None:
        raise HTTPException(status_code=404, detail="검사 이력이 없습니다.")
    path = Path(last["image_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="이미지 파일을 찾을 수 없습니다.")
    suffix    = path.suffix.lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".bmp": "image/bmp"}
    media     = media_map.get(suffix, "image/png")
    return FileResponse(str(path), media_type=media)


@router.get("/api/inspection/anomaly-map/last")
def get_last_anomaly_map():
    """마지막 Anomaly Map → JET colormap PNG StreamingResponse."""
    state = get_state()
    amap: np.ndarray | None = state.get("insp_last_anomaly_map")
    if amap is None:
        raise HTTPException(status_code=404, detail="Anomaly Map이 없습니다.")
    heatmap_pil = anomaly_map_to_heatmap(amap)
    return StreamingResponse(_pil_to_png_stream(heatmap_pil), media_type="image/png")


@router.get("/api/inspection/overlay/last")
def get_last_overlay():
    """마지막 원본 + 이상영역 빨간 반투명 오버레이 PNG StreamingResponse."""
    state  = get_state()
    last   = state.get("insp_last_result")
    amap   = state.get("insp_last_anomaly_map")
    active = state.get("insp_active_model")

    if last is None or amap is None:
        raise HTTPException(status_code=404, detail="검사 이력이 없습니다.")
    if active is None:
        raise HTTPException(status_code=400, detail="모델이 선택되지 않았습니다.")

    overlay_pil = _make_anomaly_overlay(
        image_path=last["image_path"],
        anomaly_map=amap,
        threshold=float(active.get("threshold", 0.5)),
        score_min=float(active.get("score_min", 0.0)),
        score_max=float(active.get("score_max", 1.0)),
    )
    return StreamingResponse(_pil_to_png_stream(overlay_pil), media_type="image/png")


# ---------------------------------------------------------------------------
# 탭2 — 검사 이력
# ---------------------------------------------------------------------------

@router.get("/api/inspection/records")
def get_records(verdict: str = "전체") -> list[dict]:
    """
    verdict query: "양품" | "불량" | "전체" (default)
    image_path 는 클라이언트에 불필요하므로 제외하여 반환.
    seq 역순 (FR-INSP-T2-01).
    """
    records = get_state()["insp_records"]
    if verdict == "양품":
        records = [r for r in records if r.get("verdict") == "양품"]
    elif verdict == "불량":
        records = [r for r in records if r.get("verdict") == "불량"]
    return [
        {k: v for k, v in r.items() if k != "image_path"}
        for r in reversed(records)
    ]


@router.get("/api/inspection/records/csv")
def download_records_csv():
    """이력 전체 UTF-8 BOM CSV 다운로드."""
    records  = get_state()["insp_records"]
    csv_bytes = _build_csv(records)
    filename  = f"inspection_history_{datetime.now(tz=KST).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv; charset=utf-8-sig",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.delete("/api/inspection/records")
def clear_records() -> dict:
    """이력 + 풀 + 마지막 결과 초기화. insp_active_model 유지 (R-INSP-05)."""
    reset_inspection_state()
    return {"success": True}


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _resolve_threshold(experiment: dict) -> float:
    """insp_tab3_model._resolve_threshold() 이식."""
    method = experiment.get("threshold_method", "absolute")
    value  = float(experiment.get("threshold_value", 0.5))

    if method == "absolute":
        return value

    metrics       = experiment.get("metrics") or {}
    scores        = metrics.get("anomaly_scores", [])
    labels        = metrics.get("image_labels", [])

    if scores and labels and len(scores) == len(labels):
        normal_scores = [s for s, l in zip(scores, labels) if l == 0]
        if normal_scores:
            return float(np.percentile(normal_scores, value))

    return value


def _make_anomaly_overlay(
    image_path: str,
    anomaly_map: np.ndarray,
    threshold: float,
    score_min: float,
    score_max: float,
    alpha: float = 0.45,
):
    """
    insp_tab1_realtime._make_anomaly_overlay() 이식.
    이상 영역(threshold 초과)에 빨간 반투명 오버레이 적용.
    """
    import cv2
    from PIL import Image as PIL_Image

    orig     = PIL_Image.open(image_path).convert("RGB")
    orig_arr = np.array(orig, dtype=np.float32)
    h, w     = orig_arr.shape[:2]

    if score_max > score_min:
        amap_norm = np.clip(
            (anomaly_map - score_min) / (score_max - score_min), 0.0, 1.0
        ).astype(np.float32)
    else:
        amap_norm = np.zeros_like(anomaly_map, dtype=np.float32)

    amap_resized    = cv2.resize(amap_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    mask            = amap_resized >= threshold

    result          = orig_arr.copy()
    result[mask, 0] = result[mask, 0] * (1 - alpha) + 255 * alpha
    result[mask, 1] = result[mask, 1] * (1 - alpha)
    result[mask, 2] = result[mask, 2] * (1 - alpha)

    return PIL_Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def _pil_to_png_stream(pil_image) -> io.BytesIO:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _build_csv(records: list[dict]) -> bytes:
    """insp_tab2_history._build_csv() 이식."""
    if not records:
        return (",".join(_CSV_COLUMNS) + "\n").encode("utf-8-sig")
    rows = [
        {col: records[i].get(_KEY_MAP[col], "") for col in _CSV_COLUMNS}
        for i in range(len(records))
    ]
    return pd.DataFrame(rows)[_CSV_COLUMNS].to_csv(index=False).encode("utf-8-sig")


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
