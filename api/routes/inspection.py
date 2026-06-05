"""
api/routes/inspection.py  — HTTP 레이어 전담

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
from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from api.schemas import (
    ActiveModelResponse,
    ApplyModelRequest,
    ApplyModelResponse,
    ClearRecordsResponse,
)
from api.services.inspection_service import apply_model, run_single_inspection
from api.state import get_state, reset_records_only
from utils.image_utils import anomaly_map_to_heatmap, make_anomaly_overlay, pil_to_png_stream

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

@router.post("/api/inspection/model", response_model=ApplyModelResponse, summary="모델 적용", tags=["탭3 · 모델 교체"])
def apply_model_endpoint(req: ApplyModelRequest) -> dict:
    try:
        return apply_model(req.experiment_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/inspection/model", response_model=ActiveModelResponse, summary="현재 적용 모델 조회", tags=["탭3 · 모델 교체"])
def get_active_model() -> dict:
    return {"active_model": get_state()["insp_active_model"]}


# ---------------------------------------------------------------------------
# 탭1 — 실시간 검사
# ---------------------------------------------------------------------------

@router.post("/api/inspection/run", summary="수동 검사 1회 실행", tags=["탭1 · 실시간 검사"])
def run_inspection_endpoint() -> dict:
    """핵심 로직은 run_single_inspection()에 분리 (WebSocket 공용)."""
    try:
        return run_single_inspection()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/inspection/image/last", summary="마지막 원본 이미지 조회", tags=["탭1 · 실시간 검사"])
def get_last_image():
    state = get_state()
    last  = state.get("insp_last_result")
    if last is None:
        raise HTTPException(status_code=404, detail="검사 이력이 없습니다.")
    path = Path(last["image_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="이미지 파일을 찾을 수 없습니다.")
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".bmp": "image/bmp"}
    media     = media_map.get(path.suffix.lower(), "image/png")
    return FileResponse(str(path), media_type=media)


@router.get("/api/inspection/anomaly-map/last", summary="마지막 Anomaly Map 조회", tags=["탭1 · 실시간 검사"])
def get_last_anomaly_map():
    state = get_state()
    amap  = state.get("insp_last_anomaly_map")
    if amap is None:
        raise HTTPException(status_code=404, detail="Anomaly Map이 없습니다.")
    return StreamingResponse(pil_to_png_stream(anomaly_map_to_heatmap(amap)), media_type="image/png")


@router.get("/api/inspection/overlay/last", summary="마지막 이상영역 오버레이 조회", tags=["탭1 · 실시간 검사"])
def get_last_overlay():
    state  = get_state()
    last   = state.get("insp_last_result")
    amap   = state.get("insp_last_anomaly_map")
    active = state.get("insp_active_model")

    if last is None or amap is None:
        raise HTTPException(status_code=404, detail="검사 이력이 없습니다.")
    if active is None:
        raise HTTPException(status_code=400, detail="모델이 선택되지 않았습니다.")

    overlay_pil = make_anomaly_overlay(
        image_path=last["image_path"],
        anomaly_map=amap,
        threshold=float(active.get("threshold", 0.5)),
        score_min=float(active.get("score_min", 0.0)),
        score_max=float(active.get("score_max", 1.0)),
    )
    return StreamingResponse(pil_to_png_stream(overlay_pil), media_type="image/png")


# ---------------------------------------------------------------------------
# 탭2 — 검사 이력
# ---------------------------------------------------------------------------

@router.get("/api/inspection/records", summary="검사 이력 목록 조회", tags=["탭2 · 검사 이력"])
def get_records(verdict: Literal["양품", "불량", "전체"] = "전체") -> list[dict]:
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


@router.get("/api/inspection/records/csv", summary="검사 이력 CSV 다운로드", tags=["탭2 · 검사 이력"])
def download_records_csv():
    """이력 전체 UTF-8 BOM CSV 다운로드."""
    records   = get_state()["insp_records"]
    csv_bytes = _build_csv(records)
    filename  = f"inspection_history_{datetime.now(tz=KST).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv; charset=utf-8-sig",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.delete("/api/inspection/records", response_model=ClearRecordsResponse, summary="검사 이력 초기화", tags=["탭2 · 검사 이력"])
def clear_records() -> dict:
    """이력만 초기화. test_pool·active_model 유지 (초기화 후 즉시 재검사 가능)."""
    reset_records_only()
    return {"success": True}


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _build_csv(records: list[dict]) -> bytes:
    if not records:
        return (",".join(_CSV_COLUMNS) + "\n").encode("utf-8-sig")
    rows = [
        {col: records[i].get(_KEY_MAP[col], "") for col in _CSV_COLUMNS}
        for i in range(len(records))
    ]
    return pd.DataFrame(rows)[_CSV_COLUMNS].to_csv(index=False).encode("utf-8-sig")
