"""
api/explorer/routes/dataset.py  — HTTP 레이어 전담

탭1 · 데이터셋:
    POST /api/dataset/validate                경로 검증 + 메타 반환
    GET  /api/dataset/thumbnail/{class_name}  클래스 대표 썸네일
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.explorer.schemas import ValidateDatasetRequest, ValidateDatasetResponse
from api.explorer.services.dataset_service import get_thumbnail, validate_dataset
from utils.image_utils import pil_to_png_stream

router = APIRouter(prefix="/api/dataset", tags=["탭1 · 데이터셋"])


@router.post("/validate", summary="데이터셋 경로 검증")
def validate_dataset_route(body: ValidateDatasetRequest) -> ValidateDatasetResponse:
    try:
        result = validate_dataset(body.path, body.product_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ValidateDatasetResponse(**result)


@router.get("/thumbnail/{class_name}", summary="클래스 대표 썸네일")
def get_thumbnail_route(
    class_name: str,
    dataset_path: str = Query(..., description="데이터셋 루트 경로"),
) -> StreamingResponse:
    try:
        img = get_thumbnail(dataset_path, class_name)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return StreamingResponse(pil_to_png_stream(img), media_type="image/png")
