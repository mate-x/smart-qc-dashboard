"""
api/explorer/routes/anomaly_map.py  — HTTP 레이어 전담

탭5 · Anomaly Map:
    POST /api/anomaly-map/{exp_id}/build                  Map 생성 job 시작
    GET  /api/anomaly-map/job/{job_id}                    build/zip job 상태 조회
    GET  /api/anomaly-map/{exp_id}/images                 이미지 목록 + 통계
    GET  /api/anomaly-map/{exp_id}/image/{path:path}/triplet  triplet PNG
    GET  /api/anomaly-map/{exp_id}/export/csv             CSV 다운로드
    POST /api/anomaly-map/{exp_id}/export/zip             ZIP 생성 job 시작
    GET  /api/anomaly-map/zip/{job_id}                    ZIP 다운로드

주의: 정적 prefix(/job/, /zip/)를 동적 /{exp_id}/ 보다 먼저 선언해야
FastAPI가 정적 경로를 우선 매칭한다.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse

from api.explorer.schemas import (
    AnomalyImagesResponse,
    AnomalyMapStatusResponse,
    BuildAnomalyMapResponse,
    ImageRowResponse,
    JobStatusResponse,
    ZipJobResponse,
    ZipRequest,
)
from api.explorer.services.anomaly_map_service import (
    get_build_status,
    get_csv,
    get_images,
    get_job_status,
    get_triplet_image,
    get_zip_result,
    start_build,
    start_zip,
)
from utils.image_utils import pil_to_png_stream

router = APIRouter(prefix="/api/anomaly-map", tags=["탭5 · Anomaly Map"])


# ── 정적 prefix 경로 — /{exp_id}/... 보다 먼저 선언 ───────────────────────────

@router.get("/job/{job_id}", summary="build/zip job 상태 조회")
def get_job_status_route(job_id: str) -> JobStatusResponse:
    try:
        return JobStatusResponse(**get_job_status(job_id))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/zip/{job_id}", summary="ZIP 다운로드")
def get_zip_route(job_id: str) -> StreamingResponse:
    try:
        zip_bytes = get_zip_result(job_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return StreamingResponse(
        iter([zip_bytes]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={job_id}_anomaly_maps.zip"},
    )


# ── /{exp_id}/... 동적 경로 ──────────────────────────────────────────────────

@router.get("/{exp_id}/status", summary="Anomaly Map 빌드 상태 조회")
def get_status_route(exp_id: str) -> AnomalyMapStatusResponse:
    try:
        result = get_build_status(exp_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return AnomalyMapStatusResponse(**result)


@router.post("/{exp_id}/build", summary="Anomaly Map 생성 job 시작")
async def build_route(exp_id: str) -> BuildAnomalyMapResponse:
    try:
        job_id = await start_build(exp_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return BuildAnomalyMapResponse(job_id=job_id)


@router.get("/{exp_id}/images", summary="이미지 목록 + 통계 조회")
def get_images_route(
    exp_id: str,
    threshold: float = Query(..., description="정규화된 threshold (0~1.2)"),
    defect_class: str = Query("전체", description="결함 유형 필터 ('전체' 또는 클래스명)"),
) -> AnomalyImagesResponse:
    try:
        result = get_images(exp_id, threshold, defect_class)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return AnomalyImagesResponse(
        images=[ImageRowResponse(**row) for row in result["images"]],
        score_max=result["score_max"],
        score_avg=result["score_avg"],
        tp=result["tp"],
        fp=result["fp"],
        tn=result["tn"],
        fn=result["fn"],
    )


@router.get("/{exp_id}/image/{image_path:path}/triplet", summary="Triplet PNG 반환")
def get_triplet_route(exp_id: str, image_path: str) -> Response:
    # image_path: "{class_name}/{image_name}"
    parts = image_path.rsplit("/", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="image_path 형식 오류 ('{class}/{filename}' 필요)")
    class_name, image_name = parts

    try:
        triplet = get_triplet_image(exp_id, class_name, image_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))

    buf = pil_to_png_stream(triplet)
    return Response(content=buf.read(), media_type="image/png")


@router.get("/{exp_id}/export/csv", summary="CSV 내보내기")
def get_csv_route(
    exp_id: str,
    threshold: float = Query(...),
    defect_class: str = Query("전체"),
) -> Response:
    try:
        csv_bytes = get_csv(exp_id, threshold, defect_class)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(
        content=csv_bytes,
        media_type="text/csv; charset=utf-8-sig",
        headers={"Content-Disposition": f"attachment; filename={exp_id}_results.csv"},
    )


@router.post("/{exp_id}/export/zip", summary="ZIP 생성 job 시작")
async def start_zip_route(exp_id: str, body: ZipRequest) -> ZipJobResponse:
    try:
        job_id = await start_zip(exp_id, body.threshold, body.defect_class)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ZipJobResponse(job_id=job_id)
