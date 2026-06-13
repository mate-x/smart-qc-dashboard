"""
api/explorer/routes/export.py  — HTTP 레이어 전담

탭4 · 모델 내보내기:
    GET  /api/export/job/{job_id}   내보내기 job 상태 조회
    POST /api/export/{exp_id}       내보내기 job 시작 → job_id

주의: 정적 prefix(/job/)를 동적 /{exp_id}/ 보다 먼저 선언해야
FastAPI가 정적 경로를 우선 매칭한다.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.explorer.jobs import get_job
from api.explorer.schemas import ExportJobStatusResponse, ExportRequest, ExportResponse
from api.explorer.services.export_service import start_export

router = APIRouter(prefix="/api/export", tags=["탭4 · 모델 내보내기"])


# ── 정적 prefix 경로 — /{exp_id} 보다 먼저 선언 ────────────────────────────────

@router.get("/job/{job_id}", summary="내보내기 job 상태 조회")
def get_export_job_route(job_id: str) -> ExportJobStatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job_id를 찾을 수 없습니다: {job_id}")
    return ExportJobStatusResponse(
        status=job["status"],
        error=job.get("error"),
        result=job.get("result"),
    )


# ── 동적 경로 ─────────────────────────────────────────────────────────────────

@router.post("/{exp_id}", summary="모델 내보내기 job 시작")
async def start_export_route(exp_id: str, body: ExportRequest) -> ExportResponse:
    try:
        job_id = await start_export(exp_id, body.format)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ExportResponse(job_id=job_id)
