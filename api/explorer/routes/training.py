"""
api/explorer/routes/training.py

탭3 · Training REST 엔드포인트

단일 학습:
    POST   /api/training/start          학습 시작
    POST   /api/training/resume         체크포인트 재시작
    POST   /api/training/pause          일시정지
    POST   /api/training/unpause        일시정지 해제
    POST   /api/training/stop           학습 중단
    GET    /api/training/status         현재 상태 조회

체크포인트:
    GET    /api/training/checkpoints    체크포인트 목록
    DELETE /api/training/checkpoints/{name}  체크포인트 삭제

배치 학습:
    POST   /api/training/batch/start    배치 시작
    POST   /api/training/batch/skip     현재 항목 건너뜀
    POST   /api/training/batch/stop     전체 배치 중단
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.explorer.schemas import (
    BatchStartResponse,
    CheckpointsResponse,
    DeleteCheckpointResponse,
    ResumeTrainingRequest,
    StartTrainingRequest,
    StartTrainingResponse,
    TrainingControlResponse,
    TrainingStatusResponse,
)
from api.explorer.services import training_service as svc

router = APIRouter(prefix="/api/training", tags=["탭3 · 학습"])


# ---------------------------------------------------------------------------
# 단일 학습
# ---------------------------------------------------------------------------

@router.post("/start", response_model=StartTrainingResponse, status_code=201, summary="학습 시작")
async def start_training(body: StartTrainingRequest):
    try:
        exp_id, model_type = svc.start_training(body.experiment_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return StartTrainingResponse(exp_id=exp_id, model_type=model_type)


@router.post("/resume", response_model=StartTrainingResponse, status_code=201, summary="체크포인트 재시작")
async def resume_training(body: ResumeTrainingRequest):
    try:
        exp_id = svc.resume_training(body.checkpoint_name)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return StartTrainingResponse(exp_id=exp_id)


@router.post("/pause", response_model=TrainingControlResponse, summary="일시정지")
async def pause_training():
    try:
        svc.pause_training()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return TrainingControlResponse(success=True, message="일시정지 신호를 전송했습니다.")


@router.post("/unpause", response_model=TrainingControlResponse, summary="일시정지 해제")
async def unpause_training():
    try:
        svc.unpause_training()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return TrainingControlResponse(success=True, message="학습을 재개합니다.")


@router.post("/stop", response_model=TrainingControlResponse, summary="학습 중단")
async def stop_training():
    try:
        svc.stop_training()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return TrainingControlResponse(success=True, message="중지 신호를 전송했습니다.")


@router.get("/status", response_model=TrainingStatusResponse, summary="현재 상태 조회")
async def get_status():
    return TrainingStatusResponse(**svc.get_status())


# ---------------------------------------------------------------------------
# 체크포인트
# ---------------------------------------------------------------------------

@router.get("/checkpoints", response_model=CheckpointsResponse, summary="체크포인트 목록")
async def list_checkpoints():
    items = svc.get_checkpoints()
    return CheckpointsResponse(checkpoints=items)


@router.delete("/checkpoints/{name}", response_model=DeleteCheckpointResponse, summary="체크포인트 삭제")
async def delete_checkpoint(name: str):
    try:
        ok = svc.remove_checkpoint(name)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return DeleteCheckpointResponse(success=ok)


# ---------------------------------------------------------------------------
# 배치 학습
# ---------------------------------------------------------------------------

@router.post("/batch/start", response_model=BatchStartResponse, status_code=201, summary="배치 시작")
async def batch_start():
    try:
        exp_id, batch_total, model_type = svc.start_batch()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return BatchStartResponse(exp_id=exp_id, batch_total=batch_total, model_type=model_type)


@router.post("/batch/skip", response_model=TrainingControlResponse, summary="현재 항목 건너뜀")
async def batch_skip():
    try:
        svc.skip_batch_item()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return TrainingControlResponse(success=True, message="건너뜀 신호를 전송했습니다.")


@router.post("/batch/stop", response_model=TrainingControlResponse, summary="전체 배치 중단")
async def batch_stop():
    try:
        svc.stop_batch_all()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return TrainingControlResponse(success=True, message="전체 배치 중단 신호를 전송했습니다.")
