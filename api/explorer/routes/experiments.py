"""
api/explorer/routes/experiments.py  — HTTP 레이어 전담

탭4 · 실험 히스토리:
    GET    /api/experiments              실험 목록 (created_at 역순)
    DELETE /api/experiments/{exp_id}     실험 삭제
    POST   /api/experiments/{exp_id}/save 모델 파일 저장
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.explorer.schemas import DeleteExperimentResponse, SaveModelRequest, SaveModelResponse
from api.explorer.services.experiments_service import (
    get_experiments,
    remove_experiment,
    save_model_to_path,
)

router = APIRouter(prefix="/api/experiments", tags=["탭4 · 실험 히스토리"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", summary="실험 목록 조회")
def list_experiments() -> list[dict]:
    return get_experiments()


@router.delete("/{exp_id}", summary="실험 삭제")
def delete_experiment_route(exp_id: str) -> DeleteExperimentResponse:
    try:
        remove_experiment(exp_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return DeleteExperimentResponse(success=True)


@router.post("/{exp_id}/save", summary="모델 파일 저장")
def save_model_route(exp_id: str, body: SaveModelRequest) -> SaveModelResponse:
    try:
        result = save_model_to_path(exp_id, body.save_path)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return SaveModelResponse(success=True, **result)
