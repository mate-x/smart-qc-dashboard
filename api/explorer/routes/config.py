"""
api/explorer/routes/config.py  — HTTP 레이어 전담

탭2 · 설정:
    GET  /api/config              현재 설정 + device_info 조회
    POST /api/config              설정 저장 (preprocessing + model)
    POST /api/config/preview      threshold 기준 정상/결함 비율 미리보기
    POST /api/config/yaml/save    서버 상태 → configs.yaml 저장
    POST /api/config/yaml/load    configs.yaml → 서버 상태 반영 후 반환
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.explorer.schemas import (
    GetConfigResponse,
    LoadYamlResponse,
    PreviewThresholdRequest,
    PreviewThresholdResponse,
    SaveConfigRequest,
)
from api.explorer.services.config_service import (
    get_config,
    load_config_yaml,
    preview_threshold,
    save_config,
    save_config_yaml,
)

router = APIRouter(prefix="/api/config", tags=["탭2 · 설정"])


@router.get("", summary="현재 설정 조회")
def get_config_route() -> GetConfigResponse:
    return GetConfigResponse(**get_config())


@router.post("", summary="설정 저장")
def save_config_route(body: SaveConfigRequest) -> GetConfigResponse:
    save_config(body.preprocessing_config, body.model_config)
    return GetConfigResponse(**get_config())


@router.post("/preview", summary="Threshold 비율 미리보기")
def preview_threshold_route(body: PreviewThresholdRequest) -> PreviewThresholdResponse:
    result = preview_threshold(body.threshold_method, body.threshold_value)
    return PreviewThresholdResponse(**result)


@router.post("/yaml/save", summary="configs.yaml 저장")
def save_yaml_route() -> dict:
    try:
        save_config_yaml()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"success": True}


@router.post("/yaml/load", summary="configs.yaml 불러오기")
def load_yaml_route() -> LoadYamlResponse:
    try:
        result = load_config_yaml()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return LoadYamlResponse(**result)
