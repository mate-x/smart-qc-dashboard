"""
api/main.py

FastAPI 앱 조립 + CORS.

실행 (프로젝트 루트에서):
    uvicorn api.main:app --reload --port 8000

주의: storage.py 상대경로 의존으로 반드시 프로젝트 루트에서 실행할 것.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.vision.routes.inspection import router as inspection_router
from api.vision.routes.models import router as models_router
from api.vision.ws.router import router as vision_ws_router
from api.explorer.routes.experiments import router as experiments_router
from api.explorer.routes.dataset import router as dataset_router
from api.explorer.routes.config import router as config_router
from api.explorer.routes.queue import router as queue_router
from api.explorer.routes.anomaly_map import router as anomaly_map_router
from api.explorer.routes.training import router as training_router
from api.explorer.ws.router import router as explorer_ws_router

app = FastAPI(
    title="스마트 QC 검사 API",
    description=(
        "비전 검사 시스템 REST API\n\n"
        "- **탭1** 실시간 검사 — 수동/자동 추론, 이미지·Anomaly Map 조회\n"
        "- **탭2** 검사 이력 — 목록 조회·필터·CSV 다운로드·초기화\n"
        "- **탭3** 모델 교체 — 완료된 실험 적용 및 현재 모델 조회\n"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inspection_router)
app.include_router(models_router)
app.include_router(vision_ws_router)
app.include_router(experiments_router)
app.include_router(dataset_router)
app.include_router(config_router)
app.include_router(queue_router)
app.include_router(anomaly_map_router)
app.include_router(training_router)
app.include_router(explorer_ws_router)
