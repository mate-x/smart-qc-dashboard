"""
api/main.py

FastAPI 앱 조립 + CORS.

실행 (프로젝트 루트에서):
    uvicorn api.main:app --reload --port 8000

주의: storage.py 상대경로 의존으로 반드시 프로젝트 루트에서 실행할 것.
"""
from __future__ import annotations

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from api.routes.inspection import router as inspection_router
from api.routes.models import router as models_router
from api.ws.auto_inspection import auto_inspection_ws

app = FastAPI(title="Smart QC Inspection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models_router)
app.include_router(inspection_router)


@app.websocket("/ws/inspection/auto")
async def ws_auto_inspection(websocket: WebSocket) -> None:
    await auto_inspection_ws(websocket)
