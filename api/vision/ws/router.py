"""
api/vision/ws/router.py

WebSocket 라우터 — APIRouter 로 등록하여 main.py 와 일관성 유지.
"""
from fastapi import APIRouter, WebSocket

from api.vision.ws.auto_inspection import auto_inspection_ws

router = APIRouter()


@router.websocket("/ws/inspection/auto")
async def ws_auto_inspection(websocket: WebSocket) -> None:
    await auto_inspection_ws(websocket)
