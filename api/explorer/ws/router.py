"""
api/explorer/ws/router.py

Explorer WebSocket 라우터.
"""
from __future__ import annotations

from fastapi import APIRouter, WebSocket

from api.explorer.ws.training import training_ws

router = APIRouter()


@router.websocket("/ws/training")
async def ws_training(websocket: WebSocket) -> None:
    await training_ws(websocket)
