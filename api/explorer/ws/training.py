"""
api/explorer/ws/training.py

WS /ws/training — 학습 progress 스트리밍 (push-only, 서버 → 클라이언트)

프로토콜:
  연결 직후 서버 → 클라이언트:
    {"type": "snapshot", "status": ..., "exp_id": ..., "progress": ...,
     "current_stage_idx": ..., "current_stage_name": ...,
     "log_lines": [...], "loss_history": [...], "last_ckpt_path": ...}

  이후 서버 → 클라이언트 (학습 진행 중 실시간):
    {"type": "progress",  "step": int, "total": int, "loss": float, "elapsed": float}
    {"type": "log",       "message": str}
    {"type": "stage",     "stage_idx": int, "stage_name": str}
    {"type": "paused",    "step": int, "ckpt_path": str}
    {"type": "completed", "exp_id": str, "auc": float, "duration_seconds": int, "message": str}
    {"type": "stopped",   "step": int}
    {"type": "error",     "message": str, "traceback": str}
    {"type": "batch_item_started",  "exp_id": str, "queue_idx": int}
    {"type": "batch_item_skipped"}
    {"type": "batch_item_error",    "traceback": str}
    {"type": "batch_stopped",       "step": int}
    {"type": "batch_completed",     "completed": int, "failed": int, "skipped": int}

단일 사용자 가정: 동시에 하나의 WS 연결만 유지.
새 연결이 들어오면 이전 연결의 브로드캐스트 큐를 교체한다.
"""
from __future__ import annotations

import asyncio

from fastapi import WebSocket, WebSocketDisconnect

from api.explorer.services.training_service import (
    get_status,
    register_ws_queue,
    unregister_ws_queue,
)


async def training_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    # 재연결 대비: 현재 상태 스냅샷을 즉시 전송
    snapshot = get_status()
    await websocket.send_json({"type": "snapshot", **snapshot})

    # 이 연결 전용 브로드캐스트 큐 등록
    q: asyncio.Queue = asyncio.Queue()
    register_ws_queue(q)

    try:
        while True:
            msg = await q.get()
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        unregister_ws_queue()
