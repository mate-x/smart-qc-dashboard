"""
api/ws/auto_inspection.py

WS /ws/inspection/auto — 자동 검사 루프

프로토콜:
  Client → Server : 텍스트 "start" | "stop"
  Server → Client : JSON (type 필드로 구분)

    {"type": "result", "seq": int, "inspected_at": str, "image_name": str,
     "verdict": str, "anomaly_score": float, "was_reshuffled": bool}

    {"type": "defect_stopped"}           — 불량 감지로 루프 자동 중지
    {"type": "stopped"}                  — "stop" 메시지 수신 후 확인
    {"type": "error",  "message": str}   — 검사 중 RuntimeError

주의: run_single_inspection()은 CPU/GPU 블로킹 작업 → asyncio.to_thread() 필수.
"""
from __future__ import annotations

import asyncio

from fastapi import WebSocket, WebSocketDisconnect

from api.services.inspection_service import run_single_inspection
from api.state import get_state

_INSPECTION_INTERVAL = 3.0   # 자동 검사 간격 (초)
_STOP_POLL_TIMEOUT   = 3.0   # 검사 완료 후 "stop" 수신 대기 시간 (= 검사 간격과 동일)


async def auto_inspection_ws(websocket: WebSocket) -> None:
    """
    WebSocket 핸들러.

    외부 루프: "start" | "stop" 제어 메시지 대기.
    내부 루프: insp_auto_active == True 동안 검사 반복.
              각 검사 후 _STOP_POLL_TIMEOUT 초 동안 "stop" 대기
              → 수신 없으면 timeout → 다음 검사 실행.
    """
    await websocket.accept()
    state = get_state()
    state["insp_auto_active"] = False

    try:
        while True:
            # 제어 메시지 대기 (블로킹)
            msg = await websocket.receive_text()

            if msg == "start":
                state["insp_auto_active"] = True
            elif msg == "stop":
                state["insp_auto_active"] = False
                await websocket.send_json({"type": "stopped"})
                continue

            # 자동 검사 루프
            while state["insp_auto_active"]:
                # 블로킹 추론을 스레드풀로 실행
                try:
                    result = await asyncio.to_thread(run_single_inspection)
                except RuntimeError as e:
                    state["insp_auto_active"] = False
                    await websocket.send_json({"type": "error", "message": str(e)})
                    break

                await websocket.send_json({"type": "result", **result})

                # 불량 감지 → 루프 자동 중지 (run_single_inspection 내부에서 이미 갱신)
                if result["verdict"] == "불량":
                    await websocket.send_json({"type": "defect_stopped"})
                    break

                # 다음 검사까지 _STOP_POLL_TIMEOUT 초 대기하면서 "stop" 감시
                try:
                    stop_msg = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=_STOP_POLL_TIMEOUT,
                    )
                    if stop_msg == "stop":
                        state["insp_auto_active"] = False
                        await websocket.send_json({"type": "stopped"})
                        break
                except asyncio.TimeoutError:
                    pass   # 타임아웃 → 다음 검사 실행

    except WebSocketDisconnect:
        state["insp_auto_active"] = False
