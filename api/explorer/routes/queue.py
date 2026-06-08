"""
api/explorer/routes/queue.py  — HTTP 레이어 전담

탭2 · 큐:
    GET    /api/queue           대기열 전체 조회
    POST   /api/queue           항목 추가
    DELETE /api/queue/{id}      항목 삭제 ("대기중" 상태만 가능)
    PATCH  /api/queue/reorder   항목 순서 변경 ("대기중" 항목만)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.explorer.schemas import (
    AddQueueRequest,
    AddQueueResponse,
    DeleteQueueResponse,
    QueueItemResponse,
    ReorderQueueRequest,
    ReorderQueueResponse,
)
from api.explorer.services.queue_service import (
    add_to_queue,
    get_queue,
    move_queue_item,
    remove_from_queue,
)

router = APIRouter(prefix="/api/queue", tags=["탭2 · 큐"])


@router.get("", summary="대기열 조회", response_model_by_alias=True)
def list_queue_route() -> list[QueueItemResponse]:
    return [QueueItemResponse(**item) for item in get_queue()]


@router.post("", summary="대기열 항목 추가")
def add_queue_route(body: AddQueueRequest) -> AddQueueResponse:
    item = add_to_queue(body.preprocessing_config, body.model_cfg, body.set_id)
    return AddQueueResponse(id=item["id"], name=item["name"])


@router.delete("/{item_id}", summary="대기열 항목 삭제")
def delete_queue_route(item_id: str) -> DeleteQueueResponse:
    try:
        remove_from_queue(item_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return DeleteQueueResponse(success=True)


@router.patch("/reorder", summary="대기열 순서 변경")
def reorder_queue_route(body: ReorderQueueRequest) -> ReorderQueueResponse:
    try:
        move_queue_item(body.item_id, body.direction)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ReorderQueueResponse(success=True)
