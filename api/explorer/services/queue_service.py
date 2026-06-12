"""
api/explorer/services/queue_service.py

탭2 · 큐:
    get_queue()                      대기열 전체 반환
    add_to_queue(pre_cfg, mdl_cfg)   항목 추가
    remove_from_queue(item_id)       항목 삭제 ("pending" 상태만 가능)
"""
from __future__ import annotations

import uuid

from api.explorer.state import get_state


def get_queue() -> list[dict]:
    return get_state()["experiment_queue"]


def add_to_queue(preprocessing_config: dict, model_config: dict, set_id: str | None = None, name: str | None = None) -> dict:
    model_type = model_config.get("model_type", "model")
    auto_name = f"{model_type.upper()}_{uuid.uuid4().hex[:4]}"
    item = {
        "id":                   str(uuid.uuid4()),
        "name":                 name or auto_name,
        "preprocessing_config": dict(preprocessing_config),
        "model_cfg":            dict(model_config),
        "status":               "pending",
        "set_id":               set_id,
    }
    get_state()["experiment_queue"].append(item)
    return item


def remove_from_queue(item_id: str) -> None:
    """
    Raises:
        LookupError: 항목 없음
        ValueError:  "pending" 외 상태는 삭제 불가
    """
    queue = get_state()["experiment_queue"]
    for i, item in enumerate(queue):
        if item["id"] == item_id:
            if item["status"] != "pending":
                raise ValueError(f"'{item['status']}' 상태의 항목은 삭제할 수 없습니다.")
            queue.pop(i)
            return
    raise LookupError(f"항목을 찾을 수 없습니다: {item_id}")


def move_queue_item(item_id: str, direction: str) -> None:
    """
    pending 항목을 위(up) 또는 아래(down)로 이동한다.
    pending 항목끼리의 상대 순서만 변경하며, 다른 상태 항목은 건너뛴다.

    Raises:
        LookupError: 항목 없음 또는 pending 아님
        ValueError:  direction 값이 잘못됨 또는 이미 경계
    """
    if direction not in ("up", "down"):
        raise ValueError(f"direction은 'up' 또는 'down'이어야 합니다: {direction!r}")

    queue = get_state()["experiment_queue"]
    pending_indices = [i for i, item in enumerate(queue) if item["status"] == "pending"]

    for pos, idx in enumerate(pending_indices):
        if queue[idx]["id"] == item_id:
            if direction == "up":
                if pos == 0:
                    raise ValueError("이미 첫 번째 항목입니다.")
                prev_idx = pending_indices[pos - 1]
                queue[idx], queue[prev_idx] = queue[prev_idx], queue[idx]
            else:
                if pos == len(pending_indices) - 1:
                    raise ValueError("이미 마지막 항목입니다.")
                next_idx = pending_indices[pos + 1]
                queue[idx], queue[next_idx] = queue[next_idx], queue[idx]
            return

    raise LookupError(f"pending 항목을 찾을 수 없습니다: {item_id}")
