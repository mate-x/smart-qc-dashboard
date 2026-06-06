"""
api/explorer/services/queue_service.py

탭2 · 큐:
    get_queue()                      대기열 전체 반환
    add_to_queue(pre_cfg, mdl_cfg)   항목 추가
    remove_from_queue(item_id)       항목 삭제 ("대기중" 상태만 가능)
"""
from __future__ import annotations

import uuid

from api.explorer.state import get_state


def get_queue() -> list[dict]:
    return get_state()["experiment_queue"]


def add_to_queue(preprocessing_config: dict, model_config: dict) -> dict:
    model_type = model_config.get("model_type", "model")
    name = f"{model_type.upper()}_{uuid.uuid4().hex[:4]}"
    item = {
        "id":                   str(uuid.uuid4()),
        "name":                 name,
        "preprocessing_config": dict(preprocessing_config),
        "model_cfg":            dict(model_config),
        "status":               "대기중",
    }
    get_state()["experiment_queue"].append(item)
    return item


def remove_from_queue(item_id: str) -> None:
    """
    Raises:
        LookupError: 항목 없음
        ValueError:  "대기중" 외 상태는 삭제 불가
    """
    queue = get_state()["experiment_queue"]
    for i, item in enumerate(queue):
        if item["id"] == item_id:
            if item["status"] != "대기중":
                raise ValueError(f"'{item['status']}' 상태의 항목은 삭제할 수 없습니다.")
            queue.pop(i)
            return
    raise LookupError(f"항목을 찾을 수 없습니다: {item_id}")
