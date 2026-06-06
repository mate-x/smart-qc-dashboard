"""
api/explorer/state.py

Explorer 도메인 in-memory 전역 상태.
- preprocessing_config, model_config: 탭2 설정 저장
- device_info: GPU/CPU 감지 결과 캐시 (1회)
- experiment_queue: 실험 대기열
"""
from __future__ import annotations

_state: dict = {
    "preprocessing_config": None,   # dict | None
    "model_config":         None,   # dict | None
    "device_info":          None,   # dict | None  — 첫 GET /api/config 시 캐싱
    "experiment_queue":     [],     # list[dict]
}


def get_state() -> dict:
    return _state
