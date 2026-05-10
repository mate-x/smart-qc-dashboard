from __future__ import annotations

import time

import streamlit as st

MAX_ANOMALY_MAP_CACHE: int = 3  # LRU 캐시 최대 항목 수
_KEY_PREFIX = "_anomaly_maps_"


def set_anomaly_map_cache(exp_id: str, data: dict) -> None:
    """
    Anomaly Map 캐시 저장. MAX_ANOMALY_MAP_CACHE 이상이면 먼저 evict 후 저장 (PRD §8.4).
    session_state 키: "_anomaly_maps_{exp_id}"
    """
    cached_keys = [k for k in st.session_state if k.startswith(_KEY_PREFIX)]
    while len(cached_keys) >= MAX_ANOMALY_MAP_CACHE:
        oldest = min(
            cached_keys,
            key=lambda k: st.session_state[k].get("cached_at", 0),
        )
        del st.session_state[oldest]
        cached_keys.remove(oldest)

    st.session_state[f"{_KEY_PREFIX}{exp_id}"] = {**data, "cached_at": time.time()}


def get_anomaly_map_cache(exp_id: str) -> dict | None:
    """캐시 항목 반환. 없으면 None."""
    return st.session_state.get(f"{_KEY_PREFIX}{exp_id}")


def invalidate_anomaly_map_cache(experiment_id: str) -> None:
    """특정 실험의 캐시 제거. 키 없으면 no-op. 실험 삭제 직후 호출."""
    key = f"{_KEY_PREFIX}{experiment_id}"
    if key in st.session_state:
        del st.session_state[key]
