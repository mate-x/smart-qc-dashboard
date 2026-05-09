from __future__ import annotations

import time

import streamlit as st

MAX_ANOMALY_MAP_CACHE: int = 3  # LRU 캐시 최대 항목 수
_KEY_PREFIX = "_anomaly_maps_"


def set_anomaly_map_cache(exp_id: str, data: dict) -> None:
    """
    Anomaly Map 캐시 저장. MAX_ANOMALY_MAP_CACHE 초과 시 가장 오래된 항목 LRU 제거.
    session_state 키: "_anomaly_maps_{exp_id}"
    """
    key = f"{_KEY_PREFIX}{exp_id}"
    st.session_state[key] = {**data, "cached_at": time.time()}
    _evict_if_needed()


def get_anomaly_map_cache(exp_id: str) -> dict | None:
    """캐시 항목 반환. 없으면 None."""
    return st.session_state.get(f"{_KEY_PREFIX}{exp_id}")


def _evict_if_needed() -> None:
    """MAX_ANOMALY_MAP_CACHE 초과 시 cached_at 기준으로 가장 오래된 항목 제거."""
    cached_keys = [k for k in st.session_state if k.startswith(_KEY_PREFIX)]
    while len(cached_keys) > MAX_ANOMALY_MAP_CACHE:
        oldest = min(
            cached_keys,
            key=lambda k: st.session_state[k].get("cached_at", 0),
        )
        del st.session_state[oldest]
        cached_keys.remove(oldest)
