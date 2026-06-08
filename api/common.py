"""
api/common.py

vision / explorer 양쪽에서 공유하는 유틸리티.
"""
from __future__ import annotations

import torch

WARN_THRESHOLD_MB = 1024  # 1 GB 미만이면 경고


def get_gpu_memory_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "free_mb":  round(free  / 1024 ** 2),
        "total_mb": round(total / 1024 ** 2),
        "used_mb":  round((total - free) / 1024 ** 2),
    }


def get_gpu_warning() -> str | None:
    """
    여유 GPU 메모리가 WARN_THRESHOLD_MB 미만이면 경고 문자열 반환.
    충분하거나 GPU 없으면 None.
    """
    info = get_gpu_memory_info()
    if not info["available"]:
        return None
    if info["free_mb"] < WARN_THRESHOLD_MB:
        return (
            f"GPU 여유 메모리 {info['free_mb']} MB. "
            "모델 로드에 실패할 수 있습니다."
        )
    return None


def get_device() -> str:
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
