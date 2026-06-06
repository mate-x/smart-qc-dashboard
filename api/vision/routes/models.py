"""
api/vision/routes/models.py

GET /api/models — completed 실험 목록 (F1 내림차순, FR-INSP-T3-01)
"""
from __future__ import annotations

from fastapi import APIRouter

from utils.storage import load_history

router = APIRouter()


@router.get("/api/models", summary="완료된 실험 목록 조회", tags=["탭3 · 모델 교체"])
def list_models() -> list[dict]:
    """
    history.json에서 status == "completed" 실험만 반환.
    F1 내림차순 정렬.

    metrics.anomaly_scores / image_labels 는 POST /api/inspection/model 에서
    threshold 재계산 시 history.json을 직접 참조하므로 여기선 그대로 포함.
    """
    all_records = load_history()
    completed = [r for r in all_records if r.get("status") == "completed"]
    completed.sort(
        key=lambda r: (r.get("metrics") or {}).get("f1_score", 0.0),
        reverse=True,
    )
    for r in completed:
        r["created_at"] = _fmt_datetime(r.get("created_at", ""))
    return completed


def _fmt_datetime(value: str) -> str:
    """ISO 8601 → 'YYYY-MM-DD HH:MM:SS'. 이미 변환됐거나 빈 값이면 그대로 반환."""
    if "T" in value:
        return value.replace("T", " ").split("+")[0].split(".")[0][:19]
    return value
