"""
api/routes/models.py

GET /api/models — completed 실험 목록 (F1 내림차순, FR-INSP-T3-01)
"""
from __future__ import annotations

from fastapi import APIRouter

from utils.storage import load_history

router = APIRouter()


@router.get("/api/models")
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
    return completed
