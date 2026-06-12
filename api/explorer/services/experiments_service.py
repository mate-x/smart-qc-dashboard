"""
api/explorer/services/experiments_service.py

실험 히스토리 비즈니스 로직. HTTP 관심사와 분리.

예외 변환 규칙 (라우터 기준):
    LookupError  → 404
    ValueError   → 400
    RuntimeError → 500
"""
from __future__ import annotations

from utils.storage import delete_experiment, load_history


def get_experiments() -> list[dict]:
    """history.json 전체 반환. created_at 역순 정렬."""
    experiments = load_history()
    experiments.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return experiments


def remove_experiment(exp_id: str) -> None:
    """
    history.json 제거 + 모델 디렉토리 삭제 + 로그 파일 삭제.

    Raises:
        LookupError: exp_id 에 해당하는 실험 없음
    """
    all_records = load_history()
    record = next((r for r in all_records if r.get("experiment_id") == exp_id), None)
    if record is None:
        raise LookupError(f"실험을 찾을 수 없습니다: {exp_id}")

    model_path = record.get("model_path")
    delete_experiment(exp_id, model_path)

