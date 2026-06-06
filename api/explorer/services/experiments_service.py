"""
api/explorer/services/experiments_service.py

실험 히스토리 비즈니스 로직. HTTP 관심사와 분리.

예외 변환 규칙 (라우터 기준):
    LookupError  → 404
    ValueError   → 400
    RuntimeError → 500
"""
from __future__ import annotations

import shutil
from pathlib import Path

from utils.storage import check_disk_space, delete_experiment, load_history


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


def save_model_to_path(exp_id: str, save_path: str) -> dict:
    """
    지정 경로에 model_state_dict.pth + configs.yaml 복사.

    Returns:
        {"saved_path": str, "size_mb": float, "warning": str | None}

    Raises:
        LookupError: exp_id 에 해당하는 실험 없음
        ValueError:  디스크 공간 100 MB 미만
        RuntimeError: 파일 복사 실패
    """
    all_records = load_history()
    record = next((r for r in all_records if r.get("experiment_id") == exp_id), None)
    if record is None:
        raise LookupError(f"실험을 찾을 수 없습니다: {exp_id}")

    src_dir = Path(record.get("model_path", ""))
    dst_dir = Path(save_path)

    check_path = str(dst_dir.parent) if dst_dir.parent.exists() else "."
    ok, free_mb = check_disk_space(required_mb=100.0, path=check_path)
    if not ok:
        raise ValueError(
            f"디스크 여유 공간이 부족합니다 ({free_mb:.0f} MB). 100 MB 이상 필요합니다."
        )

    model_type = record.get("model_type", "")
    warn_mb = 1024.0 if model_type == "patchcore" else 500.0
    warning: str | None = None
    if free_mb < warn_mb:
        warning = (
            f"디스크 여유 공간이 {free_mb:.0f} MB입니다. "
            f"{warn_mb:.0f} MB 이상 권장합니다."
        )

    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        total_bytes = 0
        for fname in ("model_state_dict.pth", "configs.yaml"):
            src = src_dir / fname
            if not src.exists():
                continue
            dst = dst_dir / fname
            if dst_dir.resolve() != src_dir.resolve():
                shutil.copy2(src, dst)
            total_bytes += (dst if dst.exists() else src).stat().st_size
    except Exception as e:
        raise RuntimeError(f"모델 저장 실패: {e}") from e

    return {
        "saved_path": str(dst_dir),
        "size_mb": round(total_bytes / (1024 ** 2), 2),
        "warning": warning,
    }
