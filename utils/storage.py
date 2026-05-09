from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import torch
import yaml

HISTORY_FILE = Path("./experiments/history.json")
MODELS_DIR = Path("./models")
LOGS_DIR = Path("./logs")
IMAGENET_PENALTY_DIR = Path("./dataset/imagenet_penalty")

_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(records: list[dict]) -> None:
    # R-ATOMIC-01: tmpfile → rename
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp.replace(HISTORY_FILE)


def append_experiment(record: dict) -> None:
    records = load_history()
    records.append(record)
    save_history(records)


def validate_imagenet_penalty_dir() -> None:
    """IMAGENET_PENALTY_DIR 유효성 검증. 테스트 시 monkeypatch 가능."""
    p = IMAGENET_PENALTY_DIR
    if not p.exists():
        raise ValueError(
            f"ImageNet penalty 디렉터리가 존재하지 않습니다: {p.resolve()}"
        )
    images = [f for f in p.rglob("*") if f.suffix.lower() in _SUPPORTED_IMAGE_EXTS]
    if not images:
        raise ValueError(
            f"ImageNet penalty 디렉터리에 이미지 파일이 없습니다: {p.resolve()}"
        )


def save_completed_experiment(
    exp_id: str,
    model: object,
    record: dict,
) -> None:
    """
    3단계 저장 프로토콜 (05_Data_Model 3단계).
      Stage 1. model_state_dict.pth 저장
      Stage 2. configs.yaml 저장 (R-ATOMIC-01)
      Stage 3. history.json append
    Stage 1 실패 시 디렉터리 정리 후 예외 재발생.
    """
    model_dir = MODELS_DIR / exp_id
    try:
        model_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: 모델 가중치
        pth_path = model_dir / "model_state_dict.pth"
        torch.save(model.state_dict(), pth_path)

        # Stage 2: configs.yaml (R-ATOMIC-01)
        configs_data = {
            "experiment": {
                "name": record.get("name", exp_id),
                "created_at": record.get("created_at", ""),
            },
            "preprocessing": record.get("preprocessing_config", {}),
            "model": record.get("model_config", {}),
        }
        configs_path = model_dir / "configs.yaml"
        tmp = configs_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(configs_data, f, allow_unicode=True, default_flow_style=False)
        tmp.replace(configs_path)

        # Stage 3: history
        record["model_path"] = str(model_dir)
        record["configs_path"] = str(configs_path)
        append_experiment(record)

    except Exception:
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        raise


# ── 추가 스토리지 함수 ─────────────────────────────────────────────────────────


def delete_experiment_from_history(experiment_id: str) -> bool:
    """해당 ID 제거 후 원자적 쓰기. Returns True(제거 성공) | False(ID 없음)."""
    records = load_history()
    new_records = [r for r in records if r.get("experiment_id") != experiment_id]
    if len(new_records) == len(records):
        return False
    save_history(new_records)
    return True


def prepare_model_dir(experiment_id: str) -> Path:
    """./models/{experiment_id}/ 생성 후 Path 반환. 중복 경로 존재 시 RuntimeError."""
    model_dir = MODELS_DIR / experiment_id
    if model_dir.exists():
        raise RuntimeError(
            f"모델 디렉터리가 이미 존재합니다: {model_dir.resolve()}"
        )
    model_dir.mkdir(parents=True, exist_ok=False)
    return model_dir


def delete_experiment(experiment_id: str, model_path: str | None = None) -> None:
    """history.json 제거 + 모델 디렉터리 삭제 + 로그 파일 삭제. ignore_errors=True."""
    delete_experiment_from_history(experiment_id)
    if model_path:
        p = Path(model_path)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    log_file = LOGS_DIR / f"{experiment_id}.log"
    if log_file.exists():
        try:
            log_file.unlink()
        except OSError:
            pass


def get_log_writer(experiment_id: str) -> io.TextIOWrapper:
    """./logs/{experiment_id}.log append 모드 파일 객체 반환 (line-buffered)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"{experiment_id}.log"
    return open(log_file, "a", encoding="utf-8", buffering=1)


def read_log_tail(experiment_id: str, n_lines: int = 100) -> str:
    """최신 n_lines줄 반환. 파일 미존재 시 빈 문자열."""
    log_file = LOGS_DIR / f"{experiment_id}.log"
    if not log_file.exists():
        return ""
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except OSError:
        return ""


def check_disk_space(
    required_mb: float = 500.0,
    path: str = ".",
) -> tuple[bool, float]:
    """Returns: (충분 여부, 여유 공간 MB)."""
    try:
        usage = shutil.disk_usage(path)
        free_mb = usage.free / (1024 * 1024)
        return free_mb >= required_mb, free_mb
    except OSError:
        return False, 0.0


def check_disk_before_save(model_type: str) -> None:
    """
    100 MB 미만: RuntimeError raise.
    500 MB 미만: st.warning() 표시 (저장 허용).
    Streamlit context에서만 호출.
    """
    import streamlit as st

    ok, free_mb = check_disk_space(required_mb=100.0)
    if not ok:
        raise RuntimeError(
            f"ERR_DISK_SPACE: 디스크 여유 공간이 부족합니다 ({free_mb:.0f} MB). "
            "모델 저장에 최소 100 MB가 필요합니다."
        )
    if free_mb < 500.0:
        st.warning(
            f"디스크 여유 공간이 {free_mb:.0f} MB입니다. "
            "저장은 허용되지만 500 MB 이상 권장합니다."
        )
