from __future__ import annotations

import io
import json
import logging
import shutil
from pathlib import Path

import torch
import yaml

HISTORY_FILE = Path("./experiments/history.json")
MODELS_DIR = Path("./models")
LOGS_DIR = Path("./logs")
IMAGENET_PENALTY_DIR = Path("./dataset/imagenet_penalty")

_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}  # PRD §10.3

_logger = logging.getLogger(__name__)


def load_history() -> list[dict]:
    """
    실험 히스토리 로드. 파일 미존재 또는 JSON 파싱 실패 시 빈 리스트 반환.
    예외를 전파하지 않는다 — UI 렌더링 중단 방지 (PRD §3.2).
    """
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        _logger.warning("history_load_failed: path=%s", HISTORY_FILE)
        return []


def save_history(records: list[dict]) -> None:
    # R-ATOMIC-01: tmpfile → rename
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp.replace(HISTORY_FILE)


def append_experiment(record: dict) -> None:
    records = load_history()
    exp_id = record.get("experiment_id")
    if exp_id:
        for r in records:
            if r.get("experiment_id") == exp_id:
                raise RuntimeError(f"ERR_DUPLICATE_EXPERIMENT_ID: {exp_id}")
    records.append(record)
    save_history(records)


def validate_imagenet_penalty_dir() -> tuple[bool, int]:
    """
    IMAGENET_PENALTY_DIR 유효성 검증.
    반환: (이미지 존재 여부, 이미지 수) — PRD §9.3, 07 PRD §3.2.
    탭4에서 ok, count = validate_imagenet_penalty_dir() 형식으로 사용.
    """
    p = IMAGENET_PENALTY_DIR
    if not p.exists():
        return False, 0
    count = sum(
        1 for f in p.iterdir() if f.suffix.lower() in _SUPPORTED_IMAGE_EXTS
    )
    return count > 0, count


def save_completed_experiment(
    exp_id: str,
    model: object,
    record: dict,
    preprocessing_config: dict | None = None,
    model_config: dict | None = None,
) -> None:
    """
    3단계 원자성 저장 프로토콜 (05_Data_Model §6).
      Stage 1. model_state_dict.pth 저장
      Stage 2. configs.yaml 스냅샷 저장 (R-ATOMIC-01)
      Stage 3. history.json append

    Stage 1/2 실패: 디렉터리 정리 후 RuntimeError 재발생.
    Stage 3 실패: 모델 파일 보존 + RuntimeError 재발생 (호출자가 UI 경고 표시).
    """
    model_dir = MODELS_DIR / exp_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: 모델 가중치
    pth_path = model_dir / "model_state_dict.pth"
    try:
        torch.save(model.state_dict(), pth_path)
    except Exception as e:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise RuntimeError(f"ERR_MODEL_SAVE_FAILED (Stage1): {e}") from e

    # Stage 2: configs.yaml 스냅샷 (R-ATOMIC-01)
    configs_path = model_dir / "configs.yaml"
    try:
        preproc_data = (
            preprocessing_config
            if preprocessing_config is not None
            else record.get("preprocessing_config", {})
        )
        model_data = (
            model_config
            if model_config is not None
            else record.get("model_config", {})
        )
        configs_data = {
            "experiment": {
                "name": record.get("name", exp_id),
                "created_at": record.get("created_at", ""),
            },
            "preprocessing": preproc_data,
            "model": model_data,
        }
        tmp = configs_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(configs_data, f, allow_unicode=True, default_flow_style=False)
        tmp.replace(configs_path)
    except Exception as e:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise RuntimeError(f"ERR_MODEL_SAVE_FAILED (Stage2): {e}") from e

    # Stage 3: history.json append — 실패해도 모델 파일은 보존
    record["model_path"] = str(model_dir)
    record["configs_path"] = str(configs_path)
    try:
        append_experiment(record)
    except Exception as e:
        raise RuntimeError(
            f"ERR_HISTORY_WRITE_FAILED: 모델 저장 성공, 히스토리 기록 실패. "
            f"model_path={model_dir} — {e}"
        ) from e


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
