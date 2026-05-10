from __future__ import annotations

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
        configs_data = {
            "experiment": {
                "name": record.get("name", exp_id),
                "created_at": record.get("created_at", ""),
            },
            "preprocessing": record.get("preprocessing_config", {}),
            "model": record.get("model_config", {}),
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


def delete_experiment(experiment_id: str, model_path: str | None = None) -> None:
    """
    실험 레코드 및 관련 파일 삭제 (PRD §7.2).
    model_path: experiment_record["model_path"]. None이면 파일 삭제 생략.
    """
    # Stage 1: history.json
    records = load_history()
    filtered = [r for r in records if r["experiment_id"] != experiment_id]
    if len(filtered) < len(records):
        path = HISTORY_FILE
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    # Stage 2: 모델 디렉터리
    if model_path:
        model_dir = Path(model_path)
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)

    # Stage 3: 로그 파일
    log_path = LOGS_DIR / f"{experiment_id}.log"
    log_path.unlink(missing_ok=True)
