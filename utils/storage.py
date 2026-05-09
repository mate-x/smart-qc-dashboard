from __future__ import annotations

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
