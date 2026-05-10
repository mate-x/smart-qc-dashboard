from __future__ import annotations

import logging
from pathlib import Path

import yaml

_logger = logging.getLogger(__name__)

_SECTION_ORDER = ("experiment", "preprocessing", "model")


def _ordered_config(config: dict) -> dict:
    ordered = {k: config[k] for k in _SECTION_ORDER if k in config}
    ordered.update({k: v for k, v in config.items() if k not in ordered})
    return ordered


class ConfigLoadError(Exception):
    """YAML 파일 파싱 실패 시 발생하는 예외."""


def load_config(path: str | Path = "./configs.yaml") -> dict:
    """
    YAML 파일 로드. 파일 미존재 시 빈 dict 반환.
    YAML 파싱 실패 시 ConfigLoadError raise.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        _logger.warning(
            "ERR_CONFIG_LOAD_FAILED: YAML 파싱 실패. path=%s error=%s",
            p,
            e,
        )
        raise ConfigLoadError(f"ERR_CONFIG_LOAD_FAILED: {p}: {e}") from e


def save_config_section(
    section: str,
    data: dict,
    path: str | Path = "./configs.yaml",
) -> None:
    """
    기존 파일의 다른 섹션을 보존하면서 지정 섹션만 업데이트.
    원자적 쓰기 적용 (.tmp → rename, R-ATOMIC-01 — PRD §4.3).

    Args:
        section: "experiment" | "preprocessing" | "model"
        data:    해당 섹션 dict
        path:    대상 파일 경로. 실험 스냅샷 저장 시
                 "./models/{exp_id}/configs.yaml" 전달.

    Raises:
        RuntimeError: 쓰기 실패 시 (tmp 파일 정리 후 raise)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(p)
    config[section] = data

    tmp = p.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(_ordered_config(config), f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        tmp.replace(p)
    except IOError as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ERR_CONFIG_WRITE_FAILED: {e}") from e


def save_experiment_snapshot(
    exp_id: str,
    name: str,
    created_at: str,
) -> Path:
    """
    루트 configs.yaml 전체를 읽어 실험 스냅샷으로 저장 (PRD §5.2).
    experiment.name, experiment.created_at을 추가한 뒤
    ./models/{exp_id}/configs.yaml 에 원자적으로 저장.

    이 파일은 저장 후 변경 불가 (불변 원칙 — PRD §5.2).
    탭6 모델 재로드 시 이 파일을 기준으로 파라미터 재현.

    Returns:
        저장된 스냅샷 Path.

    Raises:
        RuntimeError: 쓰기 실패 시
    """
    snapshot_path = Path(f"./models/{exp_id}/configs.yaml")
    config = load_config("./configs.yaml")
    config.setdefault("experiment", {})
    config["experiment"]["name"] = name
    config["experiment"]["created_at"] = created_at

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = snapshot_path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(_ordered_config(config), f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        tmp.replace(snapshot_path)
    except IOError as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ERR_CONFIG_WRITE_FAILED (스냅샷): {e}") from e

    return snapshot_path


def get_preprocessing_config(path: str | Path = "./configs.yaml") -> dict | None:
    return load_config(path).get("preprocessing")


def get_model_config(path: str | Path = "./configs.yaml") -> dict | None:
    return load_config(path).get("model")
