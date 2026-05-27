"""
utils/logger.py — 시스템 공통 로그 모듈

00_Global §7 Observability Standards / 12_Observability §B.1·B.2 구현체.

콘솔: JSON 1줄 출력 (Python logging 경유).
파일: ExperimentLogWriter — ./logs/{exp_id}.log 라인 단위 기록.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Literal

KST = timezone(timedelta(hours=9))
LogLevel = Literal["INFO", "WARNING", "ERROR"]

_py_logger = logging.getLogger("smart_qc")
_py_logger.setLevel(logging.DEBUG)
if not _py_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _py_logger.addHandler(_handler)

_LEVEL_MAP = {
    "INFO":    logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR":   logging.ERROR,
}


def _now_kst() -> str:
    return datetime.now(tz=KST).isoformat(timespec="milliseconds")


def _build_entry(
    level: LogLevel,
    event: str,
    message: str,
    experiment_id: str | None = None,
    tab: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict:
    return {
        "timestamp":     _now_kst(),
        "level":         level,
        "experiment_id": experiment_id,
        "tab":           tab,
        "event":         event,
        "message":       message,
        "data":          data or {},
    }


def _write(entry: dict) -> None:
    _py_logger.log(
        _LEVEL_MAP[entry["level"]],
        json.dumps(entry, ensure_ascii=False),
    )


def log_info(
    event: str,
    message: str,
    *,
    experiment_id: str | None = None,
    tab: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    _write(_build_entry("INFO", event, message, experiment_id=experiment_id, tab=tab, data=data))


def log_warning(
    event: str,
    message: str,
    *,
    experiment_id: str | None = None,
    tab: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    _write(_build_entry("WARNING", event, message, experiment_id=experiment_id, tab=tab, data=data))


def log_error(
    event: str,
    message: str,
    *,
    experiment_id: str | None = None,
    tab: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    _write(_build_entry("ERROR", event, message, experiment_id=experiment_id, tab=tab, data=data))


# ── 실험별 파일 로거 ──────────────────────────────────────────────────────────


class ExperimentLogWriter:
    """
    ./logs/{experiment_id}.log 에 라인 단위로 기록 (라인 버퍼).
    TrainingWorker.__init__() 에서 1회 생성; run() finally 에서 close().
    """

    def __init__(self, experiment_id: str) -> None:
        log_dir = Path("./logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / f"{experiment_id}.log"
        self._f = open(self._path, "a", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        self._f.write(f"{_now_kst()}\t{message}\n")

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __del__(self) -> None:
        self.close()


def get_log_writer(experiment_id: str) -> ExperimentLogWriter:
    """TrainingWorker 전용. 반환된 writer는 호출자가 close() 책임."""
    return ExperimentLogWriter(experiment_id)
