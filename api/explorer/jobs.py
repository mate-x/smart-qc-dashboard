"""
api/explorer/jobs.py

비동기 Job 저장소.
- build job: Anomaly Map 생성 (결과 없음, 완료 여부만 확인)
- zip  job: ZIP 파일 생성 (완료 시 result에 bytes 저장)
"""
from __future__ import annotations

import uuid

_jobs: dict[str, dict] = {}


def create_job(job_type: str) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "type":   job_type,
        "status": "pending",
        "result": None,
        "error":  None,
    }
    return job_id


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def set_running(job_id: str) -> None:
    if job_id in _jobs:
        _jobs[job_id]["status"] = "running"


def set_completed(job_id: str, result=None) -> None:
    if job_id in _jobs:
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = result


def set_failed(job_id: str, error: str) -> None:
    if job_id in _jobs:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"]  = error
