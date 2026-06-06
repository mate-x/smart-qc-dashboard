"""
api/vision/jobs.py

검사 Job 관리 — job_id 생성, 상태 추적, 비동기 추론 실행.
routes 레이어와 분리하여 WebSocket 등 다른 진입점에서도 재사용 가능.
"""
from __future__ import annotations

import asyncio
import uuid

from api.vision.services.inspection_service import run_single_inspection

# job_id → {"status": ..., "result": ..., "error": ...}
_jobs: dict[str, dict] = {}


def create_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending"}
    return job_id


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def pop_job(job_id: str) -> dict | None:
    return _jobs.pop(job_id, None)


async def run_inspection_job(job_id: str) -> None:
    _jobs[job_id] = {"status": "running"}
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, run_single_inspection)
        _jobs[job_id] = {"status": "completed", "result": result}
    except Exception as e:
        _jobs[job_id] = {"status": "failed", "error": str(e)}
