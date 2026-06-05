from __future__ import annotations

from pydantic import BaseModel, field_validator


class ApplyModelRequest(BaseModel):
    experiment_id: str
    source_path: str | None = None

    @field_validator("experiment_id")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("experiment_id는 빈 문자열일 수 없습니다.")
        return v


class ApplyModelResponse(BaseModel):
    success: bool
    active_model: dict | None
    gpu_warning: str | None


class ActiveModelResponse(BaseModel):
    active_model: dict | None


class ClearRecordsResponse(BaseModel):
    success: bool


class InspectionJobStartedResponse(BaseModel):
    job_id: str


class InspectionJobStatusResponse(BaseModel):
    status: str           # "pending" | "running" | "completed" | "failed"
    result: dict | None = None
    error: str | None = None
