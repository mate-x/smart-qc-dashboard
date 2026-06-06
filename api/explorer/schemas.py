from __future__ import annotations

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# 탭4 · 실험 히스토리
# ---------------------------------------------------------------------------

class SaveModelRequest(BaseModel):
    save_path: str


class SaveModelResponse(BaseModel):
    success: bool
    saved_path: str
    size_mb: float
    warning: str | None = None

class DeleteExperimentResponse(BaseModel):
    success: bool
