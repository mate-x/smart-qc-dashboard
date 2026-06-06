from __future__ import annotations

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# 탭1 · 데이터셋
# ---------------------------------------------------------------------------

class ValidateDatasetRequest(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def path_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("경로를 입력하세요.")
        return v.strip()


class ValidateDatasetResponse(BaseModel):
    dataset_format: str               # "mvtec" | "oking"
    channels: int
    train_good_count: int
    test_counts: dict[str, int]
    gt_counts: dict[str, int]
    total_test_count: int
    defect_classes: list[str]
    supported_formats: list[str]
    has_invalid_files: bool
    invalid_file_count: int
    folder_tree: str
    # OK/NG 전용
    oking_ok_dir: str | None = None
    oking_ng_dir: str | None = None
    oking_ok_count: int | None = None
    oking_ng_count: int | None = None
    train_ratio: float | None = None


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
