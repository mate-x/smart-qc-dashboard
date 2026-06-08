from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# 탭1 · 데이터셋
# ---------------------------------------------------------------------------

class ValidateDatasetRequest(BaseModel):
    path: str
    product_name: str = ""

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
    has_background_clean: bool        # {dataset_path}/background_clean/ 폴더 존재 여부
    # OK/NG 전용
    oking_ok_dir: str | None = None
    oking_ng_dir: str | None = None
    oking_ok_count: int | None = None
    oking_ng_count: int | None = None
    train_ratio: float | None = None


# ---------------------------------------------------------------------------
# 탭2 · 설정
# ---------------------------------------------------------------------------

class SaveConfigRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    preprocessing_config: dict
    model_cfg: dict = Field(alias="model_config")


class GetConfigResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    preprocessing_config: dict | None = None
    model_cfg: dict | None = Field(default=None, alias="model_config")
    device_info: dict


class PreviewThresholdRequest(BaseModel):
    threshold_method: str
    threshold_value: float


class PreviewThresholdResponse(BaseModel):
    normal_ratio: float | None = None
    defect_ratio: float | None = None


class LoadYamlResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    preprocessing_config: dict | None = None
    model_cfg: dict | None = Field(default=None, alias="model_config")


# ---------------------------------------------------------------------------
# 탭2 · 큐
# ---------------------------------------------------------------------------

class AddQueueRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    preprocessing_config: dict
    model_cfg: dict = Field(alias="model_config")
    set_id: str | None = None


class QueueItemResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    name: str
    preprocessing_config: dict
    model_cfg: dict = Field(alias="model_config")
    status: str
    set_id: str | None = None


class AddQueueResponse(BaseModel):
    id: str
    name: str


class DeleteQueueResponse(BaseModel):
    success: bool


class ReorderQueueRequest(BaseModel):
    item_id: str
    direction: str  # "up" | "down"


class ReorderQueueResponse(BaseModel):
    success: bool


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


# ---------------------------------------------------------------------------
# 탭5 · Anomaly Map
# ---------------------------------------------------------------------------

class BuildAnomalyMapResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    status: str               # "pending" | "running" | "completed" | "failed"
    error: str | None = None


class ImageRowResponse(BaseModel):
    image_name: str
    defect_class: str
    anomaly_score: float
    verdict: str              # "OK" | "NG"
    gt_match: bool
    classification: str       # "TP" | "FP" | "TN" | "FN"
    image_path: str           # "{class}/{filename}" — triplet 엔드포인트 키


class AnomalyImagesResponse(BaseModel):
    images: list[ImageRowResponse]
    score_max: float
    score_avg: float
    tp: int
    fp: int
    tn: int
    fn: int


class ZipRequest(BaseModel):
    threshold: float
    defect_class: str = "전체"


class ZipJobResponse(BaseModel):
    job_id: str


class AnomalyMapStatusResponse(BaseModel):
    built: bool
    image_count: int


# ---------------------------------------------------------------------------
# 탭3 · Training
# ---------------------------------------------------------------------------

class StartTrainingRequest(BaseModel):
    experiment_name: str = ""


class ResumeTrainingRequest(BaseModel):
    checkpoint_name: str


class StartTrainingResponse(BaseModel):
    exp_id: str


class TrainingControlResponse(BaseModel):
    success: bool
    message: str = ""


class CheckpointItem(BaseModel):
    name: str
    model_type: str = ""
    created_at: str = ""
    # EfficientAD
    step: int | None = None
    total_steps: int | None = None
    # PatchCore
    batch_idx: int | None = None
    total_batches: int | None = None
    n_patches: int | None = None


class CheckpointsResponse(BaseModel):
    checkpoints: list[CheckpointItem]


class DeleteCheckpointResponse(BaseModel):
    success: bool


class TrainingStatusResponse(BaseModel):
    status: str                           # "idle" | "running" | "paused"
    exp_id: str | None = None
    batch_mode: bool = False
    batch_total: int = 0
    progress: dict | None = None          # {step, total, loss, elapsed}
    current_stage_idx: int | None = None
    current_stage_name: str | None = None
    log_lines: list[str] = []
    loss_history: list[dict] = []
    last_ckpt_path: str | None = None


class BatchStartResponse(BaseModel):
    exp_id: str
    batch_total: int
