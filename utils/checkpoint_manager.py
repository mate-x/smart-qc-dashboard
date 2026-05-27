from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

CHECKPOINT_DIR = Path("./models/checkpoints")
KST = timezone(timedelta(hours=9))


def save_checkpoint(data: dict, exp_id: str, label: int) -> Path:
    """
    체크포인트를 .ckpt 파일로 저장.

    파일명: {exp_id}_step{label}.ckpt
      - EfficientAD: label = 현재 step
      - PatchCore:   label = 현재 batch_idx

    data 공통 키:
      model_type, experiment_id, model_config, preprocessing_config, dataset_path
    EfficientAD 추가 키:
      step, total_steps, loss_history,
      student_state_dict, autoencoder_state_dict,
      optimizer_st_state_dict, optimizer_ae_state_dict,
      scheduler_st_state_dict, scheduler_ae_state_dict
    PatchCore 추가 키:
      batch_idx, total_batches, accumulated_features (Tensor)
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{exp_id}_step{label}.ckpt"
    path = CHECKPOINT_DIR / fname
    data.setdefault("created_at", datetime.now(tz=KST).isoformat())
    torch.save(data, path)
    return path


def load_checkpoint(path: str | Path) -> dict:
    """체크포인트 파일 로드 후 dict 반환."""
    return torch.load(str(path), map_location="cpu", weights_only=False)


def list_checkpoints() -> list[Path]:
    """CHECKPOINT_DIR 의 .ckpt 파일을 수정시간 역순으로 반환."""
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(
        CHECKPOINT_DIR.glob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def delete_checkpoint(path: str | Path) -> bool:
    """체크포인트 파일 삭제. 성공 시 True 반환."""
    p = Path(path)
    if p.exists():
        try:
            p.unlink()
            return True
        except OSError:
            return False
    return False
