from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def mvtec_dataset(tmp_path):
    """MVTec AD 형식 더미 데이터셋 생성 (5장 × 3 클래스)."""
    root = tmp_path / "dataset"
    (root / "train" / "good").mkdir(parents=True)
    (root / "test" / "good").mkdir(parents=True)
    (root / "test" / "scratch").mkdir(parents=True)
    (root / "ground_truth" / "scratch").mkdir(parents=True)

    rng = np.random.default_rng(seed=42)
    for i in range(5):
        arr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(root / "train" / "good" / f"{i:03d}.png")
        Image.fromarray(arr).save(root / "test" / "good" / f"{i:03d}.png")
        Image.fromarray(arr).save(root / "test" / "scratch" / f"{i:03d}.png")
        mask = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        mask.save(root / "ground_truth" / "scratch" / f"{i:03d}_mask.png")
    return root


@pytest.fixture
def minimal_model_config() -> dict:
    return {
        "model_type": "patchcore",
        "image_size": 64,
        "backbone": "resnet18",
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 9,
    }


@pytest.fixture
def minimal_preprocessing_config() -> dict:
    return {
        "method": "none",
        "image_size": 64,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
