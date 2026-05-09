from __future__ import annotations

import pytest


@pytest.mark.slow
def test_golden_path_dataset_scan_to_config_save(mvtec_dataset, tmp_path):
    """E2E: 데이터셋 스캔 → 설정 저장 → 다시 로드 golden path."""
    from utils.dataset_scanner import scan_dataset
    from utils.config_manager import save_config_section, get_preprocessing_config

    meta = scan_dataset(str(mvtec_dataset))
    assert meta["train_good_count"] > 0

    config_path = str(tmp_path / "configs.yaml")
    preprocessing = {
        "method": "none",
        "image_size": meta["channels"] and 64 or 64,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    save_config_section("preprocessing", preprocessing, config_path)

    loaded = get_preprocessing_config(config_path)
    assert loaded == preprocessing


@pytest.mark.slow
def test_golden_path_image_preprocessing_pipeline(mvtec_dataset):
    """E2E: 이미지 로드 → 전처리 → tensor 변환 전체 파이프라인."""
    from utils.image_utils import apply_preprocessing

    test_image = next((mvtec_dataset / "test" / "scratch").glob("*.png"))
    config = {
        "method": "none",
        "image_size": 64,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    preview, tensor = apply_preprocessing(str(test_image), config)
    assert tensor.shape == (3, 64, 64)
    assert preview.size == (64, 64)
