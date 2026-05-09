from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_normalize_scores_reproducibility():
    """R-SEED-01: 동일 입력에 대해 정규화 결과가 항상 동일해야 한다."""
    from utils.metrics import normalize_anomaly_scores

    scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2], dtype=np.float32)
    result1 = normalize_anomaly_scores(scores.copy(), method="minmax")
    result2 = normalize_anomaly_scores(scores.copy(), method="minmax")
    np.testing.assert_array_equal(result1, result2)


@pytest.mark.slow
def test_compute_metrics_reproducibility():
    """동일 입력에 대해 metrics 계산이 항상 동일해야 한다."""
    from utils.metrics import compute_all_metrics

    y_true = [0, 0, 1, 1, 0, 1]
    scores = [0.1, 0.2, 0.8, 0.9, 0.3, 0.7]
    threshold = 0.5

    result1 = compute_all_metrics(y_true, scores, threshold)
    result2 = compute_all_metrics(y_true, scores, threshold)
    assert result1 == result2


@pytest.mark.slow
def test_config_save_load_reproducibility(tmp_path):
    """설정 저장 후 반복 로드 시 결과가 항상 동일해야 한다."""
    from utils.config_manager import save_config_section, get_preprocessing_config

    config_path = str(tmp_path / "configs.yaml")
    data = {"method": "clahe", "image_size": 256, "params": {"clip_limit": 2.0}}
    save_config_section("preprocessing", data, config_path)

    results = [get_preprocessing_config(config_path) for _ in range(5)]
    for r in results:
        assert r == data
