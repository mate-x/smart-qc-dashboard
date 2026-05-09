from __future__ import annotations

import numpy as np
import pytest

from utils.metrics import normalize_anomaly_scores


class TestNormalizeAnomalyScores:
    def test_minmax_output_range(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = normalize_anomaly_scores(scores, method="minmax")
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_all_same_scores_returns_zeros(self):
        scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = normalize_anomaly_scores(scores, method="minmax")
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))

    def test_output_dtype_float32(self):
        scores = np.array([0.0, 1.0, 2.0])
        result = normalize_anomaly_scores(scores)
        assert result.dtype == np.float32

    def test_single_element(self):
        scores = np.array([3.7])
        result = normalize_anomaly_scores(scores, method="minmax")
        assert result[0] == pytest.approx(0.0)

    def test_unknown_method_raises_value_error(self):
        scores = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_anomaly_scores(scores, method="zscore")
