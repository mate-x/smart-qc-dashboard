from __future__ import annotations

import numpy as np
import pytest

from utils.metrics import compute_threshold


class TestComputeThreshold:
    def test_percentile_method(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        result = compute_threshold(scores, method="percentile", value=90.0)
        assert result == pytest.approx(np.percentile(scores, 90.0))

    def test_absolute_method(self):
        scores = np.array([0.1, 0.5, 0.9])
        result = compute_threshold(scores, method="absolute", value=0.75)
        assert result == pytest.approx(0.75)

    def test_absolute_ignores_scores(self):
        scores = np.array([10.0, 20.0, 30.0])
        assert compute_threshold(scores, method="absolute", value=42.0) == pytest.approx(42.0)

    def test_unknown_method_raises_value_error(self):
        scores = np.array([0.5])
        with pytest.raises(ValueError, match="Unknown threshold method"):
            compute_threshold(scores, method="iqr", value=1.5)

    def test_percentile_100_returns_max(self):
        scores = np.array([0.1, 0.5, 0.9])
        result = compute_threshold(scores, method="percentile", value=100.0)
        assert result == pytest.approx(0.9)
