from __future__ import annotations

import numpy as np
import pytest

from utils.metrics import compute_all_metrics, compute_metrics


class TestComputeAllMetrics:
    def test_returns_required_keys(self):
        y_true = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        result = compute_all_metrics(y_true, scores, threshold=0.5)
        required = {"accuracy", "precision", "recall", "f1_score", "f2_score", "auc",
                    "confusion_matrix", "anomaly_scores", "image_labels"}
        assert required.issubset(result.keys())

    def test_values_rounded_to_6_decimals(self):
        y_true = [0, 1, 0, 1]
        scores = [0.1, 0.9, 0.2, 0.8]
        result = compute_all_metrics(y_true, scores, threshold=0.5)
        for key in ("accuracy", "precision", "recall", "f1_score", "f2_score", "auc"):
            val = result[key]
            assert val == round(val, 6), f"{key} not rounded to 6 decimals"

    def test_confusion_matrix_keys(self):
        y_true = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        cm = compute_all_metrics(y_true, scores, threshold=0.5)["confusion_matrix"]
        assert set(cm.keys()) == {"tp", "fp", "tn", "fn"}

    def test_perfect_prediction(self):
        y_true = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.9, 0.8]
        result = compute_all_metrics(y_true, scores, threshold=0.5)
        assert result["accuracy"] == 1.0
        assert result["auc"] == 1.0
        assert result["confusion_matrix"]["tp"] == 2
        assert result["confusion_matrix"]["tn"] == 2
        assert result["confusion_matrix"]["fp"] == 0
        assert result["confusion_matrix"]["fn"] == 0

    def test_auc_in_valid_range(self):
        y_true = [0, 1, 0, 1, 0, 1]
        scores = [0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
        result = compute_all_metrics(y_true, scores, threshold=0.5)
        assert 0.0 <= result["auc"] <= 1.0

    def test_anomaly_scores_rounded(self):
        y_true = [0, 1]
        scores = [0.123456789, 0.987654321]
        result = compute_all_metrics(y_true, scores, threshold=0.5)
        for s in result["anomaly_scores"]:
            assert s == round(s, 6)

    def test_image_labels_are_integers(self):
        y_true = [0, 1, 0]
        scores = [0.1, 0.9, 0.2]
        result = compute_all_metrics(y_true, scores, threshold=0.3)
        for v in result["image_labels"]:
            assert isinstance(v, int)


def test_compute_metrics_alias():
    assert compute_metrics is compute_all_metrics
