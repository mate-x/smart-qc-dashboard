from __future__ import annotations

import numpy as np


def resolve_threshold(experiment: dict) -> float:
    """threshold_method/threshold_value 설정에 따라 raw threshold 값을 반환."""
    method = experiment.get("threshold_method", "absolute")
    value  = float(experiment.get("threshold_value", 0.5))

    if method == "absolute":
        return value

    metrics = experiment.get("metrics") or {}
    scores  = metrics.get("anomaly_scores", [])
    labels  = metrics.get("image_labels", [])

    if scores and labels and len(scores) == len(labels):
        normal_scores = [s for s, l in zip(scores, labels) if l == 0]
        if normal_scores:
            return float(np.percentile(normal_scores, value))

    return value
