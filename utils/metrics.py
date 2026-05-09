from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def compute_all_metrics(
    y_true: list[int],
    anomaly_scores: list[float],
    threshold: float,
) -> dict:
    y_pred = [1 if s >= threshold else 0 for s in anomaly_scores]

    accuracy = round(float(accuracy_score(y_true, y_pred)), 6)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    f2 = round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 6)
    auc = round(float(roc_auc_score(y_true, anomaly_scores)), 6)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1_score": round(float(f1), 6),
        "f2_score": f2,
        "auc": auc,
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
        "anomaly_scores": [round(float(s), 6) for s in anomaly_scores],
        "image_labels": [int(v) for v in y_true],
    }


# 하위 호환 alias
compute_metrics = compute_all_metrics


def compute_roc_curve(
    y_true: list[int],
    anomaly_scores: list[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
    auc = float(roc_auc_score(y_true, anomaly_scores))
    return fpr, tpr, auc


def compute_threshold_from_percentile(
    normal_scores: list[float],
    percentile: float,
) -> float:
    return float(np.percentile(normal_scores, percentile))


def normalize_anomaly_scores(
    scores: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """Anomaly Score 배열 정규화. 모두 같은 값이면 0으로 처리 (division by zero 방지)."""
    if method == "minmax":
        min_val = scores.min()
        max_val = scores.max()
        if max_val == min_val:
            return np.zeros_like(scores, dtype=np.float32)
        return ((scores - min_val) / (max_val - min_val)).astype(np.float32)
    raise ValueError(f"Unknown normalization method: {method}")


def compute_threshold(
    normal_scores: np.ndarray,
    method: str,
    value: float,
) -> float:
    """threshold_method에 따라 임계값 계산."""
    if method == "percentile":
        return float(np.percentile(normal_scores, value))
    if method == "absolute":
        return float(value)
    raise ValueError(f"Unknown threshold method: {method}")
