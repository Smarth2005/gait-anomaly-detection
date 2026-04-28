"""
Evaluation metrics for gait anomaly detection.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, confusion_matrix,
)


def compute_anomaly_threshold(val_scores: np.ndarray, percentile: float = 95.0) -> float:
    """
    Compute anomaly threshold from validation set reconstruction errors.

    Args:
        val_scores: Array of reconstruction errors on validation set
        percentile: Percentile to use as threshold (e.g., 95 = top 5% are anomalous)

    Returns:
        threshold: scalar
    """
    return float(np.percentile(val_scores, percentile))


def compute_anomaly_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = None,
) -> Dict[str, float]:
    """
    Compute anomaly detection metrics.

    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: Binary labels (1 = anomaly, 0 = normal)
        threshold: If provided, compute binary metrics at this threshold

    Returns:
        Dictionary of metric name → value
    """
    metrics = {}

    # ROC-AUC
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["avg_precision"] = float(average_precision_score(labels, scores))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["avg_precision"] = float("nan")

    # Threshold-based binary metrics
    if threshold is not None:
        predictions = (scores >= threshold).astype(int)
        metrics["threshold"] = threshold

        if len(np.unique(predictions)) > 1 and len(np.unique(labels)) > 1:
            metrics["f1"] = float(f1_score(labels, predictions))
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            metrics["precision"] = tp / max(tp + fp, 1)
            metrics["recall"] = tp / max(tp + fn, 1)
            metrics["specificity"] = tn / max(tn + fp, 1)
            metrics["fpr"] = fp / max(fp + tn, 1)  # False positive rate
        else:
            metrics["f1"] = 0.0
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0

    # Score distribution stats
    metrics["score_mean"] = float(np.mean(scores))
    metrics["score_std"] = float(np.std(scores))
    metrics["score_median"] = float(np.median(scores))
    metrics["score_95pct"] = float(np.percentile(scores, 95))
    metrics["score_99pct"] = float(np.percentile(scores, 99))

    return metrics


def reconstruction_error_stats(errors: np.ndarray) -> Dict[str, float]:
    """
    Summarize reconstruction error distribution.

    Args:
        errors: Array of per-window reconstruction errors

    Returns:
        Dictionary of statistics
    """
    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "median": float(np.median(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
    }
