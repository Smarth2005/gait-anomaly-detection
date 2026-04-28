"""
Evaluation metrics for gait anomaly detection.
"""

import numpy as np
from typing import Dict, Tuple



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
