"""
Visualization utilities for gait analysis and anomaly detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict

from src.config import RESULTS_DIR, FEATURE_COLUMNS, ACTIVITY_MAP


def plot_training_history(history: dict, save_path: Optional[Path] = None):
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Reconstruction Loss (MSE)", fontsize=12)
    ax1.set_title("Training Progress", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Learning rate
    ax2.plot(epochs, history["lr"], "g-", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Learning Rate", fontsize=12)
    ax2.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    save_path = save_path or RESULTS_DIR / "training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_reconstruction_error_distribution(
    train_errors: np.ndarray,
    val_errors: np.ndarray,
    threshold: float,
    test_errors: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
):
    """Plot distribution of reconstruction errors with anomaly threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(train_errors, bins=100, alpha=0.5, label="Train", color="steelblue", density=True)
    ax.hist(val_errors, bins=100, alpha=0.5, label="Validation", color="coral", density=True)
    if test_errors is not None:
        ax.hist(test_errors, bins=100, alpha=0.5, label="Test", color="seagreen", density=True)

    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Reconstruction Error Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_path or RESULTS_DIR / "error_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_reconstruction_examples(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    errors: np.ndarray,
    n_examples: int = 4,
    channel_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
):
    """
    Plot original vs reconstructed signals for selected windows.
    Shows both low-error (normal) and high-error (anomalous) examples.
    """
    if channel_names is None:
        channel_names = FEATURE_COLUMNS

    # Select channels to visualize (one per sensor type)
    viz_channels = [0, 3, 9, 18, 27, 36]  # RF_acc_x, RS_acc_x, LF_acc_x, RF_gyro_x, LF_gyro_x, EMG_r
    viz_names = [channel_names[i] for i in viz_channels]

    # Sort by error: pick best and worst
    sorted_idx = np.argsort(errors)
    best_idx = sorted_idx[:n_examples // 2]
    worst_idx = sorted_idx[-(n_examples // 2):]
    selected = np.concatenate([best_idx, worst_idx])

    fig, axes = plt.subplots(
        len(selected), len(viz_channels),
        figsize=(4 * len(viz_channels), 3 * len(selected)),
        sharex=True,
    )

    for row, idx in enumerate(selected):
        error_val = errors[idx]
        label = "NORMAL" if row < n_examples // 2 else "ANOMALOUS"

        for col, (ch_idx, ch_name) in enumerate(zip(viz_channels, viz_names)):
            ax = axes[row, col] if len(selected) > 1 else axes[col]
            ax.plot(originals[idx, :, ch_idx], "b-", alpha=0.7, label="Original", linewidth=1)
            ax.plot(reconstructions[idx, :, ch_idx], "r--", alpha=0.7, label="Reconstructed", linewidth=1)

            if row == 0:
                ax.set_title(ch_name, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{label}\nMSE={error_val:.4f}", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

            ax.grid(True, alpha=0.2)

    plt.suptitle("Original vs Reconstructed Signals", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = save_path or RESULTS_DIR / "reconstruction_examples.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_gait_symmetry_features(
    features: np.ndarray,
    feature_names: List[str],
    errors: np.ndarray,
    threshold: float,
    save_path: Optional[Path] = None,
):
    """
    Scatter plots of gait symmetry features colored by anomaly score.
    """
    is_anomaly = errors >= threshold

    # Select key symmetry features to visualize
    sym_features = [n for n in feature_names if "si_" in n or "asymmetry" in n or "ratio" in n]
    n_features = min(len(sym_features), 6)
    sym_features = sym_features[:n_features]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat_name in enumerate(sym_features):
        ax = axes[i]
        feat_idx = feature_names.index(feat_name)
        feat_vals = features[:, feat_idx]

        scatter = ax.scatter(
            feat_vals[~is_anomaly], errors[~is_anomaly],
            c="steelblue", alpha=0.3, s=10, label="Normal"
        )
        ax.scatter(
            feat_vals[is_anomaly], errors[is_anomaly],
            c="crimson", alpha=0.5, s=15, label="Anomaly"
        )
        ax.axhline(threshold, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel(feat_name, fontsize=10)
        ax.set_ylabel("Recon. Error", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Gait Symmetry Features vs Anomaly Score",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = save_path or RESULTS_DIR / "symmetry_features.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_sensor_signals(
    window: np.ndarray,
    title: str = "Sensor Signals",
    channel_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
):
    """Plot all sensor channels for a single window."""
    if channel_names is None:
        channel_names = FEATURE_COLUMNS

    n_channels = window.shape[1]
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, n_channels * 0.8), sharex=True)

    for i in range(n_channels):
        axes[i].plot(window[:, i], linewidth=0.8)
        axes[i].set_ylabel(channel_names[i], fontsize=7, rotation=0, ha="right")
        axes[i].tick_params(labelsize=6)
        axes[i].grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (samples)", fontsize=10)
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")
    else:
        plt.show()
