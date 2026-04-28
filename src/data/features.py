"""
Gait symmetry and temporal feature extraction from accelerometer + gyroscope + EMG data.

These hand-crafted features complement the learned autoencoder features
for a richer anomaly scoring pipeline.
"""

import numpy as np
from typing import Dict
from scipy import signal as scipy_signal


# Sensor column index ranges within FEATURE_COLUMNS (0-indexed)
# FEATURE_COLUMNS = ACC (18) + GYRO (18) + EMG (2) = 38 total
#
# Accelerometer layout (indices 0-17):
#   RF: 0-2, RS: 3-5, RT: 6-8, LF: 9-11, LS: 12-14, LT: 15-17
#
# Gyroscope layout (indices 18-35):
#   RF: 18-20, RS: 21-23, RT: 24-26, LF: 27-29, LS: 30-32, LT: 33-35
#
# EMG layout (indices 36-37):
#   R: 36, L: 37

# Right side accelerometer indices
ACC_RIGHT = {
    "RF": slice(0, 3),   # Right Foot
    "RS": slice(3, 6),   # Right Shank
    "RT": slice(6, 9),   # Right Thigh
}

# Left side accelerometer indices
ACC_LEFT = {
    "LF": slice(9, 12),  # Left Foot
    "LS": slice(12, 15), # Left Shank
    "LT": slice(15, 18), # Left Thigh
}

# Right side gyroscope indices
GYRO_RIGHT = {
    "RF": slice(18, 21),
    "RS": slice(21, 24),
    "RT": slice(24, 27),
}

# Left side gyroscope indices
GYRO_LEFT = {
    "LF": slice(27, 30),
    "LS": slice(30, 33),
    "LT": slice(33, 36),
}

# EMG indices
EMG_R_IDX = 36
EMG_L_IDX = 37


def _magnitude(data_3axis: np.ndarray) -> np.ndarray:
    """Compute vector magnitude from 3-axis data. Shape: (n,3) → (n,)"""
    return np.sqrt(np.sum(data_3axis ** 2, axis=-1))


def compute_symmetry_index(right_mag: np.ndarray, left_mag: np.ndarray) -> float:
    """
    Symmetry Index: measures imbalance between left and right sides.
    SI = mean(|R - L| / (0.5 * (R + L))) * 100

    Perfect symmetry → SI = 0. Higher SI → more asymmetric (pathological).
    """
    denom = 0.5 * (right_mag + left_mag)
    # Avoid division by zero for static segments
    valid = denom > 1e-6
    if valid.sum() == 0:
        return 0.0
    si = np.abs(right_mag[valid] - left_mag[valid]) / denom[valid]
    return float(np.mean(si) * 100)


def compute_rms(data: np.ndarray) -> float:
    """Root Mean Square of a signal."""
    return float(np.sqrt(np.mean(data ** 2)))


def compute_jerk(data: np.ndarray, sampling_rate: float = 60.0) -> float:
    """
    Jerk metric: rate of change of acceleration.
    High jerk → jerky/unsmooth movement → indicator of pathology.
    """
    dt = 1.0 / sampling_rate
    jerk = np.diff(data, axis=0) / dt
    return float(np.mean(np.sqrt(np.sum(jerk ** 2, axis=-1))))


def compute_step_regularity(vertical_acc: np.ndarray) -> float:
    """
    Step regularity via autocorrelation.
    High regularity → consistent gait rhythm. Low → irregular gait.

    Uses the first dominant autocorrelation peak (after lag 0) as the
    regularity metric.
    """
    if len(vertical_acc) < 30:
        return 0.0

    # Normalize
    x = vertical_acc - np.mean(vertical_acc)
    norm = np.sum(x ** 2)
    if norm < 1e-8:
        return 0.0

    # Full autocorrelation
    acf = np.correlate(x, x, mode="full")
    acf = acf[len(acf) // 2:]  # Keep positive lags only
    acf = acf / acf[0]  # Normalize so lag-0 = 1.0

    # Find first peak after lag 0 (skip first 15 samples ≈ 0.25 sec)
    min_lag = 15
    if len(acf) <= min_lag:
        return 0.0

    peaks, _ = scipy_signal.find_peaks(acf[min_lag:], height=0)
    if len(peaks) == 0:
        return 0.0

    # Return height of first peak as regularity score
    first_peak_idx = peaks[0] + min_lag
    return float(acf[first_peak_idx])


def compute_harmonic_ratio(vertical_acc: np.ndarray, sampling_rate: float = 60.0) -> float:
    """
    Harmonic Ratio: ratio of even to odd harmonics in vertical acceleration.
    Higher ratio → smoother, more symmetric gait.
    Lower ratio → asymmetric, potentially pathological.
    """
    n = len(vertical_acc)
    if n < 30:
        return 0.0

    # FFT
    fft_vals = np.abs(np.fft.rfft(vertical_acc))

    # Use first 20 harmonics (skip DC component at index 0)
    n_harmonics = min(20, len(fft_vals) - 1)
    if n_harmonics < 4:
        return 0.0

    harmonics = fft_vals[1:n_harmonics + 1]

    # Even harmonics (indices 1, 3, 5, ... in the harmonics array → 2nd, 4th, 6th actual)
    even = harmonics[1::2]
    # Odd harmonics (indices 0, 2, 4, ... → 1st, 3rd, 5th actual)
    odd = harmonics[0::2]

    odd_sum = np.sum(odd)
    if odd_sum < 1e-8:
        return 0.0

    return float(np.sum(even) / odd_sum)


def extract_gait_features(window: np.ndarray, sampling_rate: float = 60.0) -> Dict[str, float]:
    """
    Extract gait symmetry and temporal features from a single window.

    Args:
        window: Array of shape (window_size, 38) — normalized sensor data
        sampling_rate: Sampling frequency in Hz

    Returns:
        Dictionary of feature name → value
    """
    features = {}

    # --- Accelerometer symmetry features ---
    # Per body location: compute magnitude then symmetry index
    for (r_name, r_slice), (l_name, l_slice) in zip(
        ACC_RIGHT.items(), ACC_LEFT.items()
    ):
        r_mag = _magnitude(window[:, r_slice])
        l_mag = _magnitude(window[:, l_slice])
        location = r_name[1]  # F, S, or T
        features[f"acc_si_{location}"] = compute_symmetry_index(r_mag, l_mag)
        features[f"acc_ratio_{location}"] = (
            float(np.mean(r_mag) / max(np.mean(l_mag), 1e-8))
        )

    # --- Gyroscope symmetry features ---
    for (r_name, r_slice), (l_name, l_slice) in zip(
        GYRO_RIGHT.items(), GYRO_LEFT.items()
    ):
        r_mag = _magnitude(window[:, r_slice])
        l_mag = _magnitude(window[:, l_slice])
        location = r_name[1]
        features[f"gyro_si_{location}"] = compute_symmetry_index(r_mag, l_mag)
        features[f"gyro_ratio_{location}"] = (
            float(np.mean(r_mag) / max(np.mean(l_mag), 1e-8))
        )

    # --- Overall body symmetry (mean across all locations) ---
    r_acc_total = np.concatenate([
        _magnitude(window[:, s]) for s in ACC_RIGHT.values()
    ])
    l_acc_total = np.concatenate([
        _magnitude(window[:, s]) for s in ACC_LEFT.values()
    ])
    features["acc_si_overall"] = compute_symmetry_index(r_acc_total, l_acc_total)

    # --- EMG asymmetry ---
    emg_r = window[:, EMG_R_IDX]
    emg_l = window[:, EMG_L_IDX]
    emg_max = np.maximum(np.abs(emg_r), np.abs(emg_l))
    valid = emg_max > 1e-6
    if valid.sum() > 0:
        features["emg_asymmetry"] = float(
            np.mean(np.abs(emg_r[valid] - emg_l[valid]) / emg_max[valid])
        )
    else:
        features["emg_asymmetry"] = 0.0

    # --- Temporal features (from right foot vertical accelerometer) ---
    # Use y-axis of right foot as vertical acceleration proxy
    rf_acc_y = window[:, 1]  # acc_rf_y
    lf_acc_y = window[:, 10]  # acc_lf_y

    features["step_regularity_R"] = compute_step_regularity(rf_acc_y)
    features["step_regularity_L"] = compute_step_regularity(lf_acc_y)

    # --- Jerk (movement smoothness) ---
    features["jerk_R_foot"] = compute_jerk(window[:, ACC_RIGHT["RF"]], sampling_rate)
    features["jerk_L_foot"] = compute_jerk(window[:, ACC_LEFT["LF"]], sampling_rate)
    features["jerk_asymmetry"] = abs(
        features["jerk_R_foot"] - features["jerk_L_foot"]
    ) / max(features["jerk_R_foot"] + features["jerk_L_foot"], 1e-8)

    # --- Harmonic ratio ---
    features["harmonic_ratio_R"] = compute_harmonic_ratio(rf_acc_y, sampling_rate)
    features["harmonic_ratio_L"] = compute_harmonic_ratio(lf_acc_y, sampling_rate)

    # --- RMS per limb ---
    for name, s in {**ACC_RIGHT, **ACC_LEFT}.items():
        features[f"rms_{name}"] = compute_rms(window[:, s])

    return features


def extract_features_batch(windows: np.ndarray) -> np.ndarray:
    """
    Extract gait features for a batch of windows.

    Args:
        windows: Array of shape (n_windows, window_size, n_features)

    Returns:
        Feature matrix of shape (n_windows, n_gait_features)
    """
    all_features = []
    for i in range(len(windows)):
        feat_dict = extract_gait_features(windows[i])
        all_features.append(list(feat_dict.values()))

    return np.array(all_features, dtype=np.float32)


def get_feature_names() -> list:
    """Get ordered list of gait feature names."""
    # Extract from a dummy window to get consistent ordering
    dummy = np.zeros((128, 38), dtype=np.float32)
    feat_dict = extract_gait_features(dummy)
    return list(feat_dict.keys())
