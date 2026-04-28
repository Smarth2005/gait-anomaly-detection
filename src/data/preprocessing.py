"""
Data preprocessing pipeline for HuGaDB.

Handles:
1. Loading data from SQLite database
2. Gyroscope correction (÷10 for corrupted channels)
3. Clipped value interpolation (values at ±32767 after 10x amplification)
4. Z-score normalization (fit on training set)
5. Sliding window segmentation
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List

from src.config import (
    DB_PATH, FEATURE_COLUMNS, LABEL_COLUMN, GYRO_COLUMNS, GYRO_GROUPS,
    GYRO_AMPLIFICATION_FACTOR, CLIP_THRESHOLD, WINDOW_SIZE, WINDOW_STRIDE,
    TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS, WALKING_ACTIVITY_ID,
)
from src.data.corruption_map import get_corrupted_gyro_columns


def extract_metadata_from_filename(filename: str) -> dict:
    """
    Parse HuGaDB filename to extract participant ID, activity, and session.

    Filename format: HuGaDB_v1_{activity}_{participant}_{counter}.txt
    Example: HuGaDB_v1_walking_04_02.txt → activity='walking', participant=4, counter=2
    """
    import os
    basename = os.path.basename(filename).replace(".txt", "")
    parts = basename.split("_")

    # Handle multi-word activities like 'sitting_in_car', 'going_upstairs'
    # Format: HuGaDB_v1_{activity_words}_{2-digit participant}_{2-digit counter}
    # The last two parts are always participant and counter (both 2-digit numbers)
    participant = int(parts[-2])
    counter = int(parts[-1])
    activity = "_".join(parts[2:-2])  # Everything between 'v1' and participant

    return {
        "filename": os.path.basename(filename),
        "participant": participant,
        "activity": activity,
        "counter": counter,
    }


def load_from_database(db_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all data from the HuGaDB SQLite database.

    Returns:
        data_df: DataFrame with all sensor readings + file_id + timestamp + act
        files_df: DataFrame mapping file_id to filename
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(str(db_path))

    files_df = pd.read_sql_query("SELECT * FROM files", conn)
    data_df = pd.read_sql_query("SELECT * FROM data", conn)

    conn.close()

    print(f"Loaded {len(data_df):,} rows from {len(files_df)} files")
    return data_df, files_df


def correct_gyroscope(data_df: pd.DataFrame, files_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply gyroscope correction to all data:
    1. For each file, check the corruption map for which gyro channels are corrupted
    2. Divide those channels by 10 (the amplification factor)
    3. Detect and interpolate clipped values (|value| >= CLIP_THRESHOLD)

    Args:
        data_df: Full data DataFrame with file_id column
        files_df: File ID to filename mapping

    Returns:
        Corrected DataFrame (modified in-place for memory efficiency)
    """
    print("Correcting gyroscope data...")

    # Build file_id → filename lookup
    id_to_filename = dict(zip(files_df["id"], files_df["filename"]))

    corrected_files = 0
    total_clipped_before = 0
    total_clipped_after = 0

    # Convert all gyroscope columns to float to allow ÷10 correction
    for col in GYRO_COLUMNS:
        if col in data_df.columns:
            data_df[col] = data_df[col].astype(np.float64)

    # Process each file
    for file_id, filename in id_to_filename.items():
        corrupted_cols = get_corrupted_gyro_columns(filename)
        if not corrupted_cols:
            continue

        corrected_files += 1
        mask = data_df["file_id"] == file_id

        # Count clipped values before correction
        file_data = data_df.loc[mask, corrupted_cols]
        clipped_before = (file_data.abs() >= CLIP_THRESHOLD).sum().sum()
        total_clipped_before += clipped_before

        # Step 1: Divide corrupted channels by amplification factor
        data_df.loc[mask, corrupted_cols] = (
            file_data.values / GYRO_AMPLIFICATION_FACTOR
        )

        # Step 2: Interpolate clipped values
        # After ÷10, formerly clipped values are at ±3276.7
        # These are still wrong (the actual value was higher), so interpolate them
        clip_after_correction = CLIP_THRESHOLD / GYRO_AMPLIFICATION_FACTOR
        for col in corrupted_cols:
            col_data = data_df.loc[mask, col].copy()
            clipped_mask = col_data.abs() >= (clip_after_correction - 1)
            n_clipped = clipped_mask.sum()
            if n_clipped > 0:
                total_clipped_after += n_clipped
                # Replace clipped values with NaN, then interpolate
                col_data[clipped_mask] = np.nan
                col_data = col_data.interpolate(method="linear", limit_direction="both")
                # If interpolation fails (all NaN), fill with 0
                col_data = col_data.fillna(0)
                data_df.loc[mask, col] = col_data.values

    print(f"  Corrected {corrected_files} files (÷{GYRO_AMPLIFICATION_FACTOR})")
    print(f"  Clipped values found: {total_clipped_before:,}")
    print(f"  Interpolated {total_clipped_after:,} post-correction clipped samples")

    return data_df


def add_participant_column(data_df: pd.DataFrame, files_df: pd.DataFrame) -> pd.DataFrame:
    """Add participant ID column by parsing filenames."""
    id_to_participant = {}
    for _, row in files_df.iterrows():
        meta = extract_metadata_from_filename(row["filename"])
        id_to_participant[row["id"]] = meta["participant"]

    data_df["participant"] = data_df["file_id"].map(id_to_participant)
    return data_df


def compute_normalization_stats(data_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Compute per-channel mean and std from TRAINING subjects only.

    Returns:
        Dict with 'mean' and 'std' arrays of shape (n_features,)
    """
    train_mask = data_df["participant"].isin(TRAIN_SUBJECTS)
    train_data = data_df.loc[train_mask, FEATURE_COLUMNS].values.astype(np.float32)

    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)

    # Prevent division by zero for constant channels
    std[std < 1e-6] = 1.0

    print(f"Normalization stats computed from {train_mask.sum():,} training samples")
    return {"mean": mean, "std": std}


def normalize(data: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Apply z-score normalization: (x - mean) / std."""
    return (data - stats["mean"]) / stats["std"]


def create_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment continuous sensor data into fixed-length overlapping windows.

    Args:
        data: Array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,) — activity labels
        window_size: Number of samples per window
        stride: Step between consecutive windows

    Returns:
        windows: Array of shape (n_windows, window_size, n_features)
        window_labels: Array of shape (n_windows,) — majority label per window
    """
    n_samples = len(data)
    if n_samples < window_size:
        return np.empty((0, window_size, data.shape[1])), np.empty(0, dtype=int)

    n_windows = (n_samples - window_size) // stride + 1
    windows = np.zeros((n_windows, window_size, data.shape[1]), dtype=np.float32)
    window_labels = np.zeros(n_windows, dtype=np.int64)

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows[i] = data[start:end]
        # Majority vote for window label
        window_labels[i] = np.bincount(labels[start:end].astype(int)).argmax()

    return windows, window_labels


def prepare_data(
    walking_only: bool = True,
    db_path: Optional[Path] = None,
) -> dict:
    """
    Full data preparation pipeline.

    Args:
        walking_only: If True, only keep walking activity (act=1) for autoencoder
        db_path: Optional override for database path

    Returns:
        Dictionary with keys:
            'train_windows', 'train_labels',
            'val_windows', 'val_labels',
            'test_windows', 'test_labels',
            'norm_stats': normalization statistics
    """
    # Step 1: Load data
    print("=" * 60)
    print("STEP 1: Loading data from SQLite...")
    data_df, files_df = load_from_database(db_path)

    # Step 2: Correct gyroscope
    print("\nSTEP 2: Correcting gyroscope data...")
    data_df = correct_gyroscope(data_df, files_df)

    # Step 3: Add participant column
    print("\nSTEP 3: Adding participant metadata...")
    data_df = add_participant_column(data_df, files_df)

    # Step 4: Filter activity if needed
    if walking_only:
        print(f"\nSTEP 4: Filtering for walking activity (act={WALKING_ACTIVITY_ID})...")
        data_df = data_df[data_df[LABEL_COLUMN] == WALKING_ACTIVITY_ID].copy()
        print(f"  Walking samples: {len(data_df):,}")

    # Step 5: Compute normalization stats (from training set only)
    print("\nSTEP 5: Computing normalization statistics...")
    norm_stats = compute_normalization_stats(data_df)

    # Step 6: Split by participant and create windows
    print("\nSTEP 6: Creating windowed datasets...")
    result = {"norm_stats": norm_stats}

    for split_name, subjects in [
        ("train", TRAIN_SUBJECTS),
        ("val", VAL_SUBJECTS),
        ("test", TEST_SUBJECTS),
    ]:
        split_mask = data_df["participant"].isin(subjects)
        split_data = data_df[split_mask]

        if len(split_data) == 0:
            print(f"  {split_name}: 0 samples (no matching subjects)")
            result[f"{split_name}_windows"] = np.empty((0, WINDOW_SIZE, len(FEATURE_COLUMNS)))
            result[f"{split_name}_labels"] = np.empty(0, dtype=int)
            continue

        # Normalize features
        features = split_data[FEATURE_COLUMNS].values.astype(np.float32)
        features = normalize(features, norm_stats)
        labels = split_data[LABEL_COLUMN].values

        # Create windows per-file to avoid cross-file contamination
        all_windows = []
        all_labels = []

        file_ids = split_data["file_id"].unique()
        for fid in file_ids:
            fmask = split_data["file_id"].values == fid
            f_features = features[fmask]
            f_labels = labels[fmask]
            w, wl = create_windows(f_features, f_labels)
            if len(w) > 0:
                all_windows.append(w)
                all_labels.append(wl)

        if all_windows:
            all_windows = np.concatenate(all_windows, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        else:
            all_windows = np.empty((0, WINDOW_SIZE, len(FEATURE_COLUMNS)))
            all_labels = np.empty(0, dtype=int)

        result[f"{split_name}_windows"] = all_windows
        result[f"{split_name}_labels"] = all_labels

        print(f"  {split_name}: {len(all_windows):,} windows from {len(subjects)} subjects")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    total = sum(len(result[f"{s}_windows"]) for s in ["train", "val", "test"])
    print(f"Total windows: {total:,}")
    print(f"Window shape: ({WINDOW_SIZE}, {len(FEATURE_COLUMNS)})")

    return result
