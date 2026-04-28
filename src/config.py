"""
Central configuration for the Pathological Gait Anomaly Detection pipeline.
All hyperparameters, column definitions, activity mappings, and path constants.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "HuGaDB.db"
DATA_DIR = PROJECT_ROOT / "HumanGaitDataBase" / "Data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# SENSOR COLUMN DEFINITIONS
# ============================================================================

# Accelerometer columns (always clean — no corruption)
ACC_COLUMNS = [
    "acc_rf_x", "acc_rf_y", "acc_rf_z",   # Right Foot
    "acc_rs_x", "acc_rs_y", "acc_rs_z",   # Right Shank
    "acc_rt_x", "acc_rt_y", "acc_rt_z",   # Right Thigh
    "acc_lf_x", "acc_lf_y", "acc_lf_z",   # Left Foot
    "acc_ls_x", "acc_ls_y", "acc_ls_z",   # Left Shank
    "acc_lt_x", "acc_lt_y", "acc_lt_z",   # Left Thigh
]

# Gyroscope columns (subject to 10x amplification corruption)
GYRO_COLUMNS = [
    "gyro_rf_x", "gyro_rf_y", "gyro_rf_z",  # Right Foot
    "gyro_rs_x", "gyro_rs_y", "gyro_rs_z",  # Right Shank
    "gyro_rt_x", "gyro_rt_y", "gyro_rt_z",  # Right Thigh
    "gyro_lf_x", "gyro_lf_y", "gyro_lf_z",  # Left Foot
    "gyro_ls_x", "gyro_ls_y", "gyro_ls_z",  # Left Shank
    "gyro_lt_x", "gyro_lt_y", "gyro_lt_z",  # Left Thigh
]

# EMG columns (always clean)
EMG_COLUMNS = ["EMG_r", "EMG_l"]

# All sensor feature columns (input to model)
FEATURE_COLUMNS = ACC_COLUMNS + GYRO_COLUMNS + EMG_COLUMNS  # 38 total

# Label column
LABEL_COLUMN = "act"

# Gyroscope channel groups by body location (for per-location corruption handling)
GYRO_GROUPS = {
    "RF": ["gyro_rf_x", "gyro_rf_y", "gyro_rf_z"],
    "RS": ["gyro_rs_x", "gyro_rs_y", "gyro_rs_z"],
    "RT": ["gyro_rt_x", "gyro_rt_y", "gyro_rt_z"],
    "LF": ["gyro_lf_x", "gyro_lf_y", "gyro_lf_z"],
    "LS": ["gyro_ls_x", "gyro_ls_y", "gyro_ls_z"],
    "LT": ["gyro_lt_x", "gyro_lt_y", "gyro_lt_z"],
}

# ============================================================================
# ACTIVITY MAPPINGS
# ============================================================================

ACTIVITY_MAP = {
    1: "walking",
    2: "running",
    3: "going_upstairs",
    4: "going_downstairs",
    5: "sitting",
    6: "sitting_down",
    7: "standing_up",
    8: "standing",
    9: "bicycling",
    10: "up_by_elevator",
    11: "down_by_elevator",
    12: "sitting_in_car",
}

# Activities relevant to gait analysis
GAIT_ACTIVITIES = [1, 2, 3, 4]  # walking, running, stairs up/down
WALKING_ACTIVITY_ID = 1

# ============================================================================
# DATA PIPELINE PARAMETERS
# ============================================================================

SAMPLING_RATE = 60  # Hz (approximate)
WINDOW_SIZE = 128   # samples (~2.13 seconds at 60Hz, ≈ 2 gait cycles)
WINDOW_STRIDE = 64  # 50% overlap

# Gyroscope corruption parameters
GYRO_AMPLIFICATION_FACTOR = 10  # corrupted data was amplified 10x
CLIP_THRESHOLD = 32767          # int16 max — values at this are clipped

# ============================================================================
# SUBJECT SPLITS (Leave-N-Subjects-Out)
# ============================================================================

# 18 participants total
TRAIN_SUBJECTS = list(range(1, 13))   # Subjects 1-12
VAL_SUBJECTS = list(range(13, 16))    # Subjects 13-15
TEST_SUBJECTS = list(range(16, 19))   # Subjects 16-18

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# LSTM Autoencoder (Base Model)
ENCODER_HIDDEN_DIM = 128
BOTTLENECK_DIM = 32
DECODER_HIDDEN_DIM = 128
NUM_LSTM_LAYERS = 2
DROPOUT = 0.3

# Transformer Autoencoder (Advanced 2024 Model)
PATCH_SIZE = 16  # Window size (128) must be divisible by PATCH_SIZE
TRANSFORMER_D_MODEL = 128
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 3
TRANSFORMER_DIM_FEEDFORWARD = 256

# Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_T0 = 10  # CosineAnnealingWarmRestarts period

# Anomaly detection
ANOMALY_PERCENTILE = 95  # threshold = 95th percentile of val reconstruction error

# ============================================================================
# DEVICE
# ============================================================================

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# RANDOM SEED
# ============================================================================
SEED = 42
