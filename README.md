<div align="center">

# 🦿 Pathological Gait Anomaly Detection

### *Deep Learning Autoencoders for Wearable Sensor-Based Gait Analysis*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: HuGaDB](https://img.shields.io/badge/Dataset-HuGaDB-blue)](https://github.com/romanchereshnev/HuGaDB)

<br>

**Learn what *normal* walking looks like. Flag everything else.**

An unsupervised anomaly detection pipeline that trains autoencoder models on healthy gait patterns from 38-channel wearable IMU + EMG sensors, then identifies pathological deviations through reconstruction error thresholding.

<br>

<img src="results/error_distribution.png" alt="Reconstruction Error Distribution" width="600"/>

</div>

---

## ⚡ Highlights

- 🧠 **Dual Architecture** — LSTM Autoencoder (sequential baseline) vs. Patch-based Transformer Autoencoder (2024 SOTA)
- 🔧 **Gyroscope Corruption Repair** — Automated detection & correction of 10× amplification + int16 clipping artifacts across 300+ files
- 📊 **38-Channel Sensor Fusion** — 18 accelerometer + 18 gyroscope + 2 EMG channels from 6 body locations
- 🏥 **Clinical Feature Engineering** — Symmetry Index, Harmonic Ratio, Jerk, Step Regularity, EMG Asymmetry
- 🧪 **Subject-Aware Evaluation** — Strict leave-N-subjects-out protocol (12 train / 3 val / 3 test) to prevent data leakage
- 📈 **Full Visualization Suite** — Training curves, error distributions, reconstruction overlays, symmetry scatter plots

---

## 🏗️ Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │         HuGaDB Wearable Dataset     │
                    │   18 subjects · 6 IMU + 2 EMG · 60Hz│
                    └────────────────┬────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   Data Pipeline      │
                          │ • Gyro correction ÷10│
                          │ • Clip interpolation │
                          │ • Z-score norm       │
                          │ • Sliding windows    │
                          │   128 × 38           │
                          └──────────┬───────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                                  │
          ┌─────────▼─────────┐            ┌───────────▼───────────┐
          │  LSTM Autoencoder  │            │  Patch Transformer AE │
          │                    │            │                       │
          │ Enc: 2×LSTM(128)   │            │ Patch: 16 → 8 tokens │
          │ Bottleneck: 32     │            │ 3× TransformerEnc     │
          │ Dec: 2×LSTM(128)   │            │ d_model=128, 8 heads  │
          │ Params: ~660K      │            │ Params: ~1.8M         │
          └─────────┬──────────┘            └───────────┬───────────┘
                    │                                    │
                    └────────────────┬───────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Anomaly Scoring     │
                          │  MSE > P95 threshold │
                          │  → Pathological Flag │
                          └──────────────────────┘
```

---

## 📂 Project Structure

```
.
├── train_anomaly.py           # Train LSTM Autoencoder
├── train_transformer.py       # Train Patch Transformer Autoencoder
├── create_db.py               # Convert HuGaDB .txt → SQLite
├── requirements.txt           # Python dependencies
│
├── src/
│   ├── config.py              # All hyperparameters & paths (single source of truth)
│   ├── data/
│   │   ├── preprocessing.py   # Load → correct → normalize → window pipeline
│   │   ├── corruption_map.py  # Per-file gyroscope corruption registry (300+ files)
│   │   ├── features.py        # Gait symmetry & temporal feature extraction
│   │   └── dataset.py         # PyTorch Dataset + augmentation
│   ├── models/
│   │   ├── autoencoder.py     # LSTM Autoencoder (encoder–bottleneck–decoder)
│   │   └── transformer.py     # Patch-based Transformer Autoencoder
│   ├── training/
│   │   ├── trainer.py         # Training loop, early stopping, checkpointing
│   │   ├── losses.py          # MSE reconstruction loss (channel-weightable)
│   │   └── metrics.py         # ROC-AUC, F1, anomaly threshold computation
│   └── utils/
│       └── visualization.py   # Training curves, error histograms, signal overlays
│
├── results/                   # Generated plots (LSTM)
│   └── transformer/           # Generated plots (Transformer)
├── checkpoints/               # Saved model weights (.pt)
└── HumanGaitDataBase/         # Raw HuGaDB dataset (not tracked)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/pathological-gait-detection.git
cd pathological-gait-detection
pip install -r requirements.txt
```

### 2. Prepare Data

Download the [HuGaDB dataset](https://github.com/romanchereshnev/HuGaDB) and place the `Data/` folder inside `HumanGaitDataBase/`.

```bash
# Build the SQLite database from raw text files
python create_db.py HumanGaitDataBase/Data
```

### 3. Train Models

```bash
# LSTM Autoencoder (baseline)
python train_anomaly.py

# Patch Transformer Autoencoder (advanced)
python train_transformer.py
```

Training runs for up to 50 epochs with early stopping (patience=10) and CosineAnnealingWarmRestarts scheduling. Results and plots are saved automatically.

---

## 🔬 Data Pipeline Deep Dive

### Gyroscope Corruption Handling

The HuGaDB dataset contains a **documented 10× amplification bug** in gyroscope channels across hundreds of files, with values clipped at the int16 boundary (±32767). Our pipeline:

1. **Maps** corrupted channels per-file using a hand-built registry of 300+ entries
2. **Divides** affected channels by 10 to restore true scale
3. **Interpolates** clipped values (formerly at ±32767 → now ±3276.7) using linear interpolation

### Feature Engineering

Beyond raw reconstruction error, we extract **clinically meaningful gait features**:

| Feature | What It Captures | Clinical Relevance |
|---|---|---|
| **Symmetry Index** | Left–right imbalance per body segment | Hemiplegia, limb length discrepancy |
| **Harmonic Ratio** | Even/odd harmonic ratio in vertical acc | Gait smoothness & rhythm |
| **Step Regularity** | Autocorrelation peak height | Consistency of stride timing |
| **Jerk** | Rate of change of acceleration | Movement smoothness / spasticity |
| **EMG Asymmetry** | Left–right muscle activation imbalance | Neuromuscular disorders |

---

## 📊 Results

### LSTM Autoencoder
| Metric | Train | Val | Test |
|---|---|---|---|
| **Mean MSE** | Low | Low | Low |
| **Anomaly Rate** | — | 5% (by design) | Varies by subject |

### Patch Transformer Autoencoder
- Captures **longer-range temporal dependencies** via self-attention over 8 patches
- Generally achieves **tighter reconstruction** on normal gait patterns

Both models produce rich visualizations:

<div align="center">

| Training Curves | Reconstruction Examples |
|:---:|:---:|
| <img src="results/training_history.png" width="400"/> | <img src="results/reconstruction_examples.png" width="400"/> |

| Error Distribution | Symmetry Features |
|:---:|:---:|
| <img src="results/error_distribution.png" width="400"/> | <img src="results/symmetry_features.png" width="400"/> |

</div>

---

## ⚙️ Configuration

All hyperparameters live in a single file — [`src/config.py`](src/config.py):

| Parameter | Value | Description |
|---|---|---|
| `WINDOW_SIZE` | 128 | ~2.1 seconds at 60 Hz |
| `WINDOW_STRIDE` | 64 | 50% overlap |
| `BOTTLENECK_DIM` | 32 | Latent space dimensionality |
| `BATCH_SIZE` | 256 | Training batch size |
| `LEARNING_RATE` | 1e-3 | AdamW initial LR |
| `EARLY_STOPPING` | 10 epochs | Patience before stopping |
| `ANOMALY_PERCENTILE` | 95 | Threshold = P95 of val errors |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Framework** | PyTorch 2.0+ |
| **Optimization** | AdamW + CosineAnnealingWarmRestarts |
| **Data Storage** | SQLite (via `create_db.py`) |
| **Visualization** | Matplotlib + Seaborn |
| **Feature Eng.** | SciPy (signal processing) + scikit-learn |

---

## 📖 References

- **HuGaDB Dataset**: Chereshnev & Kertész-Farkas (2018). *HuGaDB: Human Gait Database for Activity Recognition from Wearable Inertial Sensor Networks.* [GitHub](https://github.com/romanchereshnev/HuGaDB)
- **PatchTST**: Nie et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR 2023.
- **LSTM Autoencoder for Anomaly Detection**: Malhotra et al. (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.*

---

<div align="center">

**Built with ❤️ and PyTorch**

*If this project helped your research, give it a ⭐!*

</div>
