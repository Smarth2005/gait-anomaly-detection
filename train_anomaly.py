"""
Pathological Gait Anomaly Detection — Main Training Script

Trains an LSTM Autoencoder on normal walking data from the HuGaDB dataset.
After training, computes anomaly scores and generates evaluation plots.

Usage:
    python train_anomaly.py
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DEVICE, SEED, WINDOW_SIZE, FEATURE_COLUMNS,
    RESULTS_DIR, CHECKPOINT_DIR,
)
from src.data.preprocessing import prepare_data
from src.data.dataset import create_dataloaders
from src.data.features import extract_features_batch, get_feature_names
from src.models.autoencoder import LSTMAutoencoder
from src.training.trainer import AnomalyTrainer, set_seed
from src.training.metrics import reconstruction_error_stats
from src.utils.visualization import (
    plot_training_history,
    plot_reconstruction_error_distribution,
    plot_reconstruction_examples,
    plot_gait_symmetry_features,
)


def main():
    print("=" * 60)
    print("  PATHOLOGICAL GAIT ANOMALY DETECTION")
    print("  LSTM Autoencoder on HuGaDB Walking Data")
    print("=" * 60)
    print(f"\nDevice: {DEVICE}")
    print(f"Seed: {SEED}")

    set_seed(SEED)

    # ================================================================
    # STEP 1: Data Preparation
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 60)

    start_time = time.time()
    prepared_data = prepare_data(walking_only=True)
    print(f"\nData preparation took {time.time() - start_time:.1f}s")

    # ================================================================
    # STEP 2: Create DataLoaders
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: CREATING DATALOADERS")
    print("=" * 60)

    dataloaders = create_dataloaders(prepared_data, augment_train=True)

    # ================================================================
    # STEP 3: Initialize Model
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: MODEL INITIALIZATION")
    print("=" * 60)

    n_features = len(FEATURE_COLUMNS)
    model = LSTMAutoencoder(
        input_dim=n_features,
        seq_len=WINDOW_SIZE,
    )
    model.summary()

    # ================================================================
    # STEP 4: Training
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: TRAINING")
    print("=" * 60)

    trainer = AnomalyTrainer(model=model, device=DEVICE)
    history = trainer.train(dataloaders["train"], dataloaders["val"])

    # ================================================================
    # STEP 5: Evaluation
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: EVALUATION")
    print("=" * 60)

    # Compute reconstruction errors for all splits
    print("\nComputing reconstruction errors...")
    train_errors = trainer.compute_reconstruction_errors(dataloaders["train"])
    val_errors = trainer.compute_reconstruction_errors(dataloaders["val"])
    test_errors = trainer.compute_reconstruction_errors(dataloaders["test"])

    print("\n--- Train Error Stats ---")
    for k, v in reconstruction_error_stats(train_errors).items():
        print(f"  {k}: {v:.6f}")

    print("\n--- Val Error Stats ---")
    for k, v in reconstruction_error_stats(val_errors).items():
        print(f"  {k}: {v:.6f}")

    print("\n--- Test Error Stats ---")
    for k, v in reconstruction_error_stats(test_errors).items():
        print(f"  {k}: {v:.6f}")

    # ================================================================
    # STEP 6: Generate Visualizations
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 6: GENERATING VISUALIZATIONS")
    print("=" * 60)

    # 6a. Training history
    print("\nPlotting training history...")
    plot_training_history(history)

    # 6b. Error distribution
    print("Plotting error distribution...")
    plot_reconstruction_error_distribution(
        train_errors, val_errors, trainer.anomaly_threshold, test_errors
    )

    # 6c. Reconstruction examples from test set
    print("Plotting reconstruction examples...")
    model.eval()
    with torch.no_grad():
        # Get a batch from test set
        test_batch = next(iter(dataloaders["test"]))
        if isinstance(test_batch, (list, tuple)):
            test_x = test_batch[0].to(DEVICE)
        else:
            test_x = test_batch.to(DEVICE)

        test_recon, _ = model(test_x)
        test_x_np = test_x.cpu().numpy()
        test_recon_np = test_recon.cpu().numpy()
        batch_errors = np.mean((test_x_np - test_recon_np) ** 2, axis=(1, 2))

    plot_reconstruction_examples(
        test_x_np, test_recon_np, batch_errors, n_examples=4
    )

    # 6d. Gait symmetry features
    print("Extracting gait symmetry features...")
    # Use un-normalized test data for feature extraction
    test_windows_raw = prepared_data["test_windows"]
    if len(test_windows_raw) > 500:
        # Subsample for speed
        idx = np.random.choice(len(test_windows_raw), 500, replace=False)
        test_windows_sub = test_windows_raw[idx]
        test_errors_sub = test_errors[idx]
    else:
        test_windows_sub = test_windows_raw
        test_errors_sub = test_errors

    gait_features = extract_features_batch(test_windows_sub)
    feature_names = get_feature_names()

    plot_gait_symmetry_features(
        gait_features, feature_names, test_errors_sub, trainer.anomaly_threshold
    )

    # ================================================================
    # DONE
    # ================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Best model saved to: {CHECKPOINT_DIR / 'best_model.pt'}")
    print(f"Anomaly threshold: {trainer.anomaly_threshold:.6f}")

    # Print some sample anomaly detections
    n_anomalies_test = (test_errors >= trainer.anomaly_threshold).sum()
    print(f"\nTest set: {n_anomalies_test}/{len(test_errors)} windows flagged as anomalous "
          f"({n_anomalies_test / len(test_errors) * 100:.1f}%)")


if __name__ == "__main__":
    main()
