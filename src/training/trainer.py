"""
Training loop for the LSTM Autoencoder.

Includes:
- Training with reconstruction loss
- Validation monitoring
- Early stopping
- Model checkpointing
- Anomaly score computation
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm

from src.config import (
    DEVICE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, SCHEDULER_T0, CHECKPOINT_DIR,
    ANOMALY_PERCENTILE, SEED,
)
from src.training.losses import ReconstructionLoss
from src.training.metrics import reconstruction_error_stats, compute_anomaly_threshold


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AnomalyTrainer:
    """
    Trainer for the LSTM Autoencoder gait anomaly detector.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = DEVICE,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        num_epochs: int = NUM_EPOCHS,
        patience: int = EARLY_STOPPING_PATIENCE,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR

        # Loss function
        self.criterion = ReconstructionLoss()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=SCHEDULER_T0, T_mult=2
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        # Anomaly threshold (set after training)
        self.anomaly_threshold = None

    def train(self, train_loader, val_loader) -> Dict:
        """
        Full training loop with early stopping and checkpointing.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader

        Returns:
            Training history dictionary
        """
        set_seed()

        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        print(f"\nTraining on {self.device}")
        print(f"  Model params: {self.model.get_num_params():,}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Early stopping patience: {self.patience}")
        print("=" * 60)

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

            # --- Training ---
            train_loss = self._train_epoch(train_loader)

            # --- Validation ---
            val_loss = self._validate(val_loader)

            # --- LR Scheduling ---
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --- Record history ---
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - epoch_start

            # --- Logging ---
            print(
                f"Epoch {epoch:3d}/{self.num_epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s",
                end=""
            )

            # --- Early stopping & checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint("best_model.pt", epoch, val_loss)
                print(" * (best)", end="")
            else:
                patience_counter += 1

            print()

            if patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}! "
                      f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
                break

        # Load best model
        self._load_checkpoint("best_model.pt")

        # Compute anomaly threshold from validation set
        print("\nComputing anomaly threshold from validation set...")
        val_scores = self.compute_reconstruction_errors(val_loader)
        self.anomaly_threshold = compute_anomaly_threshold(
            val_scores, ANOMALY_PERCENTILE
        )

        stats = reconstruction_error_stats(val_scores)
        print(f"  Val reconstruction error stats:")
        for k, v in stats.items():
            print(f"    {k}: {v:.6f}")
        print(f"  Anomaly threshold (p{ANOMALY_PERCENTILE}): {self.anomaly_threshold:.6f}")

        return self.history

    def _train_epoch(self, train_loader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            # Unpack — for autoencoder, input = target
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)

            # Forward pass
            reconstruction, z = self.model(x)
            loss = self.criterion(reconstruction, x)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate(self, val_loader) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)

                reconstruction, z = self.model(x)
                loss = self.criterion(reconstruction, x)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def compute_reconstruction_errors(self, data_loader) -> np.ndarray:
        """
        Compute per-window reconstruction errors for anomaly scoring.

        Returns:
            Array of shape (n_samples,) — MSE per window
        """
        self.model.eval()
        all_errors = []

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)

                reconstruction, _ = self.model(x)
                errors = self.criterion.per_sample_loss(reconstruction, x)
                all_errors.append(errors.cpu().numpy())

        return np.concatenate(all_errors)

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "anomaly_threshold": self.anomaly_threshold,
            "history": self.history,
        }, path)

    def _load_checkpoint(self, filename: str):
        """Load model from checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "anomaly_threshold" in checkpoint:
            self.anomaly_threshold = checkpoint["anomaly_threshold"]
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
              f"(val_loss: {checkpoint['val_loss']:.6f})")
