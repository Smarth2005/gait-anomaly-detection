"""
PyTorch Dataset classes for gait anomaly detection.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict

from src.config import BATCH_SIZE, SEED


class GaitWindowDataset(Dataset):
    """
    Dataset for windowed gait sensor data.

    Each sample is a fixed-length window of multi-channel sensor readings.
    Used for both autoencoder training (input = target) and inference.

    Args:
        windows: Array of shape (n_windows, window_size, n_features)
        labels: Optional array of shape (n_windows,) — activity labels
        transform: Optional callable for data augmentation
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: Optional[np.ndarray] = None,
        transform=None,
    ):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]  # (window_size, n_features)

        if self.transform is not None:
            window = self.transform(window)

        if self.labels is not None:
            return window, self.labels[idx]
        return window


class TimeSeriesAugmentation:
    """
    Data augmentation for time series sensor data.
    Applied during training to improve generalization.
    """

    def __init__(self, jitter_std=0.05, scale_range=(0.9, 1.1), p=0.5):
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        # Jittering: add small Gaussian noise
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(x) * self.jitter_std
            x = x + noise

        # Scaling: multiply by random factor
        if torch.rand(1).item() < 0.5:
            scale = torch.empty(1).uniform_(*self.scale_range).item()
            x = x * scale

        return x


def create_dataloaders(
    prepared_data: dict,
    batch_size: int = BATCH_SIZE,
    augment_train: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders from prepared windowed data.

    Args:
        prepared_data: Output of preprocessing.prepare_data()
        batch_size: Batch size for all splits
        augment_train: Whether to apply augmentation to training data

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    transform = TimeSeriesAugmentation() if augment_train else None

    # Set random seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(SEED)

    dataloaders = {}

    for split in ["train", "val", "test"]:
        windows = prepared_data[f"{split}_windows"]
        labels = prepared_data[f"{split}_labels"]

        dataset = GaitWindowDataset(
            windows=windows,
            labels=labels,
            transform=transform if split == "train" else None,
        )

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,  # Windows compatibility
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
            generator=generator if split == "train" else None,
        )

        print(f"  {split} DataLoader: {len(dataset)} samples, "
              f"{len(dataloaders[split])} batches")

    return dataloaders
