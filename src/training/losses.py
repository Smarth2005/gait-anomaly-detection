"""
Loss functions for the gait anomaly detection pipeline.
"""

import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """
    MSE reconstruction loss for the autoencoder.

    Computes per-sample and per-channel MSE between input and reconstruction,
    with optional channel-wise weighting to emphasize certain sensor channels.
    """

    def __init__(self, channel_weights=None):
        """
        Args:
            channel_weights: Optional tensor of shape (n_features,) to weight
                             different sensor channels differently.
                             E.g., weight gyroscope channels lower if less reliable.
        """
        super().__init__()
        self.channel_weights = channel_weights

    def forward(self, reconstruction, target):
        """
        Args:
            reconstruction: (batch, seq_len, n_features)
            target: (batch, seq_len, n_features)

        Returns:
            loss: scalar — mean reconstruction error across batch
        """
        error = (reconstruction - target) ** 2  # (batch, seq_len, n_features)

        if self.channel_weights is not None:
            # Apply per-channel weights
            weights = self.channel_weights.to(error.device)
            error = error * weights.unsqueeze(0).unsqueeze(0)

        # Mean across all dimensions
        loss = error.mean()
        return loss

    def per_sample_loss(self, reconstruction, target):
        """
        Compute per-sample reconstruction error (for anomaly scoring).

        Returns:
            (batch,) — MSE per window
        """
        error = (reconstruction - target) ** 2
        if self.channel_weights is not None:
            weights = self.channel_weights.to(error.device)
            error = error * weights.unsqueeze(0).unsqueeze(0)
        return error.mean(dim=(1, 2))
