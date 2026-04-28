"""
LSTM Autoencoder for gait anomaly detection.

Architecture:
    Input (128 × 38) → LSTM Encoder → Bottleneck (32-dim) → LSTM Decoder → Reconstructed (128 × 38)

Trained on normal walking data. At inference, high reconstruction error
indicates anomalous/pathological gait.
"""

import torch
import torch.nn as nn
from src.config import (
    ENCODER_HIDDEN_DIM, BOTTLENECK_DIM, DECODER_HIDDEN_DIM,
    NUM_LSTM_LAYERS, DROPOUT,
)


class LSTMEncoder(nn.Module):
    """
    LSTM-based encoder that compresses a sequence into a fixed-length latent vector.

    Input:  (batch, seq_len, n_features)
    Output: (batch, bottleneck_dim)
    """

    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Project final hidden state to bottleneck
        self.fc = nn.Linear(hidden_dim, bottleneck_dim)
        self.bn = nn.BatchNorm1d(bottleneck_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            z: (batch, bottleneck_dim) — latent representation
        """
        # LSTM processes the full sequence
        _, (h_n, _) = self.lstm(x)

        # Use the last layer's hidden state
        # h_n shape: (num_layers, batch, hidden_dim)
        h_last = h_n[-1]  # (batch, hidden_dim)

        # Project to bottleneck
        z = self.fc(h_last)
        z = self.bn(z)

        return z


class LSTMDecoder(nn.Module):
    """
    LSTM-based decoder that reconstructs a sequence from a latent vector.

    Input:  (batch, bottleneck_dim)
    Output: (batch, seq_len, n_features)
    """

    def __init__(self, output_dim: int, hidden_dim: int, bottleneck_dim: int,
                 seq_len: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Project bottleneck to initial input for decoder LSTM
        self.fc_in = nn.Linear(bottleneck_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Project LSTM output to original feature dimension
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        """
        Args:
            z: (batch, bottleneck_dim)
        Returns:
            reconstruction: (batch, seq_len, output_dim)
        """
        # Expand latent vector to sequence length
        # Repeat the projected latent vector for each timestep
        h = self.fc_in(z)  # (batch, hidden_dim)
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)

        # Decode sequence
        lstm_out, _ = self.lstm(h)  # (batch, seq_len, hidden_dim)

        # Project to output dimension
        reconstruction = self.fc_out(lstm_out)  # (batch, seq_len, output_dim)

        return reconstruction


class LSTMAutoencoder(nn.Module):
    """
    Complete LSTM Autoencoder for gait anomaly detection.

    Compresses walking sensor windows into a fixed-length latent code,
    then reconstructs them. Normal walking patterns are reconstructed
    accurately; anomalous patterns yield high reconstruction error.
    """

    def __init__(
        self,
        input_dim: int = 38,
        seq_len: int = 128,
        encoder_hidden: int = ENCODER_HIDDEN_DIM,
        bottleneck_dim: int = BOTTLENECK_DIM,
        decoder_hidden: int = DECODER_HIDDEN_DIM,
        num_layers: int = NUM_LSTM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.bottleneck_dim = bottleneck_dim

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden,
            bottleneck_dim=bottleneck_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder = LSTMDecoder(
            output_dim=input_dim,
            hidden_dim=decoder_hidden,
            bottleneck_dim=bottleneck_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) — input sensor windows

        Returns:
            reconstruction: (batch, seq_len, input_dim) — reconstructed windows
            z: (batch, bottleneck_dim) — latent representation
        """
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z

    def encode(self, x):
        """Get only the latent representation."""
        return self.encoder(x)

    def compute_anomaly_score(self, x):
        """
        Compute per-window reconstruction error as anomaly score.

        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            scores: (batch,) — MSE per window (higher = more anomalous)
        """
        self.eval()
        with torch.no_grad():
            reconstruction, _ = self.forward(x)
            # Per-window MSE
            mse = torch.mean((x - reconstruction) ** 2, dim=(1, 2))
        return mse

    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print model summary."""
        print(f"LSTM Autoencoder")
        print(f"  Input:      ({self.seq_len}, {self.input_dim})")
        print(f"  Bottleneck: {self.bottleneck_dim}")
        print(f"  Parameters: {self.get_num_params():,}")
        print(f"  Encoder:    {sum(p.numel() for p in self.encoder.parameters()):,} params")
        print(f"  Decoder:    {sum(p.numel() for p in self.decoder.parameters()):,} params")
